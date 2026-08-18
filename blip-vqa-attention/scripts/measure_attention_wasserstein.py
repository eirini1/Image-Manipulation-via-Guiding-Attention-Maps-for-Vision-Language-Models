import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    import ot  
except Exception:  
    ot = None

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import (
    apply_override,
    data_path as DATA_PATH,
    ensure_mask,
    guess_focus_words,
    image_root as IMAGE_ROOT,
    iter_jsonl,
    masks_root,
    parse_heads,
    revert_override,
    select_override_indices,
)

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root
output = Path("att_wasserstein/focused_wasserstein.jsonl")


def _normalize_id(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    return text.lower()


def _sanitize_name(text: Optional[str]) -> str:
    text = "" if text is None else str(text)
    cleaned = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in text)
    return cleaned[:80]


def _load_mask_from_dir(
    mask_dir: Path,
    image_stem: str,
    prompt: str,
    rec_id: Optional[str],
) -> Optional[np.ndarray]:
    sanitized = _sanitize_name(prompt)
    patterns = []
    if rec_id:
        patterns.append(f"{rec_id}_{image_stem}_{sanitized}_*.npy")
    patterns.append(f"*_{image_stem}_{sanitized}_*.npy")

    for pattern in patterns:
        candidates = sorted(mask_dir.glob(pattern))
        if not candidates:
            continue
        path = candidates[-1]
        try:
            arr = np.load(path)
            if arr.ndim > 2:
                arr = arr.squeeze()
            return arr.astype(np.float32)
        except Exception:
            continue
    return None


def _gaussian_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Fit a 2D Gaussian to the mask mass and return a unit-peak weighting."""
    height, width = mask.shape[-2], mask.shape[-1]
    if float(mask.max().item()) <= 0.0:
        return torch.zeros_like(mask)

    ys = torch.arange(height, device=mask.device, dtype=mask.dtype)
    xs = torch.arange(width, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    total = mask.sum()
    if float(total.item()) <= 0.0:
        return torch.zeros_like(mask)

    weights = mask / total
    mu_x = (xx * weights).sum()
    mu_y = (yy * weights).sum()

    var_x = ((xx - mu_x) ** 2 * weights).sum()
    var_y = ((yy - mu_y) ** 2 * weights).sum()
    sigma_x = torch.sqrt(torch.clamp(var_x, min=1.0))
    sigma_y = torch.sqrt(torch.clamp(var_y, min=1.0))

    gx = (xx - mu_x) / sigma_x
    gy = (yy - mu_y) / sigma_y
    gauss = torch.exp(-0.5 * (gx * gx + gy * gy))
    return gauss / gauss.max().clamp_min(1e-6)


def _to_numpy_uint8(mask_hw: torch.Tensor) -> np.ndarray:
    x = mask_hw.detach().float().clamp(0, 1).cpu().numpy()
    return (x * 255.0).astype(np.uint8)


def _to_torch_float(mask_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device)


def translate_mask(mask_hw: torch.Tensor, tx: int = 0, ty: int = 0, border_value: float = 0.0) -> torch.Tensor:
    if tx == 0 and ty == 0:
        return mask_hw
    h, w = mask_hw.shape[-2], mask_hw.shape[-1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    src = _to_numpy_uint8(mask_hw)
    dst = cv2.warpAffine(
        src,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.clip(border_value, 0, 1) * 255),
    )
    return _to_torch_float(dst, mask_hw.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply attention overrides and compute per-layer/head Wasserstein distance for focus tokens."
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer spec: 'all' or comma-separated indices.",
    )
    parser.add_argument(
        "--heads",
        default="all",
        help="Head spec: 'all' or comma-separated indices.",
    )
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="Blend a fitted Gaussian into the softened mask before overriding.",
    )
    parser.add_argument(
        "--keep-cls",
        action="store_true",
        help="Keep the image CLS token when computing Wasserstein distance (default drops it).",
    )
    parser.add_argument(
        "--pairwise-wasserstein",
        action="store_true",
        help="Compute Wasserstein directly between baseline and override attentions instead of using the mask as reference.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate Wasserstein heatmaps for each token.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("att_wasserstein/plots"),
        help="Directory where token Wasserstein heatmap images will be stored when --save-plots is set.",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help=(
            "Restrict processing to records whose id/image stem/path matches the given value. "
            "Can be supplied multiple times or as comma-separated values."
        ),
    )
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--ty", type=int, default=0)
    return parser.parse_args()



def _build_cost_matrix(grid_h: int, grid_w: int, include_cls: bool) -> np.ndarray:
    coords: List[Tuple[float, float]] = []
    if include_cls:
        coords.append((-1.0, -1.0))  # pseudo coordinate for CLS
    for y in range(grid_h):
        for x in range(grid_w):
            coords.append((float(y), float(x)))
    coords_np = np.array(coords, dtype=np.float32)
    diff = coords_np[:, None, :] - coords_np[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def compute_wasserstein_tensor_2d(
    cross_attentions: Sequence[torch.Tensor],
    reference_distribution: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    drop_image_cls: bool = True,
) -> Tuple[np.ndarray, float]:

    if ot is None:
        raise RuntimeError(
            "Python Optimal Transport (POT) is required for 2D OT. "
            "Install with `pip install pot`."
        )
    if not cross_attentions:
        raise ValueError("Cross-attention tensors are required.")

    reference = cross_attentions[0]
    _, _, _, num_keys = reference.shape

    expected_grid = grid_h * grid_w
    if drop_image_cls:
        valid_key_counts = {expected_grid, expected_grid + 1}  # allow optional image CLS
    else:
        valid_key_counts = {expected_grid + 1}

    if num_keys not in valid_key_counts:
        raise ValueError(
            f"Cross-attention keys ({num_keys}) do not match expected grid counts {sorted(valid_key_counts)}."
        )

    has_image_cls = num_keys == expected_grid + 1
    key_start = 1 if (drop_image_cls and has_image_cls) else 0
    num_effective_keys = num_keys - key_start
    if num_effective_keys <= 0:
        raise ValueError("Cannot compute Wasserstein with zero effective key tokens.")

    if reference_distribution.numel() != num_effective_keys:
        raise ValueError(
            f"Reference distribution length {reference_distribution.numel()} "
            f"does not match effective keys {num_effective_keys}."
        )

    cost_matrix = _build_cost_matrix(grid_h, grid_w, include_cls=has_image_cls)
    if drop_image_cls and has_image_cls:
        cost_matrix = cost_matrix[1:, 1:]
    max_distance = float(cost_matrix.max())

    ref_np = reference_distribution.detach().float().cpu().numpy()
    ref_np = ref_np / max(ref_np.sum(), 1e-12)

    layers_out: List[np.ndarray] = []
    for layer_tensor in cross_attentions:
        attn = layer_tensor[0][:, :, key_start:]
        probs = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs.clamp_min(1e-12).detach().cpu().numpy()

        layer_scores = np.zeros(probs.shape[:2], dtype=np.float32)
        for h in range(probs.shape[0]):
            for q in range(probs.shape[1]):
                target = probs[h, q]
                target = target / max(target.sum(), 1e-12)
                layer_scores[h, q] = float(ot.emd2(ref_np, target, cost_matrix))
        layers_out.append(layer_scores)

    return np.stack(layers_out, axis=0), max_distance


def compute_wasserstein_between_runs_2d(
    baseline_cross_attentions: Sequence[torch.Tensor],
    override_cross_attentions: Sequence[torch.Tensor],
    *,
    grid_h: int,
    grid_w: int,
    drop_image_cls: bool = True,
) -> Tuple[np.ndarray, float]:

    if ot is None:
        raise RuntimeError(
            "Python Optimal Transport (POT) is required for 2D OT. "
            "Install with `pip install pot`."
        )
    if not baseline_cross_attentions or not override_cross_attentions:
        raise ValueError("Both baseline and override cross-attention tensors are required.")
    if len(baseline_cross_attentions) != len(override_cross_attentions):
        raise ValueError("Baseline and override attention sequences must have the same length.")

    num_layers = len(baseline_cross_attentions)
    layers_out: List[np.ndarray] = []
    max_distance: float = 0.0

    for idx in range(num_layers):
        base_layer = baseline_cross_attentions[idx]
        over_layer = override_cross_attentions[idx]
        if base_layer.shape != over_layer.shape:
            raise ValueError(f"Shape mismatch at layer {idx}: {base_layer.shape} vs {over_layer.shape}")

        _, _, _, num_keys = base_layer.shape
        expected_grid = grid_h * grid_w
        has_image_cls = num_keys == expected_grid + 1
        key_start = 1 if (drop_image_cls and has_image_cls) else 0
        num_effective_keys = num_keys - key_start
        if num_effective_keys <= 0:
            raise ValueError("Cannot compute Wasserstein with zero effective key tokens.")

        base = base_layer[0][:, :, key_start:]
        over = over_layer[0][:, :, key_start:]

        cost_matrix = _build_cost_matrix(grid_h, grid_w, include_cls=has_image_cls)
        if drop_image_cls and has_image_cls:
            cost_matrix = cost_matrix[1:, 1:]
        max_distance = max(max_distance, float(cost_matrix.max()))

        base_probs = base / base.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        over_probs = over / over.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        base_probs = base_probs.clamp_min(1e-12).detach().cpu().numpy()
        over_probs = over_probs.clamp_min(1e-12).detach().cpu().numpy()

        layer_scores = np.zeros(base_probs.shape[:2], dtype=np.float32)
        for h in range(base_probs.shape[0]):
            for q in range(base_probs.shape[1]):
                ref = base_probs[h, q]
                tgt = over_probs[h, q]
                ref = ref / max(ref.sum(), 1e-12)
                tgt = tgt / max(tgt.sum(), 1e-12)
                layer_scores[h, q] = float(ot.emd2(ref, tgt, cost_matrix))
        layers_out.append(layer_scores)

    return np.stack(layers_out, axis=0), max_distance


def resolve_image_path(image_value: str, image_root: Path) -> Path:
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    return image_path


def _print_record_summary(
    record_id: str,
    image_value: str,
    question: str,
    record_tokens: Sequence[Dict[str, object]],
    wass_mode: str = "mask",
) -> None:
    print(f"\n[record] id={record_id}  image={image_value}")
    print(f"question: {question}")
    print(f"Wasserstein mode: {wass_mode}")
    print(" idx | override | token                 | base_W | new_W | delta_W ")
    print("-----+----------+-----------------------+--------+-------+---------")
    for tok in record_tokens:
        token_text = str(tok["token"])
        mark = "*" if tok.get("overridden") else " "
        print(
            f"{tok['index']:>4} |    {mark}     | {token_text:<21} | "
            f"{tok['baseline_wasserstein_mean']:.4f} | {tok['override_wasserstein_mean']:.4f} | "
            f"{tok['delta_wasserstein_mean']:+.4f}"
        )


def _format_matrix(values: np.ndarray, precision: int = 4) -> List[str]:
    rows: List[str] = []
    for row in values:
        rows.append(" ".join(f"{float(cell):.{precision}f}" for cell in row))
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_ids: Optional[Set[str]] = None
    if getattr(args, "record_ids", None):
        collected: Set[str] = set()
        for raw in args.record_ids:
            if not raw:
                continue
            for piece in str(raw).split(","):
                norm = _normalize_id(piece)
                if norm:
                    collected.add(norm)
        if collected:
            target_ids = collected

    records = list(iter_jsonl(DATA_PATH))
    if not records:
        print("No records to process.")
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base")
    model.eval()
    model = model.to(device)
    tokenizer = model.tokenizer

    heads_spec = parse_heads(args.heads)
    layers_spec = parse_heads(args.layers)

    plots_dir: Optional[Path] = None
    if args.save_plots:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        module = layer.crossattention.self
        originals.append((module, module.forward, getattr(module, "save_attention", False)))

    if layers_spec == (-1,):
        target_layers = tuple(range(total_layers))
    else:
        target_layers = tuple(int(idx) for idx in layers_spec)

    heads_arg = heads_spec[0] if len(heads_spec) == 1 else heads_spec

    output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    skipped_missing = 0
    skipped_fail = 0
    running_baseline_w: List[float] = []
    running_override_w: List[float] = []
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}

    found_targets: Set[str] = set()

    with output.open("w", encoding="utf-8") as writer:
        for idx, entry in enumerate(records, 1):
            processed += 1

            image_value = entry.get("image")
            question = entry.get("question")
            if not image_value or not question:
                if target_ids:
                    rec_id_raw = entry.get("id")
                    rec_key = _normalize_id(rec_id_raw) or _normalize_id(image_value)
                    if rec_key in target_ids:
                        print(f"[warn] Skipping requested record '{rec_id_raw or image_value}': missing image/question.")
                skipped_missing += 1
                continue

            prompt = entry.get("prompt") or ""
            rec_id = entry.get("id")
            image_rel = Path(image_value)
            image_path = resolve_image_path(image_value, IMAGE_ROOT)
            rec_id_norm = _normalize_id(rec_id)
            image_stem_norm = _normalize_id(image_rel.stem)
            image_value_norm = _normalize_id(image_value)

            is_target = True
            if target_ids:
                is_target = (
                    rec_id_norm in target_ids
                    or image_stem_norm in target_ids
                    or image_value_norm in target_ids
                )
                if not is_target:
                    continue

            if not image_path.exists():
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': image not found.")
                skipped_missing += 1
                continue

            try:
                image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
            except Exception:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': failed to load image.")
                skipped_fail += 1
                continue

            cache_key = (image_rel.as_posix(), prompt)
            mask_array = mask_cache.get(cache_key)
            if mask_array is None:
                mask_array = _load_mask_from_dir(MASK_DIR, image_rel.stem, prompt, rec_id)
                if mask_array is not None:
                    mask_cache[cache_key] = mask_array

            gh = gw = IMAGE_SIZE // PATCH_SIZE
            try:
                mask_np = ensure_mask(mask_array, gh, gw, stem=image_rel.stem)
            except RuntimeError:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': mask validation failed.")
                skipped_fail += 1
                continue

            mask_tensor = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
            mask_small = F.interpolate(mask_tensor, size=(gh, gw), mode="bilinear", align_corners=False)
            mask_small = mask_small.squeeze(0).squeeze(0).clamp_(0.0, 1.0)
            mask_soft = soften_mask(mask_small, ksize=5, iters=2)
            if args.gaussian:
                mask_soft = (_gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)

            # Use the original (untranslated) mask for all Wasserstein comparisons; only overrides may shift.
            mask_for_distribution = mask_soft.clone()
            mask_for_override = mask_soft
            if args.tx or args.ty:
                mask_for_override = translate_mask(mask_soft, tx=args.tx, ty=args.ty, border_value=0.0)
            mask_vector = mask_for_distribution.reshape(-1)

            question_inputs = tokenizer(
                question,
                padding="longest",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            )
            question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
            question_inputs["input_ids"][:, 0] = tokenizer.enc_token_id

            tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
            focus_words = guess_focus_words(question)
            focus_indices = select_override_indices(tokens, focus_words, tokenizer)
            if not focus_indices:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': no focus indices found.")
                skipped_missing += 1
                continue

            record_id_str = str(rec_id or image_rel.stem)

            override_rows = {idx_token: mask_for_override for idx_token in focus_indices}
            new_forward = make_forward(heads_arg, override_rows)

            with torch.no_grad():
                image_embeds = model.visual_encoder(image)
                image_att_mask = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=device,
                )
                baseline_outputs = model.text_encoder(
                    input_ids=question_inputs["input_ids"],
                    attention_mask=question_inputs["attention_mask"],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_att_mask,
                    output_attentions=True,
                    return_dict=True,
                )

            baseline_cross_attentions = baseline_outputs.cross_attentions
            if not baseline_cross_attentions:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': baseline cross-attention empty.")
                skipped_fail += 1
                continue

            mask_distribution = mask_vector.clone()
            total_mask_mass = mask_distribution.sum()
            if float(total_mask_mass.item()) <= 0.0:
                fill_value = 1.0 / max(mask_distribution.numel(), 1)
                mask_distribution = torch.full_like(mask_distribution, fill_value)
            else:
                mask_distribution = mask_distribution / total_mask_mass

            if args.keep_cls:
                cls_mass = mask_distribution.mean()
                mask_distribution = torch.cat([cls_mass.unsqueeze(0), mask_distribution], dim=0)
                mask_distribution = mask_distribution / mask_distribution.sum().clamp_min(1e-12)

            baseline_wass_tensor: Optional[np.ndarray] = None
            override_wass_tensor: Optional[np.ndarray] = None
            max_wass: float = 0.0
            if not args.pairwise_wasserstein:
                try:
                    baseline_wass_tensor, max_wass = compute_wasserstein_tensor_2d(
                        baseline_cross_attentions,
                        mask_distribution,
                        grid_h=gh,
                        grid_w=gw,
                        drop_image_cls=not args.keep_cls,
                    )
                except (ValueError, RuntimeError) as exc:
                    if target_ids and is_target:
                        print(
                            f"[warn] Skipping requested record '{rec_id or image_value}': "
                            f"baseline Wasserstein computation failed ({exc})."
                        )
                    skipped_fail += 1
                    continue

            try:
                apply_override(new_forward, target_layers, originals)
                with torch.no_grad():
                    override_outputs = model.text_encoder(
                        input_ids=question_inputs["input_ids"],
                        attention_mask=question_inputs["attention_mask"],
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_att_mask,
                        output_attentions=True,
                        return_dict=True,
                    )
            except Exception:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': model forward failed.")
                skipped_fail += 1
                continue
            finally:
                revert_override(target_layers, originals)

            override_cross_attentions = override_outputs.cross_attentions
            if not override_cross_attentions:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': cross-attention empty.")
                skipped_fail += 1
                continue
            try:
                if args.pairwise_wasserstein:
                    pairwise_wass_tensor, max_wass = compute_wasserstein_between_runs_2d(
                        baseline_cross_attentions,
                        override_cross_attentions,
                        grid_h=gh,
                        grid_w=gw,
                        drop_image_cls=not args.keep_cls,
                    )
                    baseline_wass_tensor = pairwise_wass_tensor
                    override_wass_tensor = pairwise_wass_tensor
                else:
                    override_wass_tensor, _ = compute_wasserstein_tensor_2d(
                        override_cross_attentions,
                        mask_distribution,
                        grid_h=gh,
                        grid_w=gw,
                        drop_image_cls=not args.keep_cls,
                    )
                if baseline_wass_tensor is None:
                    raise ValueError("Baseline Wasserstein tensor missing.")
            except (ValueError, RuntimeError) as exc:
                if target_ids and is_target:
                    print(
                        f"[warn] Skipping requested record '{rec_id or image_value}': "
                        f"override Wasserstein computation failed ({exc})."
                    )
                skipped_fail += 1
                continue

            record_tokens: List[Dict[str, object]] = []
            assert baseline_wass_tensor is not None
            num_tokens = min(len(tokens), baseline_wass_tensor.shape[2], override_wass_tensor.shape[2])
            norm_denom = max(max_wass, 1e-12)
            focus_set = set(focus_indices)
            attention_flags = question_inputs["attention_mask"][0].detach().cpu().tolist()

            valid_token_indices = [
                idx_token
                for idx_token in range(num_tokens)
                if idx_token < len(attention_flags) and attention_flags[idx_token] != 0
            ]
            if valid_token_indices:
                mask_vs_baseline_wasserstein = float(
                    np.mean(baseline_wass_tensor[:, :, valid_token_indices])
                )
                mask_vs_override_wasserstein = float(
                    np.mean(override_wass_tensor[:, :, valid_token_indices])
                )
            else:
                mask_vs_baseline_wasserstein = 0.0
                mask_vs_override_wasserstein = 0.0
            mask_vs_delta_wasserstein = mask_vs_override_wasserstein - mask_vs_baseline_wasserstein

            for token_idx in range(num_tokens):
                if token_idx >= len(attention_flags) or attention_flags[token_idx] == 0:
                    continue
                baseline_matrix = baseline_wass_tensor[:, :, token_idx]
                override_matrix = override_wass_tensor[:, :, token_idx]
                delta_matrix = override_matrix - baseline_matrix

                baseline_mean = float(baseline_matrix.mean())
                override_mean = float(override_matrix.mean())
                delta_mean = override_mean - baseline_mean

                baseline_norm = baseline_matrix / norm_denom
                override_norm = override_matrix / norm_denom
                delta_norm = delta_matrix / norm_denom

                token_entry: Dict[str, object] = {
                    "token": tokens[token_idx],
                    "index": int(token_idx),
                    "overridden": token_idx in focus_set,
                    "baseline_wasserstein_matrix": _format_matrix(baseline_matrix),
                    "baseline_wasserstein_norm_matrix": _format_matrix(baseline_norm),
                    "override_wasserstein_matrix": _format_matrix(override_matrix),
                    "override_wasserstein_norm_matrix": _format_matrix(override_norm),
                    "delta_wasserstein_matrix": _format_matrix(delta_matrix),
                    "delta_wasserstein_norm_matrix": _format_matrix(delta_norm),
                    "baseline_wasserstein_mean": baseline_mean,
                    "override_wasserstein_mean": override_mean,
                    "delta_wasserstein_mean": delta_mean,
                    "baseline_wasserstein_mean_norm": float(baseline_mean / norm_denom),
                    "override_wasserstein_mean_norm": float(override_mean / norm_denom),
                    "delta_wasserstein_mean_norm": float(delta_mean / norm_denom),
                }
                record_tokens.append(token_entry)
                running_baseline_w.append(baseline_mean)
                running_override_w.append(override_mean)

                if plots_dir is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, baseline_matrix.shape[0] * 0.6)))
                    matrices = [
                        (baseline_matrix, "Baseline", "viridis"),
                        (override_matrix, "Override", "viridis"),
                        (delta_matrix, "Delta", "RdBu_r"),
                    ]

                    vmin = min(baseline_matrix.min(), override_matrix.min())
                    vmax = max(baseline_matrix.max(), override_matrix.max())

                    for axis, (matrix, title, cmap) in zip(axes, matrices):
                        if title == "Delta":
                            abs_max = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
                            im = axis.imshow(
                                matrix,
                                vmin=-abs_max,
                                vmax=abs_max,
                                aspect="auto",
                                origin="upper",
                                cmap=cmap,
                            )
                        else:
                            im = axis.imshow(
                                matrix,
                                vmin=vmin,
                                vmax=vmax,
                                aspect="auto",
                                origin="upper",
                                cmap=cmap,
                            )

                        axis.set_title(f"{title}\nWasserstein")
                        axis.set_xlabel("Head")
                        axis.set_ylabel("Layer")
                        axis.set_xticks(range(matrix.shape[1]))
                        axis.set_yticks(range(matrix.shape[0]))

                        x_labels = []
                        y_labels = []
                        for i in range(matrix.shape[0]):
                            if i in target_layers:
                                y_labels.append(f"$\\mathbf{{{i}}}$")
                            else:
                                y_labels.append(str(i))
                        for i in range(matrix.shape[1]):
                            if heads_spec == (-1,) or i in heads_spec:
                                x_labels.append(f"$\\mathbf{{{i}}}$")
                            else:
                                x_labels.append(str(i))

                        axis.set_xticklabels(x_labels)
                        axis.set_yticklabels(y_labels)
                        plt.colorbar(im, ax=axis, shrink=0.8)

                    token_text = tokens[token_idx]
                    override_mark = "*" if token_idx in focus_set else "-"
                    fig.suptitle(
                        f"{record_id_str} - token {token_idx}: {token_text} (override: {override_mark})",
                        fontsize=12,
                        y=1.05,
                    )

                    plt.tight_layout()

                    plot_name = f"{_sanitize_name(record_id_str)}_tok{token_idx:02d}_{_sanitize_name(tokens[token_idx])}.pdf"
                    plot_path = plots_dir / plot_name
                    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
                    plt.close(fig)
                    token_entry["plot_path"] = str(plot_path)

            if not record_tokens:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': no valid tokens collected.")
                skipped_missing += 1
                continue

            layer_list = list(range(total_layers)) if layers_spec == (-1,) else list(target_layers)
            head_repr: object
            if heads_spec == (-1,):
                head_repr = "all"
            else:
                head_repr = list(int(h) for h in heads_spec)

            wass_mode_str = "pairwise" if args.pairwise_wasserstein else "mask"
            output_record = {
                "id": record_id_str,
                "image": image_value,
                "question": question,
                "wasserstein_mode": wass_mode_str,
                "focus_words": focus_words,
                "override_indices": focus_indices,
                "target_layers": layer_list,
                "heads": head_repr,
                "max_wasserstein": max_wass,
                "mask_vs_baseline_wasserstein_mean": mask_vs_baseline_wasserstein,
                "mask_vs_override_wasserstein_mean": mask_vs_override_wasserstein,
                "mask_vs_delta_wasserstein_mean": mask_vs_delta_wasserstein,
                "tokens": record_tokens,
            }

            writer.write(json.dumps(output_record, indent=2) + "\n")
            written += 1
            if target_ids and is_target:
                matched_keys = {key for key in (rec_id_norm, image_stem_norm, image_value_norm) if key}
                found_targets.update(matched_keys)
                _print_record_summary(
                    record_id_str,
                    image_value,
                    question,
                    record_tokens,
                    wass_mode=wass_mode_str,
                )
                if args.pairwise_wasserstein:
                    print(f"pairwise W={mask_vs_baseline_wasserstein:.4f}")
                else:
                    print(
                        f"mask-vs-baseline W={mask_vs_baseline_wasserstein:.4f} "
                        f"mask-vs-override W={mask_vs_override_wasserstein:.4f} "
                        f"delta={mask_vs_delta_wasserstein:+.4f}"
                    )

            if idx == 1 or idx % 10 == 0:
                print(
                    f"[progress] {idx}/{len(records)} processed | written={written} "
                    f"skipped_missing={skipped_missing} skipped_fail={skipped_fail}",
                    flush=True,
                )

    if running_baseline_w:
        mean_baseline_w = float(np.mean(running_baseline_w))
        std_baseline_w = float(np.std(running_baseline_w))
    else:
        mean_baseline_w = 0.0
        std_baseline_w = 0.0

    if running_override_w:
        mean_override_w = float(np.mean(running_override_w))
        std_override_w = float(np.std(running_override_w))
    else:
        mean_override_w = 0.0
        std_override_w = 0.0

    print(
        "[done] "
        f"records={processed} written={written} skipped_missing={skipped_missing} skipped_fail={skipped_fail} "
        f"baseline_wasserstein={mean_baseline_w:.4f} +/- {std_baseline_w:.4f} "
        f"override_wasserstein={mean_override_w:.4f} +/- {std_override_w:.4f}"
    )

    if target_ids:
        missing_targets = target_ids - found_targets
        if missing_targets:
            readable = ", ".join(sorted(missing_targets))
            print(f"[warn] No Wasserstein results produced for requested id(s): {readable}")


if __name__ == "__main__":
    main()
