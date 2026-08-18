import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, NamedTuple

import cv2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
output = Path("att_magnitude/focused_magnitude.jsonl")


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
    x = mask_hw.detach().float().clamp(0,1).cpu().numpy()
    return (x * 255.0).astype(np.uint8)

def _to_torch_float(mask_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device)

def translate_mask(mask_hw: torch.Tensor, tx: int = 0, ty: int = 0,
                   border_value: float = 0.0) -> torch.Tensor:
    if tx == 0 and ty == 0:
        return mask_hw
    h, w = mask_hw.shape[-2], mask_hw.shape[-1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])   # translation matrix
    src = _to_numpy_uint8(mask_hw)
    dst = cv2.warpAffine(
        src, M, (w, h),
        flags=cv2.INTER_NEAREST,              # preserve hard mask edges
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.clip(border_value, 0, 1) * 255)
    )
    return _to_torch_float(dst, mask_hw.device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply attention overrides and compute per-layer/head attention magnitude statistics for focus tokens."
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
        help="Keep the image CLS token when computing magnitude metrics (default drops it).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate side-by-side baseline/override heatmaps for each token.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("att_magnitude/plots"),
        help="Directory where token heatmap images will be stored when --save-plots is set.",
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


class MagnitudeStats(NamedTuple):
    magnitude: np.ndarray
    magnitude_norm: np.ndarray
    inside_rms: np.ndarray
    outside_rms: np.ndarray
    mer: np.ndarray
    uniform_magnitude: float


def compute_attention_magnitudes(
    cross_attentions: Sequence[torch.Tensor],
    *,
    drop_image_cls: bool = True,
    mask_weights: Optional[torch.Tensor] = None,
) -> MagnitudeStats:

    if not cross_attentions:
        raise ValueError("No cross-attention tensors were provided.")

    reference_tensor = cross_attentions[0]
    layers = len(cross_attentions)
    _, _, _, num_keys = reference_tensor.shape
    key_start = 1 if drop_image_cls else 0
    num_effective_keys = num_keys - key_start
    if num_effective_keys <= 0:
        raise ValueError("Cannot compute magnitudes with zero effective key tokens.")

    uniform_magnitude = 1.0 / math.sqrt(num_effective_keys)

    magnitude_layers: List[np.ndarray] = []
    magnitude_norm_layers: List[np.ndarray] = []
    inside_rms_layers: List[np.ndarray] = []
    outside_rms_layers: List[np.ndarray] = []
    mer_layers: List[np.ndarray] = []

    mask_tensor = None
    outside_tensor = None
    inside_den = None
    outside_den = None
    if mask_weights is not None:
        if mask_weights.numel() != num_effective_keys:
            raise ValueError(
                f"Mask has {mask_weights.numel()} entries, expected {num_effective_keys}."
            )
        mask_tensor = (
            mask_weights.reshape(1, 1, -1)
            .to(reference_tensor.device)
            .to(dtype=reference_tensor.dtype)
            .clamp_(0.0, 1.0)
        )
        outside_tensor = (1.0 - mask_tensor).clamp_(0.0, 1.0)
        inside_den = mask_tensor.sum().clamp_min(1e-12)
        outside_den = outside_tensor.sum().clamp_min(1e-12)

    for layer_tensor in cross_attentions:
        attn = layer_tensor[0][:, :, key_start:]  # (heads, tokens, keys_without_cls)
        probs = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs_sq = probs * probs

        l2 = torch.linalg.vector_norm(probs, dim=-1)
        if uniform_magnitude >= 1.0 - 1e-12:
            magnitude_norm = torch.zeros_like(l2)
        else:
            magnitude_norm = (l2 - uniform_magnitude) / (1.0 - uniform_magnitude)
            magnitude_norm = magnitude_norm.clamp(0.0, 1.0)

        if mask_tensor is not None:
            inside_rms = torch.sqrt(
                (probs_sq * mask_tensor).sum(dim=-1) / inside_den
            )
            outside_rms = torch.sqrt(
                (probs_sq * outside_tensor).sum(dim=-1) / outside_den
            )
            mer = inside_rms / (inside_rms + outside_rms).clamp_min(1e-12)
        else:
            inside_rms = torch.zeros_like(l2)
            outside_rms = torch.zeros_like(l2)
            mer = torch.full_like(l2, 0.5)

        magnitude_layers.append(l2.detach().cpu().numpy())
        magnitude_norm_layers.append(magnitude_norm.detach().cpu().numpy())
        inside_rms_layers.append(inside_rms.detach().cpu().numpy())
        outside_rms_layers.append(outside_rms.detach().cpu().numpy())
        mer_layers.append(mer.detach().cpu().numpy())

    return MagnitudeStats(
        magnitude=np.stack(magnitude_layers, axis=0),
        magnitude_norm=np.stack(magnitude_norm_layers, axis=0),
        inside_rms=np.stack(inside_rms_layers, axis=0),
        outside_rms=np.stack(outside_rms_layers, axis=0),
        mer=np.stack(mer_layers, axis=0),
        uniform_magnitude=uniform_magnitude,
    )


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
    uniform_magnitude: float,
) -> None:
    print(f"\n[record] id={record_id}  image={image_value}")
    print(f"question: {question}")
    print(f"uniform magnitude (L2): {uniform_magnitude:.4f}")
    print(
        " idx | override | token                 | base_L2 | new_L2 | delta | base_norm | new_norm | delta_norm | base_MER | new_MER | delta_MER"
    )
    print(
        "-----+----------+-----------------------+--------+--------+-------+-----------+----------+------------+----------+---------+-----------"
    )
    for tok in record_tokens:
        token_text = str(tok["token"])
        mark = "*" if tok.get("overridden") else " "
        print(
            f"{tok['index']:>4} |    {mark}     | {token_text:<21} | "
            f"{tok['baseline_mean_l2']:.4f} | {tok['override_mean_l2']:.4f} | {tok['delta_mean_l2']:+.4f} | "
            f"{tok['baseline_mean_norm']:.4f} | {tok['override_mean_norm']:.4f} | {tok['delta_mean_norm']:+.4f} | "
            f"{tok['baseline_mean_mer']:.4f} | {tok['override_mean_mer']:.4f} | {tok['delta_mean_mer']:+.4f}"
        )


def _format_matrix(values: np.ndarray, precision: int = 4) -> List[str]:
    rows: List[str] = []
    for row in values:
        rows.append(" ".join(f"{float(cell):.{precision}f}" for cell in row))
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    running_magnitude: List[float] = []
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}

    found_targets: Set[str] = set()

    with output.open("w", encoding="utf-8") as writer:
        for idx, entry in enumerate(records, 1):
            processed += 1

            image_value = entry.get("image")
            question = entry.get("question")
            if not image_value or not question:
                # If this record was requested explicitly, report the skip.
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

            mask_soft = translate_mask(mask_soft, tx=args.tx, ty=args.ty, border_value=0.0)
            mask_vector = mask_soft.reshape(-1)

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

            override_rows = {idx_token: mask_soft for idx_token in focus_indices}
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

            try:
                baseline_stats = compute_attention_magnitudes(
                    baseline_cross_attentions,
                    drop_image_cls=not args.keep_cls,
                    mask_weights=mask_vector,
                )
            except ValueError:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': baseline magnitude failed.")
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
                override_stats = compute_attention_magnitudes(
                    override_cross_attentions,
                    drop_image_cls=not args.keep_cls,
                    mask_weights=mask_vector,
                )
            except ValueError:
                if target_ids and is_target:
                    print(f"[warn] Skipping requested record '{rec_id or image_value}': magnitude computation failed.")
                skipped_fail += 1
                continue

            record_tokens: List[Dict[str, object]] = []
            num_tokens = min(
                len(tokens),
                baseline_stats.magnitude.shape[2],
                override_stats.magnitude.shape[2],
            )
            focus_set = set(focus_indices)
            attention_flags = question_inputs["attention_mask"][0].detach().cpu().tolist()
            uniform_l2 = baseline_stats.uniform_magnitude
            for token_idx in range(num_tokens):
                if token_idx >= len(attention_flags) or attention_flags[token_idx] == 0:
                    continue
                baseline_matrix = baseline_stats.magnitude[:, :, token_idx]
                override_matrix = override_stats.magnitude[:, :, token_idx]
                delta_matrix = override_matrix - baseline_matrix

                baseline_norm = baseline_stats.magnitude_norm[:, :, token_idx]
                override_norm = override_stats.magnitude_norm[:, :, token_idx]
                delta_norm = override_norm - baseline_norm

                baseline_inside = baseline_stats.inside_rms[:, :, token_idx]
                override_inside = override_stats.inside_rms[:, :, token_idx]
                delta_inside = override_inside - baseline_inside

                baseline_outside = baseline_stats.outside_rms[:, :, token_idx]
                override_outside = override_stats.outside_rms[:, :, token_idx]
                delta_outside = override_outside - baseline_outside

                baseline_mer = baseline_stats.mer[:, :, token_idx]
                override_mer = override_stats.mer[:, :, token_idx]
                delta_mer = override_mer - baseline_mer

                baseline_mean_l2 = float(baseline_matrix.mean())
                override_mean_l2 = float(override_matrix.mean())
                delta_mean_l2 = override_mean_l2 - baseline_mean_l2

                baseline_mean_norm = float(baseline_norm.mean())
                override_mean_norm = float(override_norm.mean())
                delta_mean_norm = override_mean_norm - baseline_mean_norm

                baseline_mean_mer = float(baseline_mer.mean())
                override_mean_mer = float(override_mer.mean())
                delta_mean_mer = override_mean_mer - baseline_mean_mer

                baseline_mean_inside = float(baseline_inside.mean())
                override_mean_inside = float(override_inside.mean())
                delta_mean_inside = override_mean_inside - baseline_mean_inside

                baseline_mean_outside = float(baseline_outside.mean())
                override_mean_outside = float(override_outside.mean())
                delta_mean_outside = override_mean_outside - baseline_mean_outside

                token_entry: Dict[str, object] = {
                    "token": tokens[token_idx],
                    "index": int(token_idx),
                    "overridden": token_idx in focus_set,
                    "baseline_l2_matrix": _format_matrix(baseline_matrix),
                    "override_l2_matrix": _format_matrix(override_matrix),
                    "delta_l2_matrix": _format_matrix(delta_matrix),
                    "baseline_l2_norm_matrix": _format_matrix(baseline_norm),
                    "override_l2_norm_matrix": _format_matrix(override_norm),
                    "delta_l2_norm_matrix": _format_matrix(delta_norm),
                    "baseline_inside_rms_matrix": _format_matrix(baseline_inside),
                    "override_inside_rms_matrix": _format_matrix(override_inside),
                    "delta_inside_rms_matrix": _format_matrix(delta_inside),
                    "baseline_outside_rms_matrix": _format_matrix(baseline_outside),
                    "override_outside_rms_matrix": _format_matrix(override_outside),
                    "delta_outside_rms_matrix": _format_matrix(delta_outside),
                    "baseline_mer_matrix": _format_matrix(baseline_mer),
                    "override_mer_matrix": _format_matrix(override_mer),
                    "delta_mer_matrix": _format_matrix(delta_mer),
                    "baseline_mean_l2": baseline_mean_l2,
                    "override_mean_l2": override_mean_l2,
                    "delta_mean_l2": delta_mean_l2,
                    "baseline_mean_norm": baseline_mean_norm,
                    "override_mean_norm": override_mean_norm,
                    "delta_mean_norm": delta_mean_norm,
                    "baseline_mean_mer": baseline_mean_mer,
                    "override_mean_mer": override_mean_mer,
                    "delta_mean_mer": delta_mean_mer,
                    "baseline_mean_inside_rms": baseline_mean_inside,
                    "override_mean_inside_rms": override_mean_inside,
                    "delta_mean_inside_rms": delta_mean_inside,
                    "baseline_mean_outside_rms": baseline_mean_outside,
                    "override_mean_outside_rms": override_mean_outside,
                    "delta_mean_outside_rms": delta_mean_outside,
                }
                record_tokens.append(token_entry)
                running_magnitude.append(override_mean_l2)

                if plots_dir is not None:
                    # Create figure with 3 subplots side by side
                    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, baseline_matrix.shape[0] * 0.6)))
                    
                    # Common min/max for consistent color scaling
                    vmin = min(baseline_matrix.min(), override_matrix.min())
                    vmax = max(baseline_matrix.max(), override_matrix.max())
                    
                    matrices = [
                        (baseline_matrix, "Baseline", "viridis"),
                        (override_matrix, "Override", "viridis"),
                        (delta_matrix, "Delta", "RdBu_r"),  # Red-Blue diverging colormap for delta
                    ]
                    
                    for axis, (matrix, title, cmap) in zip(axes, matrices):
                        if title == "Delta":
                            # For delta, use symmetric limits centered at 0
                            abs_max = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
                            im = axis.imshow(
                                matrix,
                                vmin=-abs_max,
                                vmax=abs_max,
                                aspect="auto",
                                origin="upper",
                                cmap=cmap
                            )
                        else:
                            im = axis.imshow(
                                matrix,
                                vmin=vmin,
                                vmax=vmax,
                                aspect="auto",
                                origin="upper",
                                cmap=cmap
                            )

                        axis.set_title(f"{title}\nL2 Magnitude")
                        axis.set_xlabel("Head")
                        axis.set_ylabel("Layer")
                        axis.set_xticks(range(matrix.shape[1]))
                        axis.set_yticks(range(matrix.shape[0]))
                        
                        # Create tick labels with bold for target layers
                        x_labels = []
                        y_labels = []
                        for i in range(matrix.shape[0]):
                            if i in target_layers:
                                y_labels.append(f"$\\mathbf{{{i}}}$")  # Bold using LaTeX
                            else:
                                y_labels.append(str(i))
                        # Bold target heads
                        for i in range(matrix.shape[1]):
                            if heads_spec == (-1,) or i in heads_spec:
                                x_labels.append(f"$\\mathbf{{{i}}}$")  # Bold using LaTeX
                            else:
                                x_labels.append(str(i))

                        axis.set_xticklabels(x_labels)                                
                        axis.set_yticklabels(y_labels)

                        # Add colorbar for each plot
                        plt.colorbar(im, ax=axis, shrink=0.8)

                    token_text = tokens[token_idx]
                    is_override = "✓" if token_idx in focus_set else "✗"
                    fig.suptitle(f"{record_id_str} – token {token_idx}: {token_text} (override: {is_override})", 
                                fontsize=12, y=1.05)
                    
                    # Adjust layout to prevent overlap
                    plt.tight_layout()
                    
                    plot_name = f"{_sanitize_name(record_id_str)}_tok{token_idx:02d}_{_sanitize_name(tokens[token_idx])}.pdf"
                    plot_path = plots_dir / plot_name
                    fig.savefig(plot_path, dpi=220, bbox_inches='tight')
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

            output_record = {
                "id": record_id_str,
                "image": image_value,
                "question": question,
                "focus_words": focus_words,
                "override_indices": focus_indices,
                "target_layers": layer_list,
                "heads": head_repr,
                "uniform_magnitude_l2": uniform_l2,
                "max_magnitude_l2": 1.0,
                "tokens": record_tokens,
            }

            writer.write(json.dumps(output_record, indent=2) + "\n")
            written += 1
            if target_ids and is_target:
                matched_keys = {key for key in (rec_id_norm, image_stem_norm, image_value_norm) if key}
                found_targets.update(matched_keys)
                _print_record_summary(record_id_str, image_value, question, record_tokens, uniform_l2)

            if idx == 1 or idx % 10 == 0:
                print(
                    f"[progress] {idx}/{len(records)} processed | written={written} "
                    f"skipped_missing={skipped_missing} skipped_fail={skipped_fail}",
                    flush=True,
                )

    if running_magnitude:
        mean_magnitude = float(np.mean(running_magnitude))
        std_magnitude = float(np.std(running_magnitude))
    else:
        mean_magnitude = 0.0
        std_magnitude = 0.0

    print(
        "[done] "
        f"records={processed} written={written} skipped_missing={skipped_missing} skipped_fail={skipped_fail} "
        f"mean_magnitude={mean_magnitude:.4f} +/- {std_magnitude:.4f}"
    )

    if target_ids:
        missing_targets = target_ids - found_targets
        if missing_targets:
            readable = ", ".join(sorted(missing_targets))
            print(f"[warn] No magnitude results produced for requested id(s): {readable}")


if __name__ == "__main__":
    main()
