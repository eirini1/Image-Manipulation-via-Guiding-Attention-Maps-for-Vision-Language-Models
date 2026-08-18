'''Grid search over transformation of attention mask.'''

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List
from itertools import product

import torch
import numpy as np

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root
DEBUG_MASK_DIR = Path("mask_debug")


def _sanitize_name(s: str) -> str:
    s = "" if s is None else str(s)
    cleaned = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in s)
    return cleaned[:80]


def _load_mask_from_dir(mask_dir: Path, image_stem: str, prompt: str, rec_id: Optional[str]) -> Optional[np.ndarray]:
    sanitized = _sanitize_name(prompt)
    patterns = []
    if rec_id:
        patterns.append(f"{rec_id}_{image_stem}_{sanitized}_*.npy")
    patterns.append(f"*_{image_stem}_{sanitized}_*.npy")

    for pat in patterns:
        candidates = sorted(mask_dir.glob(pat))
        if candidates:
            path = candidates[-1]
            try:
                arr = np.load(path)
                if arr.ndim > 2:
                    arr = arr.squeeze()
                return arr.astype(np.float32)
            except Exception:
                continue
    return None


def _save_mask_image(image_path: Path, mask, path: Path, alpha: float = 0.45) -> None:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().float().clamp(0, 1).cpu().numpy()
    else:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.size and (arr.min() < 0 or arr.max() > 1):
            arr = np.clip(arr, 0.0, 1.0)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    h, w = image.shape[:2]
    mask_resized = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_resized = np.clip(mask_resized, 0.0, 1.0)
    overlay = image.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255.0
    alpha_map = (alpha * mask_resized)[..., None]
    overlay = overlay * (1.0 - alpha_map) + red * alpha_map
    overlay = overlay.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def _gaussian_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Create a 2D Gaussian weighting based on the mask's spatial statistics.
    Expects `mask` as float tensor of shape (H, W) on the target device, in [0,1].
    Returns a tensor of shape (H, W) with peak 1.0.
    """
    h, w = mask.shape[-2], mask.shape[-1]
    if float(mask.max().item()) <= 0.0:
        return torch.zeros_like(mask)

    ys = torch.arange(h, device=mask.device, dtype=mask.dtype)
    xs = torch.arange(w, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij') if hasattr(torch.meshgrid, "__call__") else torch.meshgrid(ys, xs)

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
    maxv = gauss.max().clamp(min=1e-6)
    gauss = gauss / maxv
    return gauss

#-----------------------------------------------------------------------------
#-----------------------Transormation of mask---------------------------------
import cv2

# ---- helpers ----
_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}

def _to_numpy_uint8(mask_hw: torch.Tensor) -> np.ndarray:
    x = mask_hw.detach().float().clamp(0,1).cpu().numpy()
    return (x * 255.0).astype(np.uint8)

def _to_torch_float(mask_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device)

def opencv_morph(mask_hw: torch.Tensor,
                 op: str = "none",       # "erode"|"dilate"|"open"|"close"|"grad"
                 ksize: int = 3,
                 shape: str = "ellipse", # "rect"|"ellipse"|"cross"
                 iters: int = 1,
                 device=None) -> torch.Tensor:
    if op == "none" or ksize <= 1 or iters <= 0:
        return mask_hw
    kernel = cv2.getStructuringElement(_SHAPES.get(shape, cv2.MORPH_ELLIPSE), (ksize, ksize))
    src = _to_numpy_uint8(mask_hw)
    if op == "erode":
        dst = cv2.erode(src, kernel, iterations=iters)
    elif op == "dilate":
        dst = cv2.dilate(src, kernel, iterations=iters)
    else:
        ops = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "grad": cv2.MORPH_GRADIENT}
        dst = cv2.morphologyEx(src, ops[op], kernel, iterations=iters)
    return _to_torch_float(dst, mask_hw.device if device is None else device)

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
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Override attention on selected layers/heads and evaluate")
    parser.add_argument("--layers", default="all", help="Layer spec: 'all' or comma-separated indices")
    parser.add_argument("--heads", default="all", help="Head spec: 'all' or comma-separated indices")
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="Apply a Gaussian weighting constrained to the mask region",
    )

    parser.add_argument("--morph", default="none",
                        choices=["none","erode","dilate","open","close","grad"],
                        help="Morphological op for single run")
    parser.add_argument("--ksize", type=int, default=3, help="Kernel size for morph op")
    parser.add_argument("--shape", default="ellipse",
                        choices=["rect","ellipse","cross"],
                        help="Kernel shape for morph op")
    parser.add_argument("--iters", type=int, default=1, help="Iterations for morph op")
    parser.add_argument("--tx", type=int, default=0, help="Mask translation in x (pixels)")
    parser.add_argument("--ty", type=int, default=0, help="Mask translation in y (pixels)")

    # Grid search options (optional)
    parser.add_argument("--grid-search", action="store_true",
                        help="Run a grid search over transform parameters")
    parser.add_argument("--morphs", type=str, default=None,
                        help="Comma-separated morph ops for grid: none,erode,dilate,open,close,grad")
    parser.add_argument("--ksizes", type=str, default=None,
                        help="Comma-separated kernel sizes for grid, e.g. 3,5,7")
    parser.add_argument("--iters-list", type=str, default=None,
                        help="Comma-separated iteration counts for grid, e.g. 1,2,3")
    parser.add_argument("--txs", type=str, default=None,
                        help="Comma-separated x translations for grid, e.g. -2,0,2")
    parser.add_argument("--tys", type=str, default=None,
                        help="Comma-separated y translations for grid, e.g. -2,0,2")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Comma-separated kernel shapes for grid: rect,ellipse,cross")
    parser.add_argument("--grid-out", type=str, default=None,
                        help="Optional path to write CSV results of grid search")

    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = parse_heads(args.heads)
    layers_spec = parse_heads(args.layers)

    records = list(iter_jsonl(data_path))
    if not records:
        print("No records to evaluate.")
        return 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base")
    model.eval()
    model = model.to(device)

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    # Resolve which layers to override
    if layers_spec == (-1,):
        target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        target_layers = tuple(int(x) for x in layers_spec)

    tokenizer = model.tokenizer

    total_trials = 0
    total_hits = 0
    retain_trials = 0  # items with correct == yes
    retain_hits = 0    # of those, stayed correct after change

    start_time = time.time()
    PROG_STEP = 10

    def evaluate_once(morph: str, ksize: int, shape: str, iters: int, tx: int, ty: int) -> Tuple[int,int,int,int]:
        total_trials = 0
        total_hits = 0
        retain_trials = 0  # items with correct == yes
        retain_hits = 0    # of those, stayed correct after change

        start_time = time.time()
        PROG_STEP = 10

        for idx, entry in enumerate(records, 1):
            image_value = entry.get("image")
            prompt = entry.get("prompt")
            if image_value is None or entry.get("question") is None or entry.get("answer") is None:
                print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
                continue

            image_rel = Path(image_value)
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
                continue

            stem = image_rel.stem
            question = entry["question"]
            gold = entry["answer"]
            gold_norm = normalize_answer(gold)
            is_yes_labeled = str(entry.get("correct", "")).strip().lower() == "yes"

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)

            # Load or fallback mask
            mask_array: Optional[np.ndarray] = None
            cache_key = (image_rel.as_posix(), prompt)
            if cache_key in mask_cache:
                mask_array = mask_cache[cache_key]
            else:
                rec_id = entry.get("id")
                mask_array = _load_mask_from_dir(MASK_DIR, stem, prompt or "", rec_id)
                if mask_array is not None:
                    mask_cache[cache_key] = mask_array

            gh = gw = IMAGE_SIZE // PATCH_SIZE
            try:
                mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
            except RuntimeError:
                continue

            debug_base = f"{entry.get('id', idx)}_{stem}_{_sanitize_name(prompt or '')}"
            _save_mask_image(image_path, mask_array, DEBUG_MASK_DIR / f"{debug_base}_before.png")
            mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
            mask_small = torch.nn.functional.interpolate(
                mask_tensor,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
            _save_mask_image(image_path, mask_small, DEBUG_MASK_DIR / f"{debug_base}_after.png")
            mask_soft = soften_mask(mask_small, ksize=5, iters=2)

            # Optionally convert region into a Gaussian weighting and constrain by the softened mask
            if args.gaussian:
                gauss = _gaussian_from_mask(mask_soft)
                mask_soft = (gauss * mask_soft).clamp_(0.0, 1.0)
                _save_mask_image(image_path, mask_soft, DEBUG_MASK_DIR / f"{debug_base}_gaussian.png")

            # Apply transforms
            mask_soft_local = opencv_morph(mask_soft, op=morph, ksize=ksize, shape=shape, iters=iters)
            mask_soft_local = translate_mask(mask_soft_local, tx=tx, ty=ty, border_value=0.0)

            # Build token-level override rows
            question_inputs = tokenizer(
                question,
                padding="longest",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            )
            question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
            question_inputs["input_ids"][0, 0] = tokenizer.enc_token_id
            tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
            focus_words = guess_focus_words(question)
            override_indices = select_override_indices(tokens, focus_words, tokenizer)
            if not override_indices:
                override_indices = [0]
            override_rows = {i: mask_soft_local for i in override_indices}

            # Prepare attention override for requested heads
            heads_arg = heads[0] if len(heads) == 1 else heads
            new_forward = make_forward(heads_arg, override_rows)

            # Apply override, run inference, revert
            apply_override(new_forward, target_layers, originals)
            with torch.no_grad():
                pred = model(image, question, train=False, inference="generate")[0]
            revert_override(target_layers, originals)

            pred_norm = normalize_answer(pred)
            hit = int(pred_norm == gold_norm)
            total_trials += 1
            if hit:
                total_hits += 1
            if is_yes_labeled:
                retain_trials += 1
                if hit:
                    retain_hits += 1

            if (idx == 1) or (idx % PROG_STEP == 0):
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                acc = (total_hits / total_trials * 100.0) if total_trials else 0.0
                ret = (retain_hits / retain_trials * 100.0) if retain_trials else 0.0
                print(
                    f"[progress] {idx}/{len(records)} | acc: {total_hits}/{total_trials} ({acc:.2f}%)"
                    + (f" | retain: {retain_hits}/{retain_trials} ({ret:.2f}%)" if retain_trials else "")
                    + f" | {rate:.2f} rec/s",
                    flush=True,
                )

        return total_trials, total_hits, retain_trials, retain_hits

    # Run single evaluation or grid search
    if not args.grid_search:
        total_trials, total_hits, retain_trials, retain_hits = evaluate_once(
            args.morph, args.ksize, args.shape, args.iters, args.tx, args.ty
        )
        print("========================================")
        print(f"Layers: {list(target_layers)} | Heads: {list(heads) if len(heads)>1 else heads[0]}")
        final_acc = (total_hits / total_trials * 100.0) if total_trials else 0.0
        print(f"Post-change accuracy: {total_hits}/{total_trials} ({final_acc:.2f}%)")
        if retain_trials:
            ret_acc = (retain_hits / retain_trials * 100.0)
            print(f"Retention (originally-correct staying correct): {retain_hits}/{retain_trials} ({ret_acc:.2f}%)")
        else:
            print("No items labeled as originally correct in dataset.")
        return 0

    # Grid search branch
    def parse_list(val: Optional[str], cast) -> Optional[List]:
        if val is None:
            return None
        items = [x.strip() for x in val.split(',') if x.strip()]
        return [cast(x) for x in items]

    morph_choices = {"none","erode","dilate","open","close","grad"}
    shape_choices = {"rect","ellipse","cross"}

    morphs = parse_list(args.morphs, str) if args.morphs else [args.morph]
    ksizes = parse_list(args.ksizes, int) if args.ksizes else [args.ksize]
    iters_list = parse_list(args.iters_list, int) if args.iters_list else [args.iters]
    txs = parse_list(args.txs, int) if args.txs else [args.tx]
    tys = parse_list(args.tys, int) if args.tys else [args.ty]
    shapes = parse_list(args.shapes, str) if args.shapes else [args.shape]

    # Validate
    for m in morphs:
        if m not in morph_choices:
            raise ValueError(f"Invalid morph in grid: {m}")
    for s in shapes:
        if s not in shape_choices:
            raise ValueError(f"Invalid shape in grid: {s}")
    for k in ksizes:
        if k < 1:
            raise ValueError(f"Invalid ksize in grid: {k}")
    for it in iters_list:
        if it < 0:
            raise ValueError(f"Invalid iters in grid: {it}")

    combos = list(product(morphs, ksizes, shapes, iters_list, txs, tys))
    print(f"[grid] Running {len(combos)} configurations...")

    results: List[Tuple[float, float, str, int, str, int, int, int, int, int, int, int]] = []
    # (acc, ret_acc, morph, ksize, shape, iters, tx, ty, total, hits, ret_total, ret_hits)

    for (morph, ksize, shape, iters, tx, ty) in combos:
        print(f"\n[grid] morph={morph} ksize={ksize} shape={shape} iters={iters} tx={tx} ty={ty}")
        total_trials, total_hits, retain_trials, retain_hits = evaluate_once(morph, ksize, shape, iters, tx, ty)
        acc = (total_hits / total_trials * 100.0) if total_trials else 0.0
        ret_acc = (retain_hits / retain_trials * 100.0) if retain_trials else 0.0
        results.append((acc, ret_acc, morph, ksize, shape, iters, tx, ty,
                        total_trials, total_hits, retain_trials, retain_hits))

    # Sort by accuracy desc, then retention desc
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print("\n========================================")
    print(f"Layers: {list(target_layers)} | Heads: {list(heads) if len(heads)>1 else heads[0]}")
    if results:
        best_acc, best_ret, bm, bk, bs, bi, btx, bty, bt, bh, brt, brh = results[0]
        print(f"Best acc: {best_acc:.2f}% with morph={bm}, ksize={bk}, shape={bs}, iters={bi}, tx={btx}, ty={bty} | ret={best_ret:.2f}%")
    else:
        print("No results computed.")

    # Print summary table
    if results:
        header = f"{'rank':>4} {'acc%':>6} {'ret%':>6} {'morph':>8} {'ks':>3} {'shape':>8} {'it':>3} {'tx':>4} {'ty':>4} {'hits':>7} {'ret':>7}"
        print("\n[grid] Summary (sorted by acc desc):")
        print(header)
        for i, (acc, ret_acc, morph, ksize, shape, iters, tx, ty, total, hits, rt, rh) in enumerate(results, 1):
            print(f"{i:>4} {acc:>6.2f} {ret_acc:>6.2f} {morph:>8} {ksize:>3} {shape:>8} {iters:>3} {tx:>4} {ty:>4} {hits:>3}/{total:<3} {rh:>3}/{rt:<3}")

    # Write CSV sorted
    if args.grid_out and results:
        out_path = Path(args.grid_out)
        lines = ["morph,ksize,shape,iters,tx,ty,total,hits,ret_total,ret_hits,acc,ret_acc"]
        for acc, ret_acc, morph, ksize, shape, iters, tx, ty, total, hits, rt, rh in results:
            lines.append(f"{morph},{ksize},{shape},{iters},{tx},{ty},{total},{hits},{rt},{rh},{acc:.4f},{ret_acc:.4f}")
        try:
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[grid] Wrote sorted results to {out_path}")
        except Exception as e:
            print(f"[warn] Failed to write grid output: {e}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())
