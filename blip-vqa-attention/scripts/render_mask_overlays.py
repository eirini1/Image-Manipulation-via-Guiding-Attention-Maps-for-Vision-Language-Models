import json
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image


def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s)[:80]


def find_mask_files(mask_root: Path, id_: Optional[str], image_stem: str, prompt: str) -> List[Path]:

    prompt_s = sanitize_name(str(prompt))
    patterns = []
    if id_:
        patterns.append(f"{id_}_{image_stem}_{prompt_s}_*.npy")
    patterns.append(f"{image_stem}_{prompt_s}_*.npy")

    matches: List[Path] = []
    for pat in patterns:
        matches = list(mask_root.glob(pat))
        if matches:
            break
    return sorted(matches)


def load_image_simple(path_like: str, image_root: Path) -> Image.Image:
    p = Path(path_like)
    return Image.open((image_root / p)).convert("RGB")


def overlay_mask(img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> Image.Image:

    if mask.ndim == 3:
        # Sometimes masks come as (1, H, W)
        mask = mask.squeeze()
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / mask.max()

    W, H = img.size
    if mask.shape != (H, W):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
        mask = np.array(mask_img, dtype=np.float32) / 255.0

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_arr = np.array(overlay, dtype=np.uint8)

    r, g, b = color
    a = (np.clip(mask, 0.0, 1.0) * (alpha * 255)).astype(np.uint8)

    overlay_arr[..., 0] = r
    overlay_arr[..., 1] = g
    overlay_arr[..., 2] = b
    overlay_arr[..., 3] = a

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def preprocess_for_blip(img: Image.Image, mask: np.ndarray, size: int = 480, patch: int = 16):

    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / mask.max()

    W, H = img.size
    scale = size / min(W, H)
    newW = int(round(W * scale))
    newH = int(round(H * scale))

    img_rs = img.resize((newW, newH), resample=Image.BICUBIC)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_rs = mask_img.resize((newW, newH), resample=Image.NEAREST)

    left = max(0, (newW - size) // 2)
    top = max(0, (newH - size) // 2)
    right = left + size
    bottom = top + size
    img_cr = img_rs.crop((left, top, right, bottom))
    mask_cr = mask_rs.crop((left, top, right, bottom))

    mask_arr = np.array(mask_cr, dtype=np.float32) / 255.0

    if size % patch != 0:
        raise ValueError(f"size {size} not divisible by patch {patch}")
    grid = size // patch
    mask_blk = mask_arr.reshape(grid, patch, grid, patch).max(axis=(1, 3))
    mask_snap = mask_blk.repeat(patch, axis=0).repeat(patch, axis=1)

    return img_cr, mask_snap


def process_jsonl(
    jsonl_path: Path,
    image_root: Path,
    mask_root: Path,
    out_dir: Path,
    alpha: float = 0.5,
    color=(255, 0, 0),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0
    missing_masks = 0
    missing_images = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {idx+1}")
                continue

            total += 1
            id_ = entry.get("id")
            image_rel = entry.get("image") or entry.get("img")
            prompt = entry.get("prompt") or entry.get("caption") or entry.get("text")
            if not image_rel or not prompt:
                print(f"[WARN] Missing image or prompt at line {idx+1}; skipping")
                continue

            img_stem = Path(image_rel).stem
            mask_files = find_mask_files(mask_root, id_, img_stem, str(prompt))
            if not mask_files:
                print(
                    f"[MISS] No mask for id={id_} image={image_rel} prompt='{prompt}' in {mask_root}"
                )
                missing_masks += 1
                continue

            mask_path = mask_files[-1]  # pick last (e.g., highest index)
            try:
                img = load_image_simple(image_rel, image_root)
            except Exception as e:
                print(f"[MISS] Failed to open image {image_rel} under {image_root}: {e}")
                missing_images += 1
                continue

            try:
                mask = np.load(mask_path)
            except Exception as e:
                print(f"[ERROR] Failed to load mask {mask_path}: {e}")
                continue

            img_p, mask_p = preprocess_for_blip(img, mask, size=480, patch=16)
            vis = overlay_mask(img_p, mask_p, color=color, alpha=alpha)

            out_name = f"{mask_path.stem}_overlay_blip480.png"
            out_path = out_dir / out_name
            try:
                vis.save(out_path)
                saved += 1
                print(f"[OK] Saved: {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save {out_path}: {e}")

    print(
        f"Done. entries={total} saved={saved} missing_masks={missing_masks} missing_images={missing_images} output={out_dir}"
    )



REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = REPO_ROOT / "dataset" / "dataset.jsonl"
IMAGE_ROOT = REPO_ROOT / "dataset"
MASK_ROOT = REPO_ROOT / "dataset" / "masks"
OUT_DIR = REPO_ROOT / "out" / "mask_overlays"
ALPHA = 0.5
COLOR = (255, 0, 0)


if __name__ == "__main__":
    process_jsonl(
        jsonl_path=JSONL_PATH,
        image_root=IMAGE_ROOT,
        mask_root=MASK_ROOT,
        out_dir=OUT_DIR,
        alpha=ALPHA,
        color=COLOR,
    )
