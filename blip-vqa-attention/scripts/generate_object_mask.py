import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict

# Segment Anything
from segment_anything import build_sam, SamPredictor

from huggingface_hub import hf_hub_download


def load_model_hf(repo_id: str, filename: str, ckpt_config_filename: str, device: str = "cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cfg = SLConfig.fromfile(cache_config_file)
    model = build_model(cfg)
    cfg.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"[GroundingDINO] Weights: {cache_file} => {log}")
    _ = model.eval()
    return model


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s)[:80]


def build_sam_predictor(checkpoint_path: Path, device: torch.device) -> SamPredictor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}. Provide --sam_checkpoint pointing to sam_vit_h_4b8939.pth"
        )
    sam = build_sam(checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamPredictor(sam)


def extract_mask_for_prompt(
    gdinomodel,
    sam_predictor: SamPredictor,
    device: torch.device,
    image_path: Path,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Optional[np.ndarray]:
    image_source, image = load_image(str(image_path))

    boxes, _, _ = predict(
        model=gdinomodel,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if boxes is None or len(boxes) == 0:
        return None

    sam_predictor.set_image(image_source)

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Combine all predicted boxes' masks into one binary mask
    # m = (masks.detach().cpu().numpy() > 0.5).astype(np.uint8)  # [N,1,H,W]
    # m = m.max(axis=0)  # [1,H,W]
    # return m[0]
    mask = (masks[0][0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    return mask


def run_batch(
    jsonl_path: Path,
    image_root: Path,
    out_dir: Path,
    sam_checkpoint: Path,
    device: torch.device,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    gdino_repo: str = "ShilongLiu/GroundingDINO",
    gdino_ckpt: str = "groundingdino_swinb_cogcoor.pth",
    gdino_cfg: str = "GroundingDINO_SwinB.cfg.py",
):
    repo_root = Path(__file__).resolve().parents[1]

    if not jsonl_path.is_absolute():
        jsonl_path = Path(__file__).resolve().parent / jsonl_path
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    if not image_root.is_absolute():
        image_root = (repo_root / image_root).resolve()

    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    ensure_dir(out_dir)

    print(f"[Setup] Device: {device}")
    print(f"[Setup] Image root: {image_root}")
    print(f"[Setup] Output dir: {out_dir}")

    device_str = device.type
    groundingdino_model = load_model_hf(gdino_repo, gdino_ckpt, gdino_cfg, device=device_str)
    sam_ckpt = sam_checkpoint
    if not sam_ckpt.is_absolute():
        sam_ckpt = (Path(__file__).resolve().parent / sam_ckpt).resolve()
    sam_predictor = build_sam_predictor(sam_ckpt, device)

    total = 0
    written = 0
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

            rel_path = entry.get("image") or entry.get("img")
            prompt = entry.get("prompt") or entry.get("caption") or entry.get("text")
            if not rel_path or not prompt:
                print(f"[WARN] Missing image or prompt at line {idx+1}; skipping")
                continue

            img_path = (image_root / rel_path).resolve()
            if not img_path.exists():
                alt = Path(rel_path)
                if alt.exists():
                    img_path = alt.resolve()
                else:
                    print(f"[WARN] Image not found for line {idx+1}: {img_path}")
                    continue

            stem_parts = [
                entry.get("id"),
                Path(rel_path).stem,
                sanitize_name(str(prompt)),
                f"{idx:05d}",
            ]
            stem = "_".join([str(p) for p in stem_parts if p])
            out_path = out_dir / f"{stem}.npy"

            try:
                mask = extract_mask_for_prompt(
                    groundingdino_model,
                    sam_predictor,
                    device,
                    img_path,
                    prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                total += 1
                if mask is None:
                    print(f"[INFO] No boxes for line {idx+1} ({rel_path} | '{prompt}')")
                    continue
                np.save(out_path, mask.astype(np.uint8))
                written += 1
                print(f"[OK] Saved mask: {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed at line {idx+1}: {e}")

    print(f"Done. Processed: {total}, Masks saved: {written}, Output: {out_dir}")


# ---- Simple editable config (no argparse) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = REPO_ROOT / "dataset" / "dataset.jsonl"
IMAGE_ROOT = REPO_ROOT / "dataset"
OUT_DIR = IMAGE_ROOT / "masks"
SAM_CHECKPOINT = Path("sam_vit_h_4b8939.pth")  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
GDINO_REPO = "ShilongLiu/GroundingDINO"
GDINO_CKPT = "groundingdino_swinb_cogcoor.pth"
GDINO_CFG = "GroundingDINO_SwinB.cfg.py"


if __name__ == "__main__":
    run_batch(
        jsonl_path=JSONL_PATH,
        image_root=IMAGE_ROOT,
        out_dir=OUT_DIR,
        sam_checkpoint=SAM_CHECKPOINT,
        device=DEVICE,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        gdino_repo=GDINO_REPO,
        gdino_ckpt=GDINO_CKPT,
        gdino_cfg=GDINO_CFG,
    )
