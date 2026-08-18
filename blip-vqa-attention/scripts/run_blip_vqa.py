import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip_vqa import blip_vqa

MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
image_size = 480
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "dataset" / "dataset.jsonl"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = REPO_ROOT / "out" / "blip_answers.jsonl"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BLIP VQA on a JSON/JSONL list of image-question pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSON or JSONL file (default: dataset/dataset.jsonl).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Directory used to resolve relative image paths (default: dataset/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path (default: out/blip_answers.jsonl).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--vit",
        type=str,
        default="base",
        help="Which BLIP ViT backbone to load (default: 'base').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-example console prints (useful when only writing to --output).",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(choice)


def load_items(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        items: List[Dict[str, Any]] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Line {line_no} is not a JSON object: {entry!r}")
            items.append(entry)
        return items
    else:
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("Input JSON must be an object, list of objects, or JSON lines file.")


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_image_tensor(image_path: Path, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image not found: {image_path}") from exc
    return transform(rgb).unsqueeze(0).to(device)


def generate_answers(
    entries: List[Dict[str, Any]],
    *,
    model,
    transform: transforms.Compose,
    image_root: Path,
    device: torch.device,
    quiet: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        image_rel = entry.get("image")
        question = entry.get("question")
        if not image_rel or not question:
            print(f"Skipping entry {idx}: missing 'image' or 'question' field", file=sys.stderr)
            continue
        image_path = (image_root / image_rel).expanduser().resolve()
        image_tensor = load_image_tensor(image_path, transform, device)
        with torch.no_grad():
            answer = model(image_tensor, question, train=False, inference="generate")
        if isinstance(answer, (list, tuple)):
            predicted = answer[0]
        else:
            predicted = str(answer)
        record = dict(entry)
        record["generated_answer"] = predicted
        results.append(record)
        if not quiet:
            print(f"[{idx}] {question} -> {predicted}")
    return results


def write_output(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    transform = build_transform(image_size)
    model = blip_vqa(pretrained=MODEL_URL, image_size=image_size, vit=args.vit)
    model.eval()
    model = model.to(device)

    entries = load_items(args.input)
    if not entries:
        print("No entries found in input file.", file=sys.stderr)
        return

    results = generate_answers(
        entries,
        model=model,
        transform=transform,
        image_root=args.image_root,
        device=device,
        quiet=args.quiet,
    )

    write_output(args.output, results)


if __name__ == "__main__":
    main()
