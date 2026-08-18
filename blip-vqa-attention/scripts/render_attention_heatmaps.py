import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert JSONL records with head_gates into heatmap images."
    )
    parser.add_argument(
        "--input",
        default=base_dir / "fine_tuning.jsonl",
        type=Path,
        help="Path to the JSONL file (pretty JSONL supported).",
    )
    parser.add_argument(
        "--output-dir",
        default=base_dir / "heatmaps",
        type=Path,
        help="Directory to write heatmap images.",
    )
    parser.add_argument(
        "--key",
        default="head_gates",
        help="JSON field name that contains the 2D matrix to plot.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )
    return parser.parse_args()


def _sanitize_name(value: str) -> str:
    safe = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in value)
    return safe[:120] if safe else "record"


def _iter_json_objects(text: str) -> Iterable[Dict[str, object]]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at position {idx}")
        yield obj
        idx = next_idx


def _load_records(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    return list(_iter_json_objects(text))


def _plot_heatmap(matrix: np.ndarray, title: str, output_path: Path, cmap: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    records = _load_records(args.input)
    if not records:
        raise SystemExit("No records found in input file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        matrix = record.get(args.key)
        if matrix is None:
            continue
        arr = np.array(matrix, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            continue

        idx = record.get("index", written + 1)
        rec_id = record.get("id")
        base_name = f"{idx}"
        if rec_id:
            base_name += f"_{rec_id}"
        name = _sanitize_name(base_name)
        title = f"Record {idx}"
        if rec_id:
            title += f" ({rec_id})"

        output_path = args.output_dir / f"{name}_{args.key}.png"
        _plot_heatmap(arr, title=title, output_path=output_path, cmap=args.cmap, dpi=args.dpi)
        written += 1

    print(f"Wrote {written} heatmap(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
