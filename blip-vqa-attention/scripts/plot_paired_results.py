''' Plots delta histograms for stage-2 paired probability matrices. (results are from probabilities3.py) '''

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _default_input_dir() -> Path:
    base = Path("/gpu-data2/emil/BLIP/probs")
    stage2 = base / "stage2"
    return stage2 if stage2.exists() else base


def _sanitize_name(value: str) -> str:
    cleaned = []
    for ch in str(value or ""):
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    return name[:80] or "record"


def _parse_record_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _read_matrix_csv(path: Path) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    record_ids: List[str] = []
    before: List[float] = []
    after: List[float] = []
    beam_before: List[float] = []
    beam_after: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("record_id", "").strip()
            if rid == "":
                continue
            try:
                b = float(row.get("prob_before", ""))
                a = float(row.get("prob_after", ""))
            except ValueError:
                continue
            record_ids.append(rid)
            before.append(b)
            after.append(a)
    return record_ids, before, after, beam_before, beam_after


def _plot_hist(
    deltas: List[float],
    out_path: Path,
    title: str,
    x_label: str,
    alpha: float,
    dpi: int,
    *,
    bins: int,
    figsize: Tuple[float, float] = (7.0, 5.0),
) -> None:
    if not deltas:
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(deltas, bins=bins, color="#2a6fbb", alpha=alpha, edgecolor="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    avg_delta = sum(deltas) / len(deltas)
    ax.set_title(f"{title} (n={len(deltas)}, mean Delta={avg_delta:+.4f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)



def _plot_record_hist(
    record_id: str,
    before: List[float],
    after: List[float],
    beam_before: List[float],
    beam_after: List[float],
    out_path: Path,
    alpha: float,
    dpi: int,
    *,
    use_beam: bool = False,
) -> None:
    if use_beam:
        deltas = [a - b for a, b in zip(beam_after, beam_before)]
        title_prefix = "Beam"
        x_label = "Delta beam (after - before)"
    else:
        deltas = [a - b for a, b in zip(after, before)]
        title_prefix = "Prob"
        x_label = "Delta (after - before)"

    _plot_hist(
        deltas,
        out_path,
        f"{title_prefix} {record_id}",
        x_label,
        alpha,
        dpi,
        bins=30,
        figsize=(7.0, 5.0),
    )


def _plot_combined(
    files: List[Path],
    out_path: Path,
    alpha: float,
    dpi: int,
    *,
    use_beam: bool = False,
) -> None:
    all_ids: List[str] = []
    all_before: List[float] = []
    all_after: List[float] = []
    all_beam_before: List[float] = []
    all_beam_after: List[float] = []
    for path in files:
        record_ids, before, after, beam_before, beam_after = _read_matrix_csv(path)
        all_ids.extend(record_ids)
        all_before.extend(before)
        all_after.extend(after)
        all_beam_before.extend(beam_before)
        all_beam_after.extend(beam_after)

    if not all_ids:
        return

    if use_beam:
        deltas = [a - b for a, b in zip(all_beam_after, all_beam_before)]
        title_prefix = "Beam"
        x_label = "Delta beam (after - before)"
    else:
        deltas = [a - b for a, b in zip(all_after, all_before)]
        title_prefix = "Prob"
        x_label = "Delta (after - before)"

    _plot_hist(
        deltas,
        out_path,
        f"{title_prefix} (files={len(files)})",
        x_label,
        alpha,
        dpi,
        bins=40,
        figsize=(7.2, 5.2),
    )


def _collect_per_record(files: List[Path]) -> Dict[str, Dict[str, List[float]]]:
    per_record: Dict[str, Dict[str, List[float]]] = {}
    for path in files:
        record_ids, before, after, beam_before, beam_after = _read_matrix_csv(path)
        for rid, b, a, bb, ba in zip(record_ids, before, after, beam_before, beam_after):
            entry = per_record.get(rid)
            if entry is None:
                entry = {
                    "before": [],
                    "after": [],
                    "beam_before": [],
                    "beam_after": [],
                }
                per_record[rid] = entry
            entry["before"].append(b)
            entry["after"].append(a)
            entry["beam_before"].append(bb)
            entry["beam_after"].append(ba)
    return per_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot delta histograms for stage-2 matrices (per record by default).")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing stage-2 matrix CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="probs/plots",
        help="Directory to write PNG plots.",
    )
    parser.add_argument(
        "--pattern",
        default="matrix_heads-*_layers-*.csv",
        help="Glob pattern for matrix CSV files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to plot (0 = no limit).",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all matching files into a single plot across all records.",
    )
    parser.add_argument(
        "--combine-out",
        default="combined_prob.pdf",
        help="Filename for the combined prob plot (written to output dir).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Histogram transparency.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI.",
    )
    parser.add_argument(
        "--record-ids",
        default=None,
        help="Optional comma-separated record ids to plot.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on number of records to plot (0 = no limit).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else _default_input_dir()
    output_dir = Path(args.output_dir)

    files = sorted(input_dir.glob(args.pattern))
    print(f"[debug] matched_files={len(files)} pattern={args.pattern} input_dir={input_dir}")
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if args.combine:
        out_path = output_dir / args.combine_out
        _plot_combined(files, out_path, alpha=args.alpha, dpi=args.dpi, use_beam=False)
        beam_out = output_dir / f"{Path(args.combine_out).stem}_beam.pdf"
        _plot_combined(files, beam_out, alpha=args.alpha, dpi=args.dpi, use_beam=True)
    else:
        per_record = _collect_per_record(files)
        print(f"[debug] record_ids_found={len(per_record)}")
        record_ids = _parse_record_ids(args.record_ids)
        if record_ids:
            target_ids = [rid for rid in record_ids if rid in per_record]
        else:
            target_ids = sorted(per_record.keys())
        if record_ids and not target_ids:
            print(f"[debug] record_ids_filter_empty requested={record_ids} available={sorted(per_record.keys())}")
        if args.max_records and args.max_records > 0:
            target_ids = target_ids[: args.max_records]

        for rid in target_ids:
            entry = per_record.get(rid)
            if not entry:
                continue
            safe_id = _sanitize_name(rid)
            out_path = output_dir / f"{safe_id}_prob.pdf"
            _plot_record_hist(
                rid,
                entry["before"],
                entry["after"],
                entry["beam_before"],
                entry["beam_after"],
                out_path,
                alpha=args.alpha,
                dpi=args.dpi,
                use_beam=False,
            )
            beam_out = output_dir / f"{safe_id}_beam.pdf"
            _plot_record_hist(
                rid,
                entry["before"],
                entry["after"],
                entry["beam_before"],
                entry["beam_after"],
                beam_out,
                alpha=args.alpha,
                dpi=args.dpi,
                use_beam=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
