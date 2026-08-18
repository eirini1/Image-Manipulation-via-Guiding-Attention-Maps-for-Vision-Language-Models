import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _collect_paths(jsonl_args: Sequence[str], glob_pattern: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    seen: Set[str] = set()

    for raw in jsonl_args:
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)

    if glob_pattern:
        for raw in sorted(glob.glob(glob_pattern, recursive=True)):
            p = Path(raw)
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)

    return paths


def _pick_candidate_norm(row: Dict[str, Any]) -> str:
    value = row.get("candidate_answer_norm")
    if value is not None:
        out = _norm_text(value)
        if out:
            return out
    return _norm_text(row.get("candidate_answer"))


def _pick_gold_norm(row: Dict[str, Any]) -> str:
    value = row.get("gold_answer_norm")
    if value is not None:
        out = _norm_text(value)
        if out:
            return out
    return _norm_text(row.get("gold_answer"))


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "gt_answer_norm",
        "gt_in_union",
        "union_size",
        "gold_conflict",
        "num_source_rows",
        "num_source_files",
        "source_files",
        "union_answers",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row.get("record_id", ""),
                    "gt_answer_norm": row.get("gt_answer_norm", ""),
                    "gt_in_union": row.get("gt_in_union", ""),
                    "union_size": row.get("union_size", 0),
                    "gold_conflict": row.get("gold_conflict", 0),
                    "num_source_rows": row.get("num_source_rows", 0),
                    "num_source_files": row.get("num_source_files", 0),
                    "source_files": ";".join(row.get("source_files", [])),
                    "union_answers": ";".join(row.get("union_answers", [])),
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check if each record's GT answer appears in the union of candidate answers "
            "across loaded candidate_stability_features*_new_answers JSONL rows."
        )
    )
    parser.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help="Input JSONL path. Repeat for multiple files.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Glob pattern for input JSONL files (supports recursive **).",
    )
    parser.add_argument(
        "--out-jsonl",
        default="gt_in_union_report.jsonl",
        help="Path to write per-record report JSONL.",
    )
    parser.add_argument(
        "--out-csv",
        default="gt_in_union_report.csv",
        help="Path to write per-record report CSV.",
    )
    parser.add_argument(
        "--show-misses",
        type=int,
        default=20,
        help="Print up to N records where GT is not in the union.",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Return exit code 2 if any record with known GT misses the union.",
    )
    args = parser.parse_args(argv)

    paths = _collect_paths(args.jsonl, args.glob)
    if not paths:
        parser.error("Provide --jsonl and/or --glob.")

    union_by_record: Dict[str, Set[str]] = defaultdict(set)
    gold_by_record: Dict[str, Set[str]] = defaultdict(set)
    rows_by_record: Dict[str, int] = defaultdict(int)
    files_by_record: Dict[str, Set[str]] = defaultdict(set)
    loaded_files = 0

    for path in paths:
        if not path.exists():
            print(f"[warn] Missing file: {path}; skipping")
            continue
        rows = _iter_jsonl(path)
        if not rows:
            print(f"[warn] Empty/invalid JSONL: {path}; skipping")
            continue
        loaded_files += 1
        resolved_source = str(path.resolve())
        for row in rows:
            rec_id = _norm_text(row.get("record_id"))
            if not rec_id:
                continue
            cand = _pick_candidate_norm(row)
            if cand:
                union_by_record[rec_id].add(cand)
            gold = _pick_gold_norm(row)
            if gold:
                gold_by_record[rec_id].add(gold)
            rows_by_record[rec_id] += 1
            files_by_record[rec_id].add(resolved_source)

    if loaded_files == 0:
        print("[error] No readable input files.")
        return 1

    all_record_ids = sorted(set(rows_by_record.keys()) | set(union_by_record.keys()) | set(gold_by_record.keys()))
    report_rows: List[Dict[str, Any]] = []

    known_gold = 0
    hit_count = 0
    miss_ids: List[str] = []
    conflict_count = 0

    for rec_id in all_record_ids:
        union_answers = sorted(union_by_record.get(rec_id, set()))
        gold_set = gold_by_record.get(rec_id, set())

        gt_answer_norm = ""
        gold_conflict = 0
        gt_in_union: Optional[int] = None

        if len(gold_set) == 1:
            gt_answer_norm = next(iter(gold_set))
            gt_in_union = int(gt_answer_norm in union_by_record.get(rec_id, set()))
            known_gold += 1
            if gt_in_union == 1:
                hit_count += 1
            else:
                miss_ids.append(rec_id)
        elif len(gold_set) > 1:
            gold_conflict = 1
            conflict_count += 1

        report_rows.append(
            {
                "record_id": rec_id,
                "gt_answer_norm": gt_answer_norm,
                "gt_in_union": gt_in_union,
                "union_size": len(union_answers),
                "gold_conflict": gold_conflict,
                "num_source_rows": int(rows_by_record.get(rec_id, 0)),
                "num_source_files": len(files_by_record.get(rec_id, set())),
                "source_files": sorted(files_by_record.get(rec_id, set())),
                "union_answers": union_answers,
            }
        )

    if args.out_jsonl:
        _write_jsonl(Path(args.out_jsonl), report_rows)
        print(f"[info] Wrote JSONL report: {args.out_jsonl}")
    if args.out_csv:
        _write_csv(Path(args.out_csv), report_rows)
        print(f"[info] Wrote CSV report: {args.out_csv}")

    total_records = len(all_record_ids)
    miss_count = len(miss_ids)
    hit_rate = (100.0 * hit_count / known_gold) if known_gold > 0 else float("nan")
    print(
        "[summary] "
        f"files_loaded={loaded_files} records={total_records} "
        f"known_gt={known_gold} hits={hit_count} misses={miss_count} "
        f"hit_rate={hit_rate:.2f}% conflicts={conflict_count}"
    )

    if miss_ids and int(args.show_misses) > 0:
        shown = miss_ids[: int(args.show_misses)]
        print(f"[misses] showing {len(shown)}/{len(miss_ids)} record_id values:")
        for rec_id in shown:
            print(f"  - {rec_id}")

    if args.fail_on_miss and miss_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
