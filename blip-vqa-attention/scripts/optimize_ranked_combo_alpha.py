import argparse
from collections import defaultdict
import glob
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import (
    apply_override,
    canonicalize_answer,
    data_path,
    ensure_mask,
    gaussian_from_mask,
    guess_focus_words,
    image_root,
    iter_jsonl,
    load_mask_from_dir,
    masks_root,
    normalize_answer,
    revert_override,
    select_override_indices,
)

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root


def _answer_key(text: str) -> str:
    canon = canonicalize_answer(text)
    if canon:
        return canon
    return normalize_answer(text)


def _answer_display(text: str) -> str:
    canon = canonicalize_answer(text)
    return canon if canon else (text or "").strip()


def _iter_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")

    # Fast path: strict JSONL (one JSON object per line).
    line_rows: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            line_rows = []
            break
        if isinstance(row, dict):
            line_rows.append(row)
    if line_rows:
        return line_rows

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if end <= idx:
            break
        if isinstance(obj, dict):
            rows.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    rows.append(item)
        idx = end

    return rows


def _expand_template(value: str, *, record_id: Optional[str] = None, top_k: Optional[int] = None) -> str:
    out = str(value)
    if top_k is not None:
        out = out.replace("{k}", str(int(top_k)))
    if record_id is not None:
        out = out.replace("{record_id}", str(record_id))
    return out


def _collect_input_paths(
    jsonl_args: Sequence[str],
    glob_pattern: Optional[str],
    *,
    record_ids: Sequence[str],
    top_k: Optional[int],
) -> List[Path]:
    paths: List[Path] = []
    seen: Set[str] = set()

    record_values: List[Optional[str]] = [None]
    if record_ids:
        record_values = [str(v) for v in record_ids]

    for raw in jsonl_args:
        raw_s = str(raw)
        if "{record_id}" in raw_s:
            for rid in record_values:
                resolved = _expand_template(raw_s, record_id=rid, top_k=top_k)
                p = Path(resolved)
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(p)
        else:
            resolved = _expand_template(raw_s, top_k=top_k)
            p = Path(resolved)
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)

    if glob_pattern:
        patterns: List[str] = []
        if "{record_id}" in str(glob_pattern):
            for rid in record_values:
                patterns.append(_expand_template(str(glob_pattern), record_id=rid, top_k=top_k))
        else:
            patterns.append(_expand_template(str(glob_pattern), top_k=top_k))
        for pattern in patterns:
            for raw in sorted(glob.glob(str(pattern), recursive=True)):
                p = Path(raw)
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(p)
    return paths


def _load_baseline_topk_answers_by_record(
    paths: Sequence[Path],
    *,
    top_k_limit: int,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    out: Dict[str, Dict[str, Any]] = {}
    loaded_files = 0
    for path in paths:
        if not path.exists():
            print(f"[warn] Missing baseline-topk file: {path}; skipping")
            continue
        rows = _iter_jsonl_rows(path)
        if not rows:
            print(f"[warn] Empty/invalid baseline-topk file: {path}; skipping")
            continue
        loaded_files += 1
        source_file = str(path.resolve())
        for row in rows:
            rec_id = str(row.get("id", row.get("record_id", ""))).strip().lower()
            if not rec_id:
                continue
            baseline_topk = row.get("baseline_topk", None)
            if not isinstance(baseline_topk, list) or not baseline_topk:
                continue
            rec = out.setdefault(
                rec_id,
                {
                    "answer_order": [],
                    "answer_text_by_norm": {},
                    "rank_positions_by_norm": defaultdict(set),
                    "source_files": set(),
                    "row_count": 0,
                },
            )
            limit = max(1, int(top_k_limit))
            for rank_idx, ans_value in enumerate(baseline_topk[:limit], 1):
                ans_text = str(ans_value or "").strip()
                if not ans_text:
                    continue
                ans_key = _answer_key(ans_text)
                if not ans_key:
                    continue
                ans_display = _answer_display(ans_text)
                if ans_key not in rec["answer_text_by_norm"]:
                    rec["answer_order"].append(ans_key)
                    rec["answer_text_by_norm"][ans_key] = ans_display
                rec["rank_positions_by_norm"][ans_key].add(int(rank_idx))
            rec["source_files"].add(source_file)
            rec["row_count"] = int(rec["row_count"]) + 1
    return out, loaded_files


def _prepare_question_inputs(tokenizer, question: str, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        question,
        padding="longest",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"][:, 0] = tokenizer.enc_token_id
    return inputs


def _parse_pair_combo_spec(spec: str) -> Dict[int, Tuple[int, ...]]:
    if not spec.startswith("pairs:"):
        return {}
    body = spec[len("pairs:") :].strip()
    if not body:
        return {}

    layer_to_heads: Dict[int, set] = defaultdict(set)
    for chunk in body.split(","):
        token = chunk.strip().lower()
        if not token or "h" not in token or not token.startswith("l"):
            continue
        layer_part, head_part = token[1:].split("h", 1)
        try:
            layer_idx = int(layer_part)
            head_idx = int(head_part)
        except ValueError:
            continue
        layer_to_heads[layer_idx].add(head_idx)

    parsed: Dict[int, Tuple[int, ...]] = {}
    for layer_idx, heads in layer_to_heads.items():
        parsed[layer_idx] = tuple(sorted(int(h) for h in heads))
    return parsed


def _encode_pair_combo_spec(pairs: Sequence[Tuple[int, int]]) -> str:
    ordered = sorted((int(layer), int(head)) for layer, head in pairs)
    return "pairs:" + ",".join(f"l{layer}h{head}" for layer, head in ordered)


def _sample_ranked_pair_combo_specs(
    ranked_pairs: Sequence[Dict[str, Any]],
    *,
    combo_count: int,
    top_pick_min: int,
    top_pick_max: int,
    next_window: int,
    next_pick_min: int,
    next_pick_max: int,
    seed: Optional[int],
) -> List[str]:
    if combo_count <= 0:
        return []

    dedup_pairs: List[Tuple[int, int]] = []
    seen_pairs = set()
    for item in ranked_pairs:
        try:
            layer = int(item.get("layer"))
            head = int(item.get("head"))
        except Exception:
            continue
        key = (layer, head)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        dedup_pairs.append(key)

    top_pool_size = min(max(int(top_pick_max), 0), len(dedup_pairs))
    top_pool = list(dedup_pairs[:top_pool_size])
    next_start = top_pool_size
    next_pool = list(dedup_pairs[next_start : next_start + max(int(next_window), 0)])

    top_hi = len(top_pool)
    next_hi = len(next_pool)
    if top_hi <= 0 or next_hi <= 0:
        return []

    top_lo = min(max(int(top_pick_min), 0), top_hi)
    next_lo = min(max(int(next_pick_min), 0), next_hi)
    if top_lo <= 0 or next_lo <= 0:
        return []

    top_hi = min(top_hi, max(int(top_pick_max), 0))
    next_hi = min(next_hi, max(int(next_pick_max), 0))
    if top_hi < top_lo or next_hi < next_lo:
        return []

    rng = random.Random(seed)
    selected_specs: List[str] = []
    seen_combo = set()
    max_attempts = max(200, combo_count * 200)
    attempts = 0
    while len(selected_specs) < combo_count and attempts < max_attempts:
        attempts += 1
        n_top = rng.randint(top_lo, top_hi)
        n_next = rng.randint(next_lo, next_hi)
        top_sel = rng.sample(top_pool, n_top)
        next_sel = rng.sample(next_pool, n_next)
        pairs = set(top_sel)
        pairs.update(next_sel)
        combo_key = tuple(sorted(pairs))
        if not combo_key or combo_key in seen_combo:
            continue
        seen_combo.add(combo_key)
        selected_specs.append(_encode_pair_combo_spec(combo_key))
    return selected_specs


def _resolve_topk_path_template(path_value: str, top_k: int) -> Path:
    if "{k}" in path_value:
        return Path(path_value.format(k=int(top_k)))
    return Path(path_value)


def _parse_record_ids(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        out.extend([part.strip().lower() for part in str(value).split(",") if part.strip()])
    return out


def _record_matches(entry: Dict[str, Any], targets: Sequence[str]) -> bool:
    if not targets:
        return True
    rec_id = str(entry.get("id", "")).strip().lower()
    return rec_id in set(targets)


def _build_cached_record(
    entry: Dict[str, Any],
    *,
    model,
    tokenizer,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    rec_id_val = entry.get("id")
    rec_id = "" if rec_id_val is None else str(rec_id_val)
    image_value = entry.get("image")
    question = entry.get("question")
    prompt = entry.get("prompt")
    if image_value is None or question is None:
        print(f"[warn] Missing fields for record={rec_id}; skipping", file=sys.stderr)
        return None

    image_rel = Path(image_value)
    image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
    if not image_path.exists():
        print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
        return None

    image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
    question_inputs = _prepare_question_inputs(tokenizer, question, image.device)

    stem = image_rel.stem
    rec_id_for_mask = rec_id if rec_id else None
    mask_array = load_mask_from_dir(MASK_DIR, stem, str(prompt or ""), rec_id_for_mask)
    gh = gw = IMAGE_SIZE // PATCH_SIZE
    try:
        mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
    except RuntimeError as exc:
        print(f"[warn] ensure_mask failed for record={rec_id}: {exc}; skipping", file=sys.stderr)
        return None

    mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32).view(1, 1, *mask_array.shape)
    mask_small = F.interpolate(mask_tensor, size=(gh, gw), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    mask_small = mask_small.clamp_(0.0, 1.0)
    mask_soft = soften_mask(mask_small, ksize=5, iters=2)
    mask_soft = (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)

    tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
    focus_words = guess_focus_words(question)
    override_indices = select_override_indices(tokens, focus_words, tokenizer)
    if not override_indices:
        override_indices = [0]
    override_rows = {i: mask_soft for i in override_indices}

    return {
        "record_id": rec_id,
        "question": str(question),
        "image": image,
        "gold_answer": str(entry.get("answer", "") or ""),
        "gold_answer_norm": _answer_key(str(entry.get("answer", "") or "")),
        "override_rows": override_rows,
    }


def _apply_pair_combo(
    *,
    pair_layer_heads: Dict[int, Tuple[int, ...]],
    override_rows: Dict[int, torch.Tensor],
    originals,
    alpha_like,
) -> Tuple[int, ...]:
    selected_layers: List[int] = []
    for layer_idx in sorted(pair_layer_heads.keys()):
        heads_for_layer = pair_layer_heads.get(layer_idx, ())
        if not heads_for_layer:
            continue
        heads_arg: Any = heads_for_layer[0] if len(heads_for_layer) == 1 else heads_for_layer
        apply_override(make_forward(heads_arg, override_rows, alpha_like), (layer_idx,), originals)
        selected_layers.append(int(layer_idx))
    return tuple(selected_layers)


def _optimize_alpha_for_combo_answer(
    *,
    model,
    record: Dict[str, Any],
    pair_layer_heads: Dict[int, Tuple[int, ...]],
    answer: str,
    steps: int,
    lr: float,
    originals,
    device: torch.device,
) -> float:
    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.Adam([raw_alpha], lr=float(lr))
    image = record["image"]
    question = str(record["question"])

    for _ in range(max(1, int(steps))):
        alpha = torch.sigmoid(raw_alpha)
        applied_layers: Tuple[int, ...] = ()
        try:
            applied_layers = _apply_pair_combo(
                pair_layer_heads=pair_layer_heads,
                override_rows=record["override_rows"],
                originals=originals,
                alpha_like=alpha,
            )
            weights = torch.tensor([1.0], device=device)
            loss = model(image, [question], answer=[answer], train=True, n=[1], weights=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        finally:
            if applied_layers:
                revert_override(applied_layers, originals)
    return float(torch.sigmoid(raw_alpha).detach().item())


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl_with_record_template(
    path_template: str,
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
) -> List[Path]:
    tmpl = str(path_template)
    if "{record_id}" not in tmpl:
        out_path = Path(_expand_template(tmpl, top_k=top_k))
        _write_jsonl(out_path, rows)
        return [out_path]

    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rec_id = str(row.get("record_id", "")).strip()
        if not rec_id:
            continue
        by_record[rec_id].append(dict(row))

    written: List[Path] = []
    for rec_id in sorted(by_record.keys()):
        out_path = Path(_expand_template(tmpl, record_id=rec_id, top_k=top_k))
        _write_jsonl(out_path, by_record[rec_id])
        written.append(out_path)
    return written


def _resolve_record_output_path(
    path_template: str,
    *,
    record_id: str,
    top_k: int,
    force_record_suffix: bool = False,
) -> Path:
    tmpl = str(path_template)
    has_record_placeholder = "{record_id}" in tmpl
    resolved = _expand_template(tmpl, record_id=record_id, top_k=top_k)
    out_path = Path(resolved)
    if force_record_suffix and not has_record_placeholder:
        stem = out_path.stem
        suffix = out_path.suffix
        out_path = out_path.with_name(f"{stem}_{record_id}{suffix}")
    return out_path


def _plot_alpha_histograms_for_record(
    *,
    record_id: str,
    answer_summaries: Sequence[Dict[str, Any]],
    out_path: Path,
    bins: int,
) -> None:
    n_answers = len(answer_summaries)
    if n_answers <= 0:
        return

    cols = 3
    rows = int(math.ceil(float(n_answers) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.8 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    flat_axes = [ax for row_axes in axes_arr for ax in row_axes]

    for idx, summary in enumerate(answer_summaries):
        ax = flat_axes[idx]
        answer = str(summary.get("answer", ""))
        values = [float(v) for v in summary.get("alpha_values", [])]
        if values:
            ax.hist(
                values,
                bins=max(1, int(bins)),
                color="#2a6fbb",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            mean_v = float(np.mean(np.asarray(values, dtype=np.float64)))
            ax.axvline(mean_v, color="#b22222", linewidth=1.2)
            ax.set_title(f"{answer}\nn={len(values)} mean={mean_v:.3f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_title(f"{answer}\nn=0", fontsize=9)
        ax.set_xlabel("Best alpha")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in flat_axes[n_answers:]:
        ax.axis("off")

    fig.suptitle(f"Alpha histograms per answer | record_id={record_id}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per record: sample ranked pair-combos, load baseline answers from "
            "fine_tuning_L baseline_topk JSONL files, and "
            "optimize alpha for each (combo, baseline-answer)."
        )
    )
    parser.add_argument("--data-path", default=None, help="Path to dataset JSONL.")
    parser.add_argument("--record-id", action="append", default=[], help="Record id(s); repeat or comma-separate.")
    parser.add_argument("--rankings-jsonl", required=True, help="Rankings JSONL path (supports {k} template).")
    parser.add_argument("--rankings-top-k", type=int, default=3, help="Value used for {k} in --rankings-jsonl.")
    parser.add_argument("--ranked-combos", type=int, required=True, help="Number of ranked pair-combos to sample.")
    parser.add_argument("--ranked-top-min", type=int, default=25)
    parser.add_argument("--ranked-top-max", type=int, default=30)
    parser.add_argument("--ranked-next-window", type=int, default=60)
    parser.add_argument("--ranked-next-min", type=int, default=30)
    parser.add_argument("--ranked-next-max", type=int, default=40)
    parser.add_argument("--ranked-seed", type=int, default=0)
    parser.add_argument(
        "--answers-jsonl",
        action="append",
        default=[],
        help="Input fine_tuning_L results JSONL path(s) containing baseline_topk. Repeat for multiple files.",
    )
    parser.add_argument(
        "--answers-glob",
        default=None,
        help="Glob pattern for fine_tuning_L results JSONL files (supports recursive '**').",
    )
    parser.add_argument(
        "--baseline-topk-size",
        type=int,
        default=10,
        help="Use first K answers from baseline_topk per record.",
    )
    parser.add_argument("--steps-per-answer", type=int, default=30, help="Alpha optimization steps for each pair.")
    parser.add_argument("--alpha-lr", type=float, default=0.05, help="Alpha optimizer learning rate.")
    parser.add_argument(
        "--out-record-jsonl",
        default="out/ranked_combo_alpha_union_per_record.jsonl",
        help="Per-record alpha distribution output JSONL.",
    )
    parser.add_argument(
        "--out-pair-jsonl",
        default="out/ranked_combo_alpha_union_per_pair.jsonl",
        help="Per (record,combo,answer) best-alpha output JSONL.",
    )
    parser.add_argument(
        "--out-hist",
        default="out/{record_id}/ranked_alpha_histograms.pdf",
        help=(
            "Output path template for per-record alpha histogram figure. "
            "Supports {record_id} and {k}. Use empty string to disable plotting."
        ),
    )
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins per answer.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on number of processed records.")
    args = parser.parse_args(argv)

    if args.ranked_combos <= 0:
        parser.error("--ranked-combos must be > 0.")
    if args.ranked_top_min <= 0 or args.ranked_top_max <= 0:
        parser.error("--ranked-top-min and --ranked-top-max must be > 0.")
    if args.ranked_next_min <= 0 or args.ranked_next_max <= 0:
        parser.error("--ranked-next-min and --ranked-next-max must be > 0.")
    if args.baseline_topk_size <= 0:
        parser.error("--baseline-topk-size must be > 0.")
    if args.hist_bins <= 0:
        parser.error("--hist-bins must be > 0.")
    dataset_path = Path(args.data_path) if args.data_path else data_path
    records = list(iter_jsonl(dataset_path))
    if not records:
        print(f"[error] No records in {dataset_path}.")
        return 1

    targets = _parse_record_ids(args.record_id)
    selected_records = [entry for entry in records if _record_matches(entry, targets)]
    if args.max_records > 0:
        selected_records = selected_records[: int(args.max_records)]
    if not selected_records:
        print("[error] No selected records.")
        return 1

    selected_record_ids_for_templates: List[str] = []
    seen_selected_ids: Set[str] = set()
    for entry in selected_records:
        rec_id_raw = str(entry.get("id", "")).strip()
        if not rec_id_raw or rec_id_raw in seen_selected_ids:
            continue
        seen_selected_ids.add(rec_id_raw)
        selected_record_ids_for_templates.append(rec_id_raw)

    answer_paths = _collect_input_paths(
        args.answers_jsonl,
        args.answers_glob,
        record_ids=selected_record_ids_for_templates,
        top_k=int(args.rankings_top_k),
    )
    if not answer_paths:
        parser.error("--answers-jsonl and/or --answers-glob is required.")

    baseline_answers_by_record, loaded_answer_files = _load_baseline_topk_answers_by_record(
        answer_paths,
        top_k_limit=int(args.baseline_topk_size),
    )
    if loaded_answer_files <= 0 or not baseline_answers_by_record:
        print("[error] No baseline_topk rows loaded from answers files.")
        return 1
    print(
        f"[info] Loaded baseline_topk answers for {len(baseline_answers_by_record)} records "
        f"from {loaded_answer_files} file(s), top_k={int(args.baseline_topk_size)}."
    )

    rankings_path = _resolve_topk_path_template(str(args.rankings_jsonl), int(args.rankings_top_k))
    if not rankings_path.exists():
        print(f"[error] Rankings file not found: {rankings_path}")
        return 1
    rankings_rows = _iter_jsonl_rows(rankings_path)
    ranked_pairs_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in rankings_rows:
        rec_id = str(row.get("id", "")).strip().lower()
        ranked_pairs = row.get("ranked_pairs", [])
        if rec_id and isinstance(ranked_pairs, list):
            ranked_pairs_by_id[rec_id] = [item for item in ranked_pairs if isinstance(item, dict)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = model.tokenizer

    originals = []
    for layer in model.text_encoder.encoder.layer:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    per_record_rows: List[Dict[str, Any]] = []
    per_pair_rows: List[Dict[str, Any]] = []
    hist_written_paths: List[Path] = []

    for rec_idx, entry in enumerate(selected_records, 1):
        rec_id = str(entry.get("id", "")).strip()
        rec_id_norm = rec_id.lower()
        ranked_pairs = ranked_pairs_by_id.get(rec_id_norm, [])
        if not ranked_pairs:
            print(f"[warn] No ranked_pairs for record_id={rec_id}; skipping")
            continue

        cached = _build_cached_record(entry, model=model, tokenizer=tokenizer, device=device)
        if not cached:
            continue

        combo_specs = _sample_ranked_pair_combo_specs(
            ranked_pairs,
            combo_count=int(args.ranked_combos),
            top_pick_min=int(args.ranked_top_min),
            top_pick_max=int(args.ranked_top_max),
            next_window=int(args.ranked_next_window),
            next_pick_min=int(args.ranked_next_min),
            next_pick_max=int(args.ranked_next_max),
            seed=int(args.ranked_seed),
        )
        if not combo_specs:
            print(f"[warn] No ranked combos sampled for record_id={rec_id}; skipping")
            continue

        parsed_combos: List[Dict[int, Tuple[int, ...]]] = []
        for spec in combo_specs:
            parsed = _parse_pair_combo_spec(spec)
            if parsed:
                parsed_combos.append(parsed)
            else:
                parsed_combos.append({})

        print(
            f"[info] Record {rec_idx}/{len(selected_records)} id={rec_id} | "
            f"sampled_combos={len(combo_specs)}"
        )

        source_row = baseline_answers_by_record.get(rec_id_norm, None)
        if source_row is None:
            print(f"[warn] No baseline_topk source rows for record_id={rec_id}; skipping")
            continue
        union_answer_order = list(source_row.get("answer_order", []))
        union_answer_text_by_norm = dict(source_row.get("answer_text_by_norm", {}))
        source_rank_positions_by_norm = source_row.get("rank_positions_by_norm", defaultdict(set))
        source_files = sorted(str(v) for v in source_row.get("source_files", set()))
        source_row_count = int(source_row.get("row_count", 0))

        if not union_answer_order:
            print(f"[warn] Empty baseline_topk answers for record_id={rec_id}; skipping")
            continue

        alpha_values_by_answer: Dict[str, List[float]] = {k: [] for k in union_answer_order}

        for combo_idx, pair_map in enumerate(parsed_combos, 1):
            print(
                f"[info]   optimizing combo {combo_idx}/{len(parsed_combos)} "
                f"over {len(union_answer_order)} union answers"
            )
            for answer_norm in union_answer_order:
                answer_text = union_answer_text_by_norm[answer_norm]
                best_alpha = _optimize_alpha_for_combo_answer(
                    model=model,
                    record=cached,
                    pair_layer_heads=pair_map,
                    answer=answer_text,
                    steps=int(args.steps_per_answer),
                    lr=float(args.alpha_lr),
                    originals=originals,
                    device=device,
                )
                alpha_values_by_answer[answer_norm].append(best_alpha)
                per_pair_rows.append(
                    {
                        "record_id": rec_id,
                        "combo_index": combo_idx,
                        "combo_spec": combo_specs[combo_idx - 1],
                        "answer": answer_text,
                        "answer_norm": answer_norm,
                        "best_alpha": float(best_alpha),
                    }
                )

        answer_summaries: List[Dict[str, Any]] = []
        for answer_norm in union_answer_order:
            values = np.asarray(alpha_values_by_answer[answer_norm], dtype=np.float64)
            if values.size == 0:
                continue
            answer_summaries.append(
                {
                    "answer": union_answer_text_by_norm[answer_norm],
                    "answer_norm": answer_norm,
                    "alpha_values": [float(v) for v in values.tolist()],
                    "alpha_mean": float(np.mean(values)),
                    "alpha_std": float(np.std(values)),
                    "alpha_min": float(np.min(values)),
                    "alpha_max": float(np.max(values)),
                    "alpha_p25": float(np.quantile(values, 0.25)),
                    "alpha_p50": float(np.quantile(values, 0.50)),
                    "alpha_p75": float(np.quantile(values, 0.75)),
                    "source_rank_positions": sorted(
                        int(v) for v in source_rank_positions_by_norm.get(answer_norm, set())
                    ),
                    "source_rank_count": int(len(source_rank_positions_by_norm.get(answer_norm, set()))),
                }
            )

        answer_summaries_sorted = sorted(answer_summaries, key=lambda x: float(x["alpha_mean"]), reverse=True)
        hist_out_path: Optional[Path] = None
        if str(args.out_hist).strip():
            hist_out_path = _resolve_record_output_path(
                str(args.out_hist),
                record_id=rec_id,
                top_k=int(args.rankings_top_k),
                force_record_suffix=(len(selected_records) > 1),
            )
            _plot_alpha_histograms_for_record(
                record_id=rec_id,
                answer_summaries=answer_summaries_sorted,
                out_path=hist_out_path,
                bins=int(args.hist_bins),
            )
            hist_written_paths.append(hist_out_path)

        per_record_rows.append(
            {
                "record_id": rec_id,
                "question": cached["question"],
                "gold_answer": cached["gold_answer"],
                "gold_answer_norm": cached["gold_answer_norm"],
                "num_combos": len(combo_specs),
                "combo_specs": combo_specs,
                "num_union_answers": len(answer_summaries_sorted),
                "answers_source_rows": source_row_count,
                "answers_source_files": source_files,
                "answers_source_type": "fine_tuning_baseline_topk",
                "baseline_topk_size": int(args.baseline_topk_size),
                "alpha_hist_path": (str(hist_out_path) if hist_out_path is not None else None),
                "union_answer_alpha_distributions": answer_summaries_sorted,
            }
        )
        print(
            f"[info] Record id={rec_id} done | union_answers={len(answer_summaries_sorted)} "
            f"pairs={len(combo_specs) * len(answer_summaries_sorted)}"
        )

    if not per_record_rows:
        print("[error] No records produced outputs.")
        return 1

    record_out_paths = _write_jsonl_with_record_template(
        str(args.out_record_jsonl),
        per_record_rows,
        top_k=int(args.rankings_top_k),
    )
    pair_out_paths = _write_jsonl_with_record_template(
        str(args.out_pair_jsonl),
        per_pair_rows,
        top_k=int(args.rankings_top_k),
    )

    total_pairs = len(per_pair_rows)
    total_records = len(per_record_rows)
    if len(record_out_paths) == 1:
        print(f"[info] Wrote per-record output: {record_out_paths[0]}")
    else:
        print(f"[info] Wrote {len(record_out_paths)} per-record output files.")
    if len(pair_out_paths) == 1:
        print(f"[info] Wrote per-pair output: {pair_out_paths[0]}")
    else:
        print(f"[info] Wrote {len(pair_out_paths)} per-pair output files.")
    if hist_written_paths:
        if len(hist_written_paths) == 1:
            print(f"[info] Wrote alpha histogram figure: {hist_written_paths[0]}")
        else:
            print(f"[info] Wrote {len(hist_written_paths)} alpha histogram figures.")
    print(f"[summary] records={total_records} optimized_pairs={total_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
