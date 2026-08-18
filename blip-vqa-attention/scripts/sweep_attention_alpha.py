import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.blip_vqa import blip_vqa
from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    _encode_state_with_pair_combo,
    answer_log_prob,
    encode_question_state,
    evaluate_combo_runs_with_consensus,
    path_with_topk,
    prepare_question_inputs,
)
from utils import load_demo_image, soften_mask
from utils2 import (
    data_path,
    ensure_mask,
    gaussian_from_mask,
    guess_focus_words,
    image_root,
    iter_jsonl,
    load_mask_from_dir,
    normalize_answer,
    sanitize_name,
    select_override_indices,
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                yield row


def _build_prompt_lookup(input_path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    try:
        for row in iter_jsonl(input_path):
            rec_id = row.get("id")
            if rec_id is None:
                continue
            lookup[str(rec_id)] = row
    except FileNotFoundError:
        return {}
    return lookup


def _parse_record_ids(record_ids: Optional[Sequence[str]]) -> List[str]:
    if not record_ids:
        return []
    targets: List[str] = []
    for item in record_ids:
        if item is None:
            continue
        parts = [part.strip() for part in str(item).split(",")]
        targets.extend(part for part in parts if part)
    return targets


def _record_matches(row: Dict[str, Any], targets: Sequence[str]) -> bool:
    if not targets:
        return True
    rec_id = row.get("id")
    if rec_id is None:
        return False
    rec_str = str(rec_id)
    return any(rec_str == target for target in targets)


def _seed_for_record(base_seed: int, rec_id: str) -> int:
    seed = int(base_seed)
    for pos, ch in enumerate(rec_id):
        seed = (seed * 131 + (pos + 1) * ord(ch)) & 0xFFFFFFFF
    return seed


def _build_override_rows(
    *,
    tokenizer,
    question_inputs: Dict[str, torch.Tensor],
    question: str,
    prompt: str,
    image_rel: Path,
    rec_id: str,
    device: torch.device,
    mask_cache: Dict[Tuple[str, str], np.ndarray],
) -> Dict[int, torch.Tensor]:
    cache_key = (image_rel.as_posix(), str(prompt or ""))
    if cache_key in mask_cache:
        mask_array = mask_cache[cache_key]
    else:
        mask_array = load_mask_from_dir(MASK_DIR, image_rel.stem, str(prompt or ""), rec_id)
        if mask_array is not None:
            mask_cache[cache_key] = mask_array

    gh = gw = IMAGE_SIZE // PATCH_SIZE
    mask_array = ensure_mask(mask_array, gh, gw, stem=image_rel.stem)
    mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32).view(
        1, 1, mask_array.shape[0], mask_array.shape[1]
    )
    mask_small = F.interpolate(
        mask_tensor,
        size=(gh, gw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
    mask_soft = soften_mask(mask_small, ksize=5, iters=2)
    mask_soft = (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)

    tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
    focus_words = guess_focus_words(question)
    override_indices = select_override_indices(tokens, focus_words, tokenizer)
    if not override_indices:
        override_indices = [0]
    return {idx: mask_soft for idx in override_indices}


def _unique_texts(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        key = normalize_answer(text) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _parse_alpha_values(spec: str) -> List[float]:
    values: List[float] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as err:
            raise ValueError(f"Invalid alpha value: {token!r}") from err
        values.append(value)
    if not values:
        raise ValueError("No alpha values provided.")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Alpha value must be in [0, 1], got {value}.")
    return values


def _build_plot_path(plot_dir: Path, rec_id: str, answer: str) -> Path:
    record_key = sanitize_name(rec_id) or "record"
    answer_key = sanitize_name(answer) or "answer"
    return plot_dir / f"{record_key}__{answer_key}.pdf"


def _plot_answer_sweep(
    *,
    out_path: Path,
    rec_id: str,
    question: str,
    answer: str,
    gold_answer: str,
    baseline_log_prob: float,
    alpha_values: Sequence[float],
    after_means: Sequence[float],
    after_stds: Sequence[float],
    delta_means: Sequence[float],
    delta_stds: Sequence[float],
) -> None:
    x = np.asarray(alpha_values, dtype=np.float64)
    y_after = np.asarray(after_means, dtype=np.float64)
    y_after_std = np.asarray(after_stds, dtype=np.float64)
    y_delta = np.asarray(delta_means, dtype=np.float64)
    y_delta_std = np.asarray(delta_stds, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axes[0].plot(x, y_after, color="#1f77b4", marker="o", linewidth=2.0)
    axes[0].fill_between(x, y_after - y_after_std, y_after + y_after_std, color="#1f77b4", alpha=0.20)
    axes[0].axhline(float(baseline_log_prob), color="#444444", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("After log-prob")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_title(f"id={rec_id} | answer={answer!r}")

    axes[1].plot(x, y_delta, color="#d62728", marker="o", linewidth=2.0)
    axes[1].fill_between(x, y_delta - y_delta_std, y_delta + y_delta_std, color="#d62728", alpha=0.20)
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Delta log-prob")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    question_line = str(question).strip()
    gold_line = str(gold_answer).strip()
    subtitle = f"Q: {question_line}"
    if gold_line:
        subtitle += f"\nGold: {gold_line}"
    fig.suptitle(subtitle, fontsize=10, y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep alpha values over ranked combo overrides, compute per-answer after log-probs "
            "and deltas, and save plots for each answer."
        )
    )
    parser.add_argument(
        "--rankings-jsonl",
        default="out/pair_override_rankings_k{k}.jsonl",
        help="Input ranking JSONL from stage 1 (supports {k}).",
    )
    parser.add_argument(
        "--rankings-top-k",
        type=int,
        default=0,
        help="Top-k token used only to resolve rankings filename template (0 = use --top-k).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/alpha_logprob_sweep_union.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("out/plots"),
        help="Directory to write per-answer PDF plots.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_path,
        help="Path to dataset JSONL (used for fallback image/question/prompt lookup).",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help="Restrict processing to record ids (can be repeated or comma-separated).",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on selected records (0 = all)")
    parser.add_argument("--top-k", type=int, default=3, help="Fallback top-k when ranking row has no metric_config")
    parser.add_argument("--max-length", type=int, default=10, help="Fallback max generation length")
    parser.add_argument("--min-length", type=int, default=1, help="Fallback min generation length")
    parser.add_argument(
        "--alpha-values",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated alpha values to evaluate.",
    )
    parser.add_argument("--union-alpha", type=float, default=1.0, help="Alpha used to build union top-k answers")
    parser.add_argument(
        "--include-gold-if-missing",
        action="store_true",
        help="If set, append gold answer when missing from the union targets.",
    )
    parser.add_argument(
        "--max-union-answers",
        type=int,
        default=0,
        help="Cap number of union answers processed per record (0 = all).",
    )
    parser.add_argument("--combo-count", type=int, default=20, help="How many pair-combinations to build per record")
    parser.add_argument("--combo-top-min", type=int, default=25, help="Min pairs sampled from top-ranked pool")
    parser.add_argument("--combo-top-max", type=int, default=30, help="Max pairs sampled from top-ranked pool")
    parser.add_argument("--combo-next-window", type=int, default=60, help="Take secondary pairs from next-N ranked pool")
    parser.add_argument("--combo-next-min", type=int, default=30, help="Min pairs sampled from next-window pool")
    parser.add_argument("--combo-next-max", type=int, default=40, help="Max pairs sampled from next-window pool")
    parser.add_argument("--combo-seed", type=int, default=0, help="Base seed for per-record combo sampling")
    parser.add_argument("--consensus-weight", type=float, default=1.0, help="Consensus weight (passed through)")
    parser.add_argument("--mv-weight", type=float, default=1.0, help="Mean-variance weight (passed through)")
    parser.add_argument("--mv-var-weight", type=float, default=1.0, help="Variance penalty (passed through)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.max_length < 1 or args.min_length < 1:
        raise ValueError("--max-length and --min-length must be >= 1")

    alpha_values = _parse_alpha_values(args.alpha_values)

    rankings_path_k = int(args.rankings_top_k) if int(args.rankings_top_k) > 0 else int(args.top_k)
    rankings_path = path_with_topk(args.rankings_jsonl, rankings_path_k)
    if not rankings_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")

    ranking_rows = list(_iter_jsonl(rankings_path))
    if not ranking_rows:
        print(f"No ranking rows found in {rankings_path}.")
        return 0

    record_targets = _parse_record_ids(args.record_ids)
    if record_targets:
        ranking_rows = [row for row in ranking_rows if _record_matches(row, record_targets)]
    if args.max_records > 0:
        ranking_rows = ranking_rows[: args.max_records]
    if not ranking_rows:
        print("No ranking rows selected.")
        return 0

    dataset_lookup = _build_prompt_lookup(args.input.expanduser())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    for p in model.parameters():
        p.requires_grad = False

    originals = []
    for layer in model.text_encoder.encoder.layer:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    results: List[Dict[str, Any]] = []
    total_rows = len(ranking_rows)

    for rec_idx, row in enumerate(ranking_rows, 1):
        rec_id = str(row.get("id", ""))
        data_row = dataset_lookup.get(rec_id, {})

        image_value = row.get("image", data_row.get("image"))
        question = row.get("question", data_row.get("question"))
        gold_answer = row.get("gold_answer", data_row.get("answer", ""))
        prompt = str(row.get("prompt", data_row.get("prompt", "")) or "")
        ranked_pairs = row.get("ranked_pairs", [])
        metric_cfg = row.get("metric_config", {})

        if image_value is None or question is None or not isinstance(ranked_pairs, list):
            print(f"[warn] Skipping malformed ranking row for id={rec_id}", file=sys.stderr)
            continue
        if not ranked_pairs:
            print(f"[warn] No ranked_pairs for id={rec_id}; skipping", file=sys.stderr)
            continue

        image_rel = Path(str(image_value))
        image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
        if not image_path.exists():
            print(f"[warn] Missing image: {image_path}; skipping id={rec_id}", file=sys.stderr)
            continue

        rec_top_k = int(metric_cfg.get("top_k", args.top_k))
        rec_max_length = int(metric_cfg.get("max_length", args.max_length))
        rec_min_length = int(metric_cfg.get("min_length", args.min_length))
        gold_answer_text = "" if gold_answer is None else str(gold_answer)
        gold_norm = normalize_answer(gold_answer_text)

        image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
        question_text = str(question)
        question_inputs = prepare_question_inputs(tokenizer, question_text, device)

        override_rows = _build_override_rows(
            tokenizer=tokenizer,
            question_inputs=question_inputs,
            question=question_text,
            prompt=prompt,
            image_rel=image_rel,
            rec_id=rec_id,
            device=device,
            mask_cache=mask_cache,
        )

        with torch.no_grad():
            image_embeds = model.visual_encoder(image)

        baseline_state = encode_question_state(model, question_inputs, image_embeds)

        rec_seed = _seed_for_record(int(args.combo_seed), rec_id)
        combo_results, consensus_payload = evaluate_combo_runs_with_consensus(
            model=model,
            tokenizer=tokenizer,
            question_inputs=question_inputs,
            image_embeds=image_embeds,
            override_rows=override_rows,
            originals=originals,
            ranked_pairs=ranked_pairs,
            gold_norm=gold_norm,
            combo_count=int(args.combo_count),
            combo_top_min=int(args.combo_top_min),
            combo_top_max=int(args.combo_top_max),
            combo_next_window=int(args.combo_next_window),
            combo_next_min=int(args.combo_next_min),
            combo_next_max=int(args.combo_next_max),
            combo_seed=int(rec_seed),
            top_k=int(rec_top_k),
            max_length=int(rec_max_length),
            min_length=int(rec_min_length),
            alpha=float(args.union_alpha),
            consensus_weight=float(args.consensus_weight),
            mv_weight=float(args.mv_weight),
            mv_var_weight=float(args.mv_var_weight),
        )

        combo_pairs_sets: List[List[List[int]]] = []
        for combo in combo_results:
            pairs = combo.get("pairs", [])
            cleaned_pairs: List[List[int]] = []
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        cleaned_pairs.append([int(pair[0]), int(pair[1])])
            if cleaned_pairs:
                combo_pairs_sets.append(cleaned_pairs)

        union_targets = _unique_texts(consensus_payload.get("union_answers", []))
        if args.include_gold_if_missing and gold_answer_text.strip():
            union_norms = {normalize_answer(a) for a in union_targets}
            if gold_norm and gold_norm not in union_norms:
                union_targets.append(gold_answer_text.strip())

        if args.max_union_answers > 0:
            union_targets = union_targets[: int(args.max_union_answers)]

        if not combo_pairs_sets:
            print(f"[warn] No combo pair sets for id={rec_id}; skipping", file=sys.stderr)
            continue
        if not union_targets:
            print(f"[warn] No union targets for id={rec_id}; skipping", file=sys.stderr)
            continue

        before_log_probs: Dict[str, float] = {}
        for answer in union_targets:
            before_log_probs[str(answer)] = float(
                answer_log_prob(
                    model,
                    tokenizer,
                    baseline_state,
                    question_inputs["attention_mask"],
                    str(answer),
                )
            )

        if not args.quiet:
            print("========================================")
            print(f"id: {rec_id}")
            print(f"record: {rec_idx}/{total_rows}")
            print(f"union targets: {len(union_targets)}")
            print(f"combo sets: {len(combo_pairs_sets)}")
            print(f"alpha values: {len(alpha_values)}")

        answer_summaries: Dict[str, Dict[str, Any]] = {}
        for answer in union_targets:
            answer_summaries[str(answer)] = {
                "answer": str(answer),
                "before_log_prob": float(before_log_probs[str(answer)]),
                "alphas": [],
            }

        for alpha in alpha_values:
            per_answer_after: Dict[str, List[float]] = {str(answer): [] for answer in union_targets}

            for combo_pairs in combo_pairs_sets:
                state = _encode_state_with_pair_combo(
                    model=model,
                    question_inputs=question_inputs,
                    image_embeds=image_embeds,
                    combo_pairs=[(int(pair[0]), int(pair[1])) for pair in combo_pairs],
                    override_rows=override_rows,
                    originals=originals,
                    alpha=float(alpha),
                )
                for answer in union_targets:
                    after_lp = float(
                        answer_log_prob(
                            model,
                            tokenizer,
                            state,
                            question_inputs["attention_mask"],
                            str(answer),
                        )
                    )
                    per_answer_after[str(answer)].append(after_lp)

            for answer in union_targets:
                answer_key = str(answer)
                after_values = per_answer_after[answer_key]
                before_lp = float(before_log_probs[answer_key])
                delta_values = [float(v) - before_lp for v in after_values]
                after_arr = np.array(after_values, dtype=np.float64) if after_values else np.array([], dtype=np.float64)
                delta_arr = np.array(delta_values, dtype=np.float64) if delta_values else np.array([], dtype=np.float64)
                answer_summaries[answer_key]["alphas"].append(
                    {
                        "alpha": float(alpha),
                        "after_logprob_values": [float(v) for v in after_values],
                        "after_logprob_mean": float(after_arr.mean()) if after_arr.size else float("nan"),
                        "after_logprob_std": float(after_arr.std()) if after_arr.size else float("nan"),
                        "delta_logprob_values": [float(v) for v in delta_values],
                        "delta_logprob_mean": float(delta_arr.mean()) if delta_arr.size else float("nan"),
                        "delta_logprob_std": float(delta_arr.std()) if delta_arr.size else float("nan"),
                    }
                )

            if not args.quiet:
                print(f"alpha={float(alpha):.6f}")
                for answer in union_targets:
                    stats = answer_summaries[str(answer)]["alphas"][-1]
                    print(
                        f"  answer={str(answer)!r} "
                        f"after_mean={float(stats['after_logprob_mean']):+.6f} "
                        f"delta_mean={float(stats['delta_logprob_mean']):+.6f}"
                    )

        per_answer_rows: List[Dict[str, Any]] = []
        for answer in union_targets:
            answer_key = str(answer)
            summary = answer_summaries[answer_key]
            alpha_rows = summary["alphas"]
            plot_path = _build_plot_path(args.plot_dir, rec_id, answer_key)
            _plot_answer_sweep(
                out_path=plot_path,
                rec_id=rec_id,
                question=question_text,
                answer=answer_key,
                gold_answer=gold_answer_text,
                baseline_log_prob=float(summary["before_log_prob"]),
                alpha_values=[float(item["alpha"]) for item in alpha_rows],
                after_means=[float(item["after_logprob_mean"]) for item in alpha_rows],
                after_stds=[float(item["after_logprob_std"]) for item in alpha_rows],
                delta_means=[float(item["delta_logprob_mean"]) for item in alpha_rows],
                delta_stds=[float(item["delta_logprob_std"]) for item in alpha_rows],
            )

            per_answer_rows.append(
                {
                    "answer": answer_key,
                    "before_log_prob": float(summary["before_log_prob"]),
                    "alpha_runs": alpha_rows,
                    "plot_path": str(plot_path),
                }
            )

        result_row = {
            "index": int(rec_idx),
            "id": rec_id,
            "image": str(image_value),
            "prompt": prompt,
            "question": question_text,
            "answer": gold_answer_text,
            "union_alpha": float(args.union_alpha),
            "union_topk_answers": union_targets,
            "selected_answer": str(consensus_payload.get("selected_answer", "")),
            "combo_count": int(len(combo_pairs_sets)),
            "combo_seed": int(rec_seed),
            "combo_pairs_sets": combo_pairs_sets,
            "rankings_source": str(rankings_path),
            "metric_config": {
                "top_k": int(rec_top_k),
                "max_length": int(rec_max_length),
                "min_length": int(rec_min_length),
            },
            "combo_config": {
                "combo_count": int(args.combo_count),
                "top_pick_min": int(args.combo_top_min),
                "top_pick_max": int(args.combo_top_max),
                "next_window": int(args.combo_next_window),
                "next_pick_min": int(args.combo_next_min),
                "next_pick_max": int(args.combo_next_max),
                "seed": int(args.combo_seed),
            },
            "score_weights": {
                "consensus_w": float(args.consensus_weight),
                "mv_w": float(args.mv_weight),
                "mv_var_w": float(args.mv_var_weight),
            },
            "alpha_values": [float(v) for v in alpha_values],
            "answers": per_answer_rows,
        }
        results.append(result_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} rows to {args.output}")
    print(f"Plots written under {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
