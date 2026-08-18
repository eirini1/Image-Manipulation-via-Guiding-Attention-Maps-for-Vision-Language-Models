''' read ranked pairs, run combo overrides, pick possible answers, and write accuracy summary. '''
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    evaluate_combo_runs_with_consensus,
    path_with_topk,
    prepare_question_inputs,
)
from models.blip_vqa import blip_vqa
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


def _combo_hit(combo: Dict[str, Any], gold_norm: str) -> int:
    after_hit = combo.get("after_hit")
    if after_hit is not None:
        try:
            return int(bool(int(after_hit)))
        except Exception:
            pass
    pred_after = str(combo.get("pred_after", ""))
    return int(normalize_answer(pred_after) == gold_norm)


def _build_prompt_lookup() -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    try:
        for row in iter_jsonl(data_path):
            rec_id = row.get("id")
            if rec_id is None:
                continue
            lookup[str(rec_id)] = row
    except FileNotFoundError:
        return {}
    return lookup


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: read ranked pairs, run combo overrides, pick possible answers, and write accuracy summary."
    )
    parser.add_argument(
        "--rankings-jsonl",
        default="out/pair_override_rankings_k{k}.jsonl",
        help="Input ranking JSONL from stage 1 (supports {k}).",
    )
    parser.add_argument(
        "--out-combos-jsonl",
        default="out/pair_override_combos_k{k}.jsonl",
        help="Output JSONL with combo override results (supports {k}).",
    )
    parser.add_argument(
        "--out-summary-jsonl",
        default="out/combo_accuracy_summary_k{k}.jsonl",
        help="Output JSONL with per-record combo accuracy (supports {k}).",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on processed records (0 = all)")
    parser.add_argument("--top-k", type=int, default=3, help="Fallback top-k when ranking row has no metric_config")
    parser.add_argument(
        "--rankings-top-k",
        type=int,
        default=0,
        help="Top-k token used only to resolve the rankings filename template (0 = use --top-k).",
    )
    parser.add_argument("--max-length", type=int, default=10, help="Fallback max generation length")
    parser.add_argument("--min-length", type=int, default=1, help="Fallback min generation length")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intervention strength")
    parser.add_argument("--combo-count", type=int, default=20, help="How many pair-combinations to build per record")
    parser.add_argument("--combo-top-min", type=int, default=25, help="Min pairs sampled from top-ranked pool")
    parser.add_argument("--combo-top-max", type=int, default=30, help="Max pairs sampled from top-ranked pool")
    parser.add_argument("--combo-next-window", type=int, default=60, help="Take secondary pairs from next-N ranked pool")
    parser.add_argument("--combo-next-min", type=int, default=30, help="Min pairs sampled from next-window pool")
    parser.add_argument("--combo-next-max", type=int, default=40, help="Max pairs sampled from next-window pool")
    parser.add_argument("--combo-seed", type=int, default=0, help="Base seed for per-record combo sampling")
    parser.add_argument("--consensus-weight", type=float, default=1.0, help="Weight for consensus frequency")
    parser.add_argument("--mv-weight", type=float, default=1.0, help="Weight for mean-variance score")
    parser.add_argument("--mv-var-weight", type=float, default=1.0, help="Variance penalty inside mean-variance score")
    args = parser.parse_args(argv)

    rankings_path_k = int(args.rankings_top_k) if int(args.rankings_top_k) > 0 else int(args.top_k)
    rankings_path = path_with_topk(args.rankings_jsonl, rankings_path_k)
    if not rankings_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")

    out_combos_path = path_with_topk(args.out_combos_jsonl, args.top_k)
    out_summary_path = path_with_topk(args.out_summary_jsonl, args.top_k)
    out_combos_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    ranking_rows = list(_iter_jsonl(rankings_path))
    if args.max_records > 0:
        ranking_rows = ranking_rows[: args.max_records]
    if not ranking_rows:
        print(f"No ranking rows found in {rankings_path}.")
        return 0

    dataset_lookup = _build_prompt_lookup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    originals = []
    for layer in model.text_encoder.encoder.layer:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    per_record: List[Dict[str, Any]] = []
    per_record_answer_tracks: List[Dict[str, Any]] = []
    total_records = 0
    total_combo_trials = 0
    total_combo_hits = 0
    total_possible_hits = 0
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    start_time = time.time()
    prog_step = 5

    with out_combos_path.open("w", encoding="utf-8") as combos_file:
        for rec_idx, row in enumerate(ranking_rows, 1):
            rec_id = str(row.get("id", ""))
            data_row = dataset_lookup.get(rec_id, {})

            image_value = row.get("image", data_row.get("image"))
            question = row.get("question", data_row.get("question"))
            gold_answer = row.get("gold_answer", data_row.get("answer", ""))
            prompt = row.get("prompt", data_row.get("prompt", ""))
            ranked_pairs = row.get("ranked_pairs", [])
            before_hit = int(row.get("before_hit", 0))
            pred_before = str(row.get("pred_before", ""))
            metric_cfg = row.get("metric_config", {})

            if image_value is None or question is None or not isinstance(ranked_pairs, list):
                print(f"[warn] Skipping malformed ranking row for id={rec_id}", file=sys.stderr)
                continue

            image_rel = Path(str(image_value))
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}; skipping id={rec_id}", file=sys.stderr)
                continue

            rec_top_k = int(metric_cfg.get("top_k", args.top_k))
            rec_max_length = int(metric_cfg.get("max_length", args.max_length))
            rec_min_length = int(metric_cfg.get("min_length", args.min_length))
            gold_norm = normalize_answer(str(gold_answer))

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
            question_inputs = prepare_question_inputs(tokenizer, str(question), device)
            with torch.no_grad():
                image_embeds = model.visual_encoder(image)

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
            mask_small = torch.nn.functional.interpolate(
                mask_tensor,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
            mask_soft = soften_mask(mask_small, ksize=5, iters=2)
            mask_soft = (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)

            tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
            focus_words = guess_focus_words(str(question))
            override_indices = select_override_indices(tokens, focus_words, tokenizer)
            if not override_indices:
                override_indices = [0]
            override_rows = {i: mask_soft for i in override_indices}

            rec_seed = int(args.combo_seed)
            for pos, ch in enumerate(rec_id):
                rec_seed = (rec_seed * 131 + (pos + 1) * ord(ch)) & 0xFFFFFFFF

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
                alpha=float(args.alpha),
                consensus_weight=float(args.consensus_weight),
                mv_weight=float(args.mv_weight),
                mv_var_weight=float(args.mv_var_weight),
            )

            combos_row = {
                "id": rec_id,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "gold_answer": str(gold_answer),
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
                    "alpha": float(args.alpha),
                },
                "consensus_config": {
                    "consensus_weight": float(args.consensus_weight),
                    "mv_weight": float(args.mv_weight),
                    "mv_var_weight": float(args.mv_var_weight),
                },
                "pred_before": pred_before,
                "before_hit": int(before_hit),
                "combo_results": combo_results,
                "union_topk_answers": consensus_payload.get("union_answers", []),
                "teacher_forced_consensus": consensus_payload,
                "possible_answer": str(consensus_payload.get("selected_answer", "")),
                "possible_answer_hit": int(consensus_payload.get("selected_hit", 0)),
                "ranking_source": str(rankings_path),
            }
            combos_file.write(json.dumps(combos_row, ensure_ascii=False) + "\n")

            hits = sum(_combo_hit(combo, gold_norm) for combo in combo_results if isinstance(combo, dict))
            total = len(combo_results)
            acc = (100.0 * hits / total) if total > 0 else 0.0
            any_correct = int(hits > 0)
            possible_answer = str(consensus_payload.get("selected_answer", ""))
            possible_hit = int(normalize_answer(possible_answer) == gold_norm)

            per_record.append(
                {
                    "id": rec_id,
                    "gold_answer": str(gold_answer),
                    "possible_answer": possible_answer,
                    "possible_hit": possible_hit,
                    "before_hit": int(before_hit),
                    "combo_hits": int(hits),
                    "combo_total": int(total),
                    "combo_accuracy_percent": float(acc),
                    "any_combo_correct": int(any_correct),
                }
            )
            per_record_answer_tracks.append(
                {
                    "id": rec_id,
                    "possible_answer": possible_answer,
                    "answer_stats": [
                        {
                            "answer": str(stat.get("answer", "")),
                            "consensus_freq": float(stat.get("consensus_freq", 0.0)),
                            "mean_prob": float(stat.get("mean_prob", 0.0)),
                            "var_prob": float(stat.get("var_prob", 0.0)),
                            "combined_score": float(stat.get("combined_score", 0.0)),
                        }
                        for stat in consensus_payload.get("answer_stats", [])
                        if isinstance(stat, dict)
                    ],
                }
            )
            total_records += 1
            total_combo_trials += int(total)
            total_combo_hits += int(hits)
            total_possible_hits += int(possible_hit)

            if (rec_idx == 1) or (rec_idx % prog_step == 0):
                elapsed = time.time() - start_time
                rate = rec_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] {rec_idx}/{len(ranking_rows)} ranking rows | combos: {total_combo_trials} | {rate:.2f} rec/s",
                    flush=True,
                )

    with out_summary_path.open("w", encoding="utf-8") as f:
        for item in per_record:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    overall_combo_acc = 100.0 * total_combo_hits / total_combo_trials if total_combo_trials > 0 else 0.0
    mean_record_acc = (
        sum(float(x["combo_accuracy_percent"]) for x in per_record) / len(per_record) if per_record else 0.0
    )
    any_correct_records = sum(int(x["any_combo_correct"]) for x in per_record)

    print("========================================")
    print(f"Rankings file: {rankings_path}")
    print(f"Wrote combos: {out_combos_path}")
    print(f"Wrote summary: {out_summary_path}")
    print(f"Records: {total_records}")
    print(f"Overall combo accuracy (micro): {total_combo_hits}/{total_combo_trials} ({overall_combo_acc:.2f}%)")
    print(f"Mean per-record combo accuracy (macro): {mean_record_acc:.2f}%")
    print(f"Records with >=1 correct combo: {any_correct_records}/{total_records}")
    print(
        f"Possible-answer accuracy: {total_possible_hits}/{total_records} "
        f"({(100.0 * total_possible_hits / total_records) if total_records else 0.0:.2f}%)"
    )
    print("")
    print("Per-record combo accuracy:")
    tracks_by_id = {str(item["id"]): item for item in per_record_answer_tracks}
    for item in per_record:
        print(
            f"id={item['id']} | combos={item['combo_hits']}/{item['combo_total']} "
            f"({item['combo_accuracy_percent']:.2f}%) | before_hit={item['before_hit']} "
            f"| possible_answer={item['possible_answer']} | gold={item['gold_answer']}"
        )
        tracks = tracks_by_id.get(str(item["id"]), {})
        answer_stats = tracks.get("answer_stats", [])
        if answer_stats:
            print("  answer_tracks:")
            for stat in answer_stats:
                print(
                    f"    answer={stat['answer']!r} | consensus_freq={stat['consensus_freq']:.6f} "
                    f"| mean_prob={stat['mean_prob']:.6f} | var_prob={stat['var_prob']:.6f} "
                    f"| combined_score={stat['combined_score']:.6f}"
                )
        else:
            print("  answer_tracks: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

