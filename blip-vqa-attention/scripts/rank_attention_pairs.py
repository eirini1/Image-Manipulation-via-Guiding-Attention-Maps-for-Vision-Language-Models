''' compute per-record (layer,head) metrics and save ranked pairs in descending score order. '''
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    answer_log_prob,
    build_ranked_pairs_from_matrices,
    encode_question_state,
    entropy_from_probs,
    generate_topk_answers,
    kl_divergence_from_probs,
    load_existing_cache,
    margin_from_probs,
    margin_from_scores,
    path_with_topk,
    prepare_question_inputs,
    print_top_pairs,
    probs_from_log_probs,
    resolve_heads,
    resolve_layers,
    union_answers,
)
from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import (
    apply_override,
    data_path,
    ensure_mask,
    gaussian_from_mask,
    guess_focus_words,
    image_root,
    iter_jsonl,
    load_mask_from_dir,
    normalize_answer,
    revert_override,
    select_override_indices,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: compute per-record (layer,head) metrics and save ranked pairs in descending score order."
    )
    parser.add_argument("--layers", default="all", help="Layer pool: 'all' or comma-separated indices")
    parser.add_argument("--heads", default="all", help="Head pool: 'all' or comma-separated indices")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k answers used before/after for union scoring")
    parser.add_argument("--max-length", type=int, default=10, help="Max generation length for top-k answers")
    parser.add_argument("--min-length", type=int, default=1, help="Min generation length for top-k answers")
    parser.add_argument(
        "--out-matrices-jsonl",
        default="out/pair_override_matrices_k{k}.jsonl",
        help="Output JSONL path template for per-record matrices cache (supports {k})",
    )
    parser.add_argument(
        "--out-rankings-jsonl",
        default="out/pair_override_rankings_k{k}.jsonl",
        help="Output JSONL path template for per-record ranked pairs (supports {k})",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on processed records (0 = all)")
    parser.add_argument("--b", type=float, default=1.0, help="Weight for delta_margin in ranking: b*delta_m + c*delta_H + d*delta_KL")
    parser.add_argument("--c", type=float, default=1.0, help="Weight for delta_entropy in ranking: b*delta_m + c*delta_H + d*delta_KL")
    parser.add_argument("--d", type=float, default=1.0, help="Weight for KL divergence in ranking: b*delta_m + c*delta_H + d*delta_KL")
    parser.add_argument("--print-top-n", type=int, default=70, help="Print top-N ranked (layer,head) pairs per record")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intervention strength")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = list(iter_jsonl(data_path))
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        print("No records to evaluate.")
        return 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    total_heads = int(getattr(encoder_layers[0].crossattention.self, "num_attention_heads", 12))
    target_layers = resolve_layers(args.layers, total_layers)
    target_heads = resolve_heads(args.heads, total_heads)
    pairs = [(layer_idx, head_idx) for layer_idx in target_layers for head_idx in target_heads]
    if not pairs:
        print("No (layer,head) pairs selected.")
        return 0
    layer_to_pos = {int(layer_idx): pos for pos, layer_idx in enumerate(target_layers)}
    head_to_pos = {int(head_idx): pos for pos, head_idx in enumerate(target_heads)}

    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    out_matrices_path = path_with_topk(args.out_matrices_jsonl, args.top_k)
    out_rankings_path = path_with_topk(args.out_rankings_jsonl, args.top_k)
    out_matrices_path.parent.mkdir(parents=True, exist_ok=True)
    out_rankings_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cache = load_existing_cache(
        out_matrices_path,
        target_layers=target_layers,
        target_heads=target_heads,
        top_k=args.top_k,
        max_length=args.max_length,
        min_length=args.min_length,
        require_kl_matrix=True,
    )
    if existing_cache:
        print(f"[info] Found {len(existing_cache)} cacheable matrix records in {out_matrices_path}")

    pair_stats: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_delta_h": 0.0,
            "sum_delta_m": 0.0,
            "sum_delta_kl": 0.0,
            "sum_delta_m_prob": 0.0,
            "sum_h_before": 0.0,
            "sum_h_after": 0.0,
            "before_hits": 0.0,
            "after_hits": 0.0,
        }
    )
    total_rows = 0
    reused_records = 0
    computed_records = 0
    start_time = time.time()
    prog_step = 5

    with out_matrices_path.open("w", encoding="utf-8") as matrices_file, out_rankings_path.open(
        "w", encoding="utf-8"
    ) as rankings_file:
        for rec_idx, entry in enumerate(records, 1):
            image_value = entry.get("image")
            question = entry.get("question")
            gold = entry.get("answer")
            prompt = entry.get("prompt")
            if image_value is None or question is None or gold is None:
                print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
                continue

            image_rel = Path(image_value)
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
                continue

            stem = image_rel.stem
            rec_id = entry.get("id")
            rec_id_str = str(rec_id) if rec_id is not None else stem
            gold_norm = normalize_answer(str(gold))

            cached_row = existing_cache.get(rec_id_str)
            if cached_row is not None:
                delta_entropy_matrix = cached_row.get("delta_entropy_matrix")
                delta_margin_matrix = cached_row.get("delta_margin_matrix")
                delta_kl_matrix = cached_row.get("delta_kl_matrix")
                if (
                    isinstance(delta_entropy_matrix, list)
                    and isinstance(delta_margin_matrix, list)
                    and isinstance(delta_kl_matrix, list)
                ):
                    pair_matrix, ranked_pairs, valid_cells = build_ranked_pairs_from_matrices(
                        delta_entropy_matrix,
                        delta_margin_matrix,
                        delta_kl_matrix=delta_kl_matrix,
                        layers=target_layers,
                        heads=target_heads,
                        b=args.b,
                        c=args.c,
                        d=args.d,
                    )
                    if valid_cells > 0:
                        total_rows += int(valid_cells)
                        matrix_row: Dict[str, Any] = {
                            "id": rec_id_str,
                            "image": str(image_value),
                            "question": str(question),
                            "prompt": str(prompt or ""),
                            "gold_answer": str(gold),
                            "metric_config": {
                                "top_k": int(args.top_k),
                                "max_length": int(args.max_length),
                                "min_length": int(args.min_length),
                            },
                            "layers": [int(v) for v in target_layers],
                            "heads": [int(v) for v in target_heads],
                            "pred_before": str(cached_row.get("pred_before", "")),
                            "before_hit": int(cached_row.get("before_hit", 0)),
                            "delta_entropy_matrix": delta_entropy_matrix,
                            "delta_margin_matrix": delta_margin_matrix,
                            "delta_kl_matrix": delta_kl_matrix,
                        }
                        ranking_row: Dict[str, Any] = {
                            "id": rec_id_str,
                            "image": str(image_value),
                            "question": str(question),
                            "prompt": str(prompt or ""),
                            "gold_answer": str(gold),
                            "rank_formula": "b*delta_m + c*delta_H + d*delta_KL",
                            "rank_weights": {"b": float(args.b), "c": float(args.c), "d": float(args.d)},
                            "metric_config": matrix_row["metric_config"],
                            "layers": matrix_row["layers"],
                            "heads": matrix_row["heads"],
                            "pred_before": matrix_row["pred_before"],
                            "before_hit": matrix_row["before_hit"],
                            "pair_matrix": pair_matrix,
                            "ranked_pairs": ranked_pairs,
                            "top_pairs": ranked_pairs[: max(0, int(args.print_top_n))],
                            "matrix_source": "cache",
                        }
                        matrices_file.write(json.dumps(matrix_row, ensure_ascii=False) + "\n")
                        rankings_file.write(json.dumps(ranking_row, ensure_ascii=False) + "\n")
                        print_top_pairs(rec_id_str, ranked_pairs, int(args.print_top_n))
                        reused_records += 1
                        if (rec_idx == 1) or (rec_idx % prog_step == 0):
                            elapsed = time.time() - start_time
                            rate = rec_idx / elapsed if elapsed > 0 else 0.0
                            print(
                                f"[progress] {rec_idx}/{len(records)} records | rows: {total_rows} | reused: {reused_records} | {rate:.2f} rec/s",
                                flush=True,
                            )
                        continue

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
            question_inputs = prepare_question_inputs(tokenizer, str(question), device)

            cache_key = (image_rel.as_posix(), str(prompt or ""))
            if cache_key in mask_cache:
                mask_array = mask_cache[cache_key]
            else:
                mask_array = load_mask_from_dir(MASK_DIR, stem, str(prompt or ""), rec_id_str)
                if mask_array is not None:
                    mask_cache[cache_key] = mask_array

            gh = gw = IMAGE_SIZE // PATCH_SIZE
            mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
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

            with torch.no_grad():
                image_embeds = model.visual_encoder(image)

            baseline_state = encode_question_state(model, question_inputs, image_embeds)
            topk_before = generate_topk_answers(
                model,
                tokenizer,
                baseline_state,
                top_k=args.top_k,
                max_length=args.max_length,
                min_length=args.min_length,
            )
            pred_before = topk_before[0] if topk_before else ""
            pred_before_norm = normalize_answer(pred_before)
            before_hit = int(pred_before_norm == gold_norm)

            baseline_logprob_cache: Dict[str, float] = {}
            delta_entropy_matrix: List[List[Optional[float]]] = [[None for _ in target_heads] for _ in target_layers]
            delta_margin_matrix: List[List[Optional[float]]] = [[None for _ in target_heads] for _ in target_layers]
            delta_kl_matrix: List[List[Optional[float]]] = [[None for _ in target_heads] for _ in target_layers]

            for layer_idx, head_idx in pairs:
                new_forward = make_forward(head_idx, override_rows, args.alpha)
                apply_override(new_forward, (layer_idx,), originals)
                try:
                    after_state = encode_question_state(model, question_inputs, image_embeds)
                    topk_after = generate_topk_answers(
                        model,
                        tokenizer,
                        after_state,
                        top_k=args.top_k,
                        max_length=args.max_length,
                        min_length=args.min_length,
                    )
                finally:
                    revert_override((layer_idx,), originals)

                merged_answers = union_answers(topk_before, topk_after)
                if not merged_answers:
                    continue

                logp_before: List[float] = []
                logp_after: List[float] = []
                for answer in merged_answers:
                    if answer not in baseline_logprob_cache:
                        baseline_logprob_cache[answer] = answer_log_prob(
                            model,
                            tokenizer,
                            baseline_state,
                            question_inputs["attention_mask"],
                            answer,
                        )
                    logp_before.append(baseline_logprob_cache[answer])
                    logp_after.append(
                        answer_log_prob(
                            model,
                            tokenizer,
                            after_state,
                            question_inputs["attention_mask"],
                            answer,
                        )
                    )

                probs_before = probs_from_log_probs(logp_before)
                probs_after = probs_from_log_probs(logp_after)
                entropy_before = entropy_from_probs(probs_before)
                entropy_after = entropy_from_probs(probs_after)
                delta_h = entropy_before - entropy_after
                delta_kl = kl_divergence_from_probs(probs_before, probs_after)

                margin_before = margin_from_scores(logp_before)
                margin_after = margin_from_scores(logp_after)
                delta_m = margin_after - margin_before
                margin_before_prob = margin_from_probs(probs_before)
                margin_after_prob = margin_from_probs(probs_after)
                delta_m_prob = margin_after_prob - margin_before_prob

                pred_after = topk_after[0] if topk_after else ""
                pred_after_norm = normalize_answer(pred_after)
                after_hit = int(pred_after_norm == gold_norm)

                layer_pos = layer_to_pos[int(layer_idx)]
                head_pos = head_to_pos[int(head_idx)]
                delta_entropy_matrix[layer_pos][head_pos] = float(delta_h)
                delta_margin_matrix[layer_pos][head_pos] = float(delta_m)
                delta_kl_matrix[layer_pos][head_pos] = float(delta_kl)
                total_rows += 1

                stats = pair_stats[(layer_idx, head_idx)]
                stats["count"] += 1.0
                stats["sum_delta_h"] += float(delta_h)
                stats["sum_delta_m"] += float(delta_m)
                stats["sum_delta_kl"] += float(delta_kl)
                stats["sum_delta_m_prob"] += float(delta_m_prob)
                stats["sum_h_before"] += float(entropy_before)
                stats["sum_h_after"] += float(entropy_after)
                stats["before_hits"] += float(before_hit)
                stats["after_hits"] += float(after_hit)

            pair_matrix, ranked_pairs, _ = build_ranked_pairs_from_matrices(
                delta_entropy_matrix,
                delta_margin_matrix,
                delta_kl_matrix=delta_kl_matrix,
                layers=target_layers,
                heads=target_heads,
                b=args.b,
                c=args.c,
                d=args.d,
            )
            matrix_row: Dict[str, Any] = {
                "id": rec_id_str,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "gold_answer": str(gold),
                "metric_config": {
                    "top_k": int(args.top_k),
                    "max_length": int(args.max_length),
                    "min_length": int(args.min_length),
                },
                "layers": [int(v) for v in target_layers],
                "heads": [int(v) for v in target_heads],
                "pred_before": pred_before,
                "before_hit": int(before_hit),
                "delta_entropy_matrix": delta_entropy_matrix,
                "delta_margin_matrix": delta_margin_matrix,
                "delta_kl_matrix": delta_kl_matrix,
            }
            ranking_row: Dict[str, Any] = {
                "id": rec_id_str,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "gold_answer": str(gold),
                "rank_formula": "b*delta_m + c*delta_H + d*delta_KL",
                "rank_weights": {"b": float(args.b), "c": float(args.c), "d": float(args.d)},
                "metric_config": matrix_row["metric_config"],
                "layers": matrix_row["layers"],
                "heads": matrix_row["heads"],
                "pred_before": matrix_row["pred_before"],
                "before_hit": matrix_row["before_hit"],
                "pair_matrix": pair_matrix,
                "ranked_pairs": ranked_pairs,
                "top_pairs": ranked_pairs[: max(0, int(args.print_top_n))],
                "matrix_source": "computed",
            }
            matrices_file.write(json.dumps(matrix_row, ensure_ascii=False) + "\n")
            rankings_file.write(json.dumps(ranking_row, ensure_ascii=False) + "\n")
            print_top_pairs(rec_id_str, ranked_pairs, int(args.print_top_n))
            computed_records += 1

            if (rec_idx == 1) or (rec_idx % prog_step == 0):
                elapsed = time.time() - start_time
                rate = rec_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] {rec_idx}/{len(records)} records | rows: {total_rows} | reused: {reused_records} | {rate:.2f} rec/s",
                    flush=True,
                )

    print("========================================")
    print(f"Processed records: {len(records)}")
    print(f"Selected layer pool: {list(target_layers)}")
    print(f"Selected head pool: {list(target_heads)}")
    print(f"Total (layer,head) pairs: {len(pairs)}")
    print(f"Reused records: {reused_records} | Computed records: {computed_records}")
    print(f"Wrote matrices cache: {out_matrices_path}")
    print(f"Wrote rankings: {out_rankings_path}")
    print("")
    print("Pair summary (mean values):")
    if reused_records:
        print("[info] Pair summary below is from newly computed records only.")
    for layer_idx, head_idx in sorted(pair_stats.keys()):
        stats = pair_stats[(layer_idx, head_idx)]
        count = max(stats["count"], 1.0)
        mean_delta_h = stats["sum_delta_h"] / count
        mean_delta_m = stats["sum_delta_m"] / count
        mean_delta_kl = stats["sum_delta_kl"] / count
        mean_h_before = stats["sum_h_before"] / count
        mean_h_after = stats["sum_h_after"] / count
        before_acc = 100.0 * stats["before_hits"] / count
        after_acc = 100.0 * stats["after_hits"] / count
        print(
            f"(L{layer_idx},H{head_idx}) n={int(stats['count'])} "
            f"| mean dH={mean_delta_h:+.6f} (H_before={mean_h_before:.6f}, H_after={mean_h_after:.6f}) "
            f"| mean dm={mean_delta_m:+.6f} "
            f"| mean dKL={mean_delta_kl:+.6f} "
            f"| acc_before={before_acc:.2f}% acc_after={after_acc:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

