import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    answer_log_prob,
    encode_question_state,
    generate_topk_answers,
    kl_divergence_from_probs,
    path_with_topk,
    prepare_question_inputs,
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


def _load_existing_image_dependence_cache(
    path: Path,
    *,
    target_layers: Sequence[int],
    target_heads: Sequence[int],
    top_k: int,
    max_length: int,
    min_length: int,
    blur_kernel_size: int,
    blur_sigma: float,
) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    cache: Dict[str, Dict[str, Any]] = {}
    wanted_layers = [int(v) for v in target_layers]
    wanted_heads = [int(v) for v in target_heads]
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_id = row.get("id")
            if rec_id is None:
                continue
            if row.get("layers") != wanted_layers or row.get("heads") != wanted_heads:
                continue
            if (
                "image_dependence_matrix" not in row
                or "kl_before_matrix" not in row
                or "kl_after_matrix" not in row
            ):
                continue
            cfg = row.get("metric_config")
            if not isinstance(cfg, dict):
                continue
            if int(cfg.get("top_k", -1)) != int(top_k):
                continue
            if int(cfg.get("max_length", -1)) != int(max_length):
                continue
            if int(cfg.get("min_length", -1)) != int(min_length):
                continue
            if int(cfg.get("blur_kernel_size", -1)) != int(blur_kernel_size):
                continue
            if not np.isclose(float(cfg.get("blur_sigma", -1.0)), float(blur_sigma), atol=1e-8):
                continue
            cache[str(rec_id)] = row
    return cache


def _build_ranked_pairs_from_image_dependence(
    image_dependence_matrix: Sequence[Sequence[Optional[float]]],
    kl_before_matrix: Sequence[Sequence[Optional[float]]],
    kl_after_matrix: Sequence[Sequence[Optional[float]]],
    *,
    layers: Sequence[int],
    heads: Sequence[int],
) -> Tuple[List[List[Optional[Dict[str, float]]]], List[Dict[str, float]], int]:
    pair_matrix: List[List[Optional[Dict[str, float]]]] = []
    ranked_pairs: List[Dict[str, float]] = []
    valid_cells = 0
    for i, layer_idx in enumerate(layers):
        row_cells: List[Optional[Dict[str, float]]] = []
        for j, head_idx in enumerate(heads):
            score = None
            kl_before = None
            kl_after = None
            if i < len(image_dependence_matrix) and j < len(image_dependence_matrix[i]):
                score = image_dependence_matrix[i][j]
            if i < len(kl_before_matrix) and j < len(kl_before_matrix[i]):
                kl_before = kl_before_matrix[i][j]
            if i < len(kl_after_matrix) and j < len(kl_after_matrix[i]):
                kl_after = kl_after_matrix[i][j]
            if score is None or kl_before is None or kl_after is None:
                row_cells.append(None)
                continue

            score_f = float(score)
            kl_before_f = float(kl_before)
            kl_after_f = float(kl_after)
            cell = {
                "score": score_f,
                "kl_before": kl_before_f,
                "kl_after": kl_after_f,
            }
            row_cells.append(cell)
            ranked_pairs.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "score": score_f,
                    "kl_before": kl_before_f,
                    "kl_after": kl_after_f,
                }
            )
            valid_cells += 1
        pair_matrix.append(row_cells)
    ranked_pairs.sort(
        key=lambda item: (item["score"], item["kl_after"], -item["kl_before"]),
        reverse=True,
    )
    return pair_matrix, ranked_pairs, valid_cells


def _print_top_pairs(record_id: str, ranked_pairs: Sequence[Dict[str, float]], top_n: int) -> None:
    if top_n <= 0:
        return
    limit = min(int(top_n), len(ranked_pairs))
    print(f"[top_pairs] id={record_id} showing {limit} of {len(ranked_pairs)}")
    for rank, item in enumerate(ranked_pairs[:limit], 1):
        print(
            f"  {rank:02d}. (L{int(item['layer'])},H{int(item['head'])}) "
            f"score={float(item['score']):+.6f} "
            f"KL_before={float(item['kl_before']):+.6f} "
            f"KL_after={float(item['kl_after']):+.6f}"
        )


def _make_corrupted_image(
    image: torch.Tensor,
    mask_soft: torch.Tensor,
    *,
    blur_kernel_size: int,
    blur_sigma: float,
) -> torch.Tensor:
    h, w = image.shape[-2], image.shape[-1]
    mask_up = torch.nn.functional.interpolate(
        mask_soft.view(1, 1, mask_soft.shape[0], mask_soft.shape[1]),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).clamp_(0.0, 1.0)
    blurred = gaussian_blur(
        image,
        kernel_size=[int(blur_kernel_size), int(blur_kernel_size)],
        sigma=[float(blur_sigma), float(blur_sigma)],
    )
    return image * (1.0 - mask_up) + blurred * mask_up


def _ensure_odd_kernel_size(value: int) -> int:
    k = max(1, int(value))
    if k % 2 == 0:
        k += 1
    return k


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 alternative: rank (layer,head) by image dependence score "
            "KL(clean_after||corrupted_after) - KL(clean_before||corrupted_before)."
        )
    )
    parser.add_argument("--layers", default="all", help="Layer pool: 'all' or comma-separated indices")
    parser.add_argument("--heads", default="all", help="Head pool: 'all' or comma-separated indices")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k answers used to build union for KL scoring")
    parser.add_argument("--max-length", type=int, default=10, help="Max generation length for top-k answers")
    parser.add_argument("--min-length", type=int, default=1, help="Min generation length for top-k answers")
    parser.add_argument(
        "--out-matrices-jsonl",
        default="out/pair_image_dependence_matrices_k{k}.jsonl",
        help="Output JSONL path template for per-record matrices cache (supports {k})",
    )
    parser.add_argument(
        "--out-rankings-jsonl",
        default="out/pair_image_dependence_rankings_k{k}.jsonl",
        help="Output JSONL path template for per-record ranked pairs (supports {k})",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on processed records (0 = all)")
    parser.add_argument("--print-top-n", type=int, default=70, help="Print top-N ranked (layer,head) pairs per record")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intervention strength")
    parser.add_argument("--blur-kernel-size", type=int, default=31, help="Gaussian blur kernel size for corruption (auto odd)")
    parser.add_argument("--blur-sigma", type=float, default=8.0, help="Gaussian blur sigma for corruption")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if float(args.blur_sigma) <= 0.0:
        raise ValueError("--blur-sigma must be > 0")

    blur_kernel_size = _ensure_odd_kernel_size(int(args.blur_kernel_size))

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
    existing_cache = _load_existing_image_dependence_cache(
        out_matrices_path,
        target_layers=target_layers,
        target_heads=target_heads,
        top_k=args.top_k,
        max_length=args.max_length,
        min_length=args.min_length,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=float(args.blur_sigma),
    )
    if existing_cache:
        print(f"[info] Found {len(existing_cache)} cacheable matrix records in {out_matrices_path}")

    pair_stats: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_score": 0.0,
            "sum_kl_before": 0.0,
            "sum_kl_after": 0.0,
            "before_clean_hits": 0.0,
            "after_clean_hits": 0.0,
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
                image_dependence_matrix = cached_row.get("image_dependence_matrix")
                kl_before_matrix = cached_row.get("kl_before_matrix")
                kl_after_matrix = cached_row.get("kl_after_matrix")
                if (
                    isinstance(image_dependence_matrix, list)
                    and isinstance(kl_before_matrix, list)
                    and isinstance(kl_after_matrix, list)
                ):
                    pair_matrix, ranked_pairs, valid_cells = _build_ranked_pairs_from_image_dependence(
                        image_dependence_matrix,
                        kl_before_matrix,
                        kl_after_matrix,
                        layers=target_layers,
                        heads=target_heads,
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
                                "blur_kernel_size": int(blur_kernel_size),
                                "blur_sigma": float(args.blur_sigma),
                            },
                            "layers": [int(v) for v in target_layers],
                            "heads": [int(v) for v in target_heads],
                            "pred_before_clean": str(cached_row.get("pred_before_clean", "")),
                            "pred_before_corrupted": str(cached_row.get("pred_before_corrupted", "")),
                            "before_hit_clean": int(cached_row.get("before_hit_clean", 0)),
                            "before_hit_corrupted": int(cached_row.get("before_hit_corrupted", 0)),
                            "image_dependence_matrix": image_dependence_matrix,
                            "kl_before_matrix": kl_before_matrix,
                            "kl_after_matrix": kl_after_matrix,
                        }
                        ranking_row: Dict[str, Any] = {
                            "id": rec_id_str,
                            "image": str(image_value),
                            "question": str(question),
                            "prompt": str(prompt or ""),
                            "gold_answer": str(gold),
                            "rank_formula": "KL(clean_after||corrupted_after) - KL(clean_before||corrupted_before)",
                            "answer_union_policy": "union(clean_before, corrupted_before, clean_after, corrupted_after)",
                            "metric_config": matrix_row["metric_config"],
                            "layers": matrix_row["layers"],
                            "heads": matrix_row["heads"],
                            "pred_before_clean": matrix_row["pred_before_clean"],
                            "pred_before_corrupted": matrix_row["pred_before_corrupted"],
                            "before_hit_clean": matrix_row["before_hit_clean"],
                            "before_hit_corrupted": matrix_row["before_hit_corrupted"],
                            "pair_matrix": pair_matrix,
                            "ranked_pairs": ranked_pairs,
                            "top_pairs": ranked_pairs[: max(0, int(args.print_top_n))],
                            "matrix_source": "cache",
                        }
                        matrices_file.write(json.dumps(matrix_row, ensure_ascii=False) + "\n")
                        rankings_file.write(json.dumps(ranking_row, ensure_ascii=False) + "\n")
                        _print_top_pairs(rec_id_str, ranked_pairs, int(args.print_top_n))
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

            corrupted_image = _make_corrupted_image(
                image,
                mask_soft,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=float(args.blur_sigma),
            )

            with torch.no_grad():
                clean_image_embeds = model.visual_encoder(image)
                corrupted_image_embeds = model.visual_encoder(corrupted_image)

            clean_before_state = encode_question_state(model, question_inputs, clean_image_embeds)
            corrupted_before_state = encode_question_state(model, question_inputs, corrupted_image_embeds)
            clean_before_topk = generate_topk_answers(
                model,
                tokenizer,
                clean_before_state,
                top_k=args.top_k,
                max_length=args.max_length,
                min_length=args.min_length,
            )
            corrupted_before_topk = generate_topk_answers(
                model,
                tokenizer,
                corrupted_before_state,
                top_k=args.top_k,
                max_length=args.max_length,
                min_length=args.min_length,
            )
            pred_before_clean = clean_before_topk[0] if clean_before_topk else ""
            pred_before_corrupted = corrupted_before_topk[0] if corrupted_before_topk else ""
            before_hit_clean = int(normalize_answer(pred_before_clean) == gold_norm)
            before_hit_corrupted = int(normalize_answer(pred_before_corrupted) == gold_norm)

            clean_before_logprob_cache: Dict[str, float] = {}
            corrupted_before_logprob_cache: Dict[str, float] = {}
            image_dependence_matrix: List[List[Optional[float]]] = [
                [None for _ in target_heads] for _ in target_layers
            ]
            kl_before_matrix: List[List[Optional[float]]] = [[None for _ in target_heads] for _ in target_layers]
            kl_after_matrix: List[List[Optional[float]]] = [[None for _ in target_heads] for _ in target_layers]

            for layer_idx, head_idx in pairs:
                new_forward = make_forward(head_idx, override_rows, args.alpha)
                apply_override(new_forward, (layer_idx,), originals)
                try:
                    clean_after_state = encode_question_state(model, question_inputs, clean_image_embeds)
                    clean_after_topk = generate_topk_answers(
                        model,
                        tokenizer,
                        clean_after_state,
                        top_k=args.top_k,
                        max_length=args.max_length,
                        min_length=args.min_length,
                    )
                    corrupted_after_state = encode_question_state(model, question_inputs, corrupted_image_embeds)
                    corrupted_after_topk = generate_topk_answers(
                        model,
                        tokenizer,
                        corrupted_after_state,
                        top_k=args.top_k,
                        max_length=args.max_length,
                        min_length=args.min_length,
                    )
                finally:
                    revert_override((layer_idx,), originals)

                merged_answers = union_answers(clean_before_topk, corrupted_before_topk)
                merged_answers = union_answers(merged_answers, clean_after_topk)
                merged_answers = union_answers(merged_answers, corrupted_after_topk)
                if not merged_answers:
                    continue

                logp_clean_before: List[float] = []
                logp_corrupted_before: List[float] = []
                logp_clean_after: List[float] = []
                logp_corrupted_after: List[float] = []
                for answer in merged_answers:
                    if answer not in clean_before_logprob_cache:
                        clean_before_logprob_cache[answer] = answer_log_prob(
                            model,
                            tokenizer,
                            clean_before_state,
                            question_inputs["attention_mask"],
                            answer,
                        )
                    if answer not in corrupted_before_logprob_cache:
                        corrupted_before_logprob_cache[answer] = answer_log_prob(
                            model,
                            tokenizer,
                            corrupted_before_state,
                            question_inputs["attention_mask"],
                            answer,
                        )

                    logp_clean_before.append(clean_before_logprob_cache[answer])
                    logp_corrupted_before.append(corrupted_before_logprob_cache[answer])
                    logp_clean_after.append(
                        answer_log_prob(
                            model,
                            tokenizer,
                            clean_after_state,
                            question_inputs["attention_mask"],
                            answer,
                        )
                    )
                    logp_corrupted_after.append(
                        answer_log_prob(
                            model,
                            tokenizer,
                            corrupted_after_state,
                            question_inputs["attention_mask"],
                            answer,
                        )
                    )

                probs_clean_before = probs_from_log_probs(logp_clean_before)
                probs_corrupted_before = probs_from_log_probs(logp_corrupted_before)
                probs_clean_after = probs_from_log_probs(logp_clean_after)
                probs_corrupted_after = probs_from_log_probs(logp_corrupted_after)
                kl_before = kl_divergence_from_probs(probs_clean_before, probs_corrupted_before)
                kl_after = kl_divergence_from_probs(probs_clean_after, probs_corrupted_after)
                score = float(kl_after - kl_before)

                pred_after_clean = clean_after_topk[0] if clean_after_topk else ""
                after_hit_clean = int(normalize_answer(pred_after_clean) == gold_norm)

                layer_pos = layer_to_pos[int(layer_idx)]
                head_pos = head_to_pos[int(head_idx)]
                image_dependence_matrix[layer_pos][head_pos] = float(score)
                kl_before_matrix[layer_pos][head_pos] = float(kl_before)
                kl_after_matrix[layer_pos][head_pos] = float(kl_after)
                total_rows += 1

                stats = pair_stats[(layer_idx, head_idx)]
                stats["count"] += 1.0
                stats["sum_score"] += float(score)
                stats["sum_kl_before"] += float(kl_before)
                stats["sum_kl_after"] += float(kl_after)
                stats["before_clean_hits"] += float(before_hit_clean)
                stats["after_clean_hits"] += float(after_hit_clean)

            pair_matrix, ranked_pairs, _ = _build_ranked_pairs_from_image_dependence(
                image_dependence_matrix,
                kl_before_matrix,
                kl_after_matrix,
                layers=target_layers,
                heads=target_heads,
            )

            matrix_row = {
                "id": rec_id_str,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "gold_answer": str(gold),
                "metric_config": {
                    "top_k": int(args.top_k),
                    "max_length": int(args.max_length),
                    "min_length": int(args.min_length),
                    "blur_kernel_size": int(blur_kernel_size),
                    "blur_sigma": float(args.blur_sigma),
                },
                "layers": [int(v) for v in target_layers],
                "heads": [int(v) for v in target_heads],
                "pred_before_clean": pred_before_clean,
                "pred_before_corrupted": pred_before_corrupted,
                "before_hit_clean": int(before_hit_clean),
                "before_hit_corrupted": int(before_hit_corrupted),
                "image_dependence_matrix": image_dependence_matrix,
                "kl_before_matrix": kl_before_matrix,
                "kl_after_matrix": kl_after_matrix,
            }
            ranking_row = {
                "id": rec_id_str,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "gold_answer": str(gold),
                "rank_formula": "KL(clean_after||corrupted_after) - KL(clean_before||corrupted_before)",
                "answer_union_policy": "union(clean_before, corrupted_before, clean_after, corrupted_after)",
                "metric_config": matrix_row["metric_config"],
                "layers": matrix_row["layers"],
                "heads": matrix_row["heads"],
                "pred_before_clean": matrix_row["pred_before_clean"],
                "pred_before_corrupted": matrix_row["pred_before_corrupted"],
                "before_hit_clean": matrix_row["before_hit_clean"],
                "before_hit_corrupted": matrix_row["before_hit_corrupted"],
                "pair_matrix": pair_matrix,
                "ranked_pairs": ranked_pairs,
                "top_pairs": ranked_pairs[: max(0, int(args.print_top_n))],
                "matrix_source": "computed",
            }
            matrices_file.write(json.dumps(matrix_row, ensure_ascii=False) + "\n")
            rankings_file.write(json.dumps(ranking_row, ensure_ascii=False) + "\n")
            _print_top_pairs(rec_id_str, ranked_pairs, int(args.print_top_n))
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
    print(
        f"Corruption: masked Gaussian blur (kernel={blur_kernel_size}, sigma={float(args.blur_sigma):.3f})"
    )
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
        mean_score = stats["sum_score"] / count
        mean_kl_before = stats["sum_kl_before"] / count
        mean_kl_after = stats["sum_kl_after"] / count
        before_acc = 100.0 * stats["before_clean_hits"] / count
        after_acc = 100.0 * stats["after_clean_hits"] / count
        print(
            f"(L{layer_idx},H{head_idx}) n={int(stats['count'])} "
            f"| mean score={mean_score:+.6f} "
            f"| mean KL_before={mean_kl_before:+.6f} "
            f"| mean KL_after={mean_kl_after:+.6f} "
            f"| clean_acc_before={before_acc:.2f}% clean_acc_after={after_acc:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
