import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.blip_vqa import blip_vqa
from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    evaluate_combo_runs_with_consensus,
    path_with_topk,
    prepare_question_inputs,
)
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


def _alpha_to_raw(alpha: float, device: torch.device) -> torch.Tensor:
    clipped = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), device=device, dtype=torch.float32)


def _loss_with_combo_override(
    *,
    model,
    image: torch.Tensor,
    question: str,
    target_answer: str,
    combo_pairs: Sequence[Sequence[int]],
    override_rows: Dict[int, torch.Tensor],
    originals,
    alpha: torch.Tensor,
) -> torch.Tensor:
    layer_to_heads: Dict[int, set] = {}
    for pair in combo_pairs:
        if len(pair) != 2:
            continue
        layer_idx = int(pair[0])
        head_idx = int(pair[1])
        if layer_idx not in layer_to_heads:
            layer_to_heads[layer_idx] = set()
        layer_to_heads[layer_idx].add(head_idx)

    applied_layers: List[int] = []
    try:
        for layer_idx in sorted(layer_to_heads.keys()):
            heads = tuple(sorted(int(h) for h in layer_to_heads[layer_idx]))
            if not heads:
                continue
            heads_arg = heads[0] if len(heads) == 1 else heads
            apply_override(make_forward(heads_arg, override_rows, alpha), (layer_idx,), originals)
            applied_layers.append(layer_idx)

        weights = torch.tensor([1.0], device=image.device)
        loss = model(image, [question], answer=[target_answer], train=True, n=[1], weights=weights)
        return loss
    finally:
        if applied_layers:
            revert_override(tuple(applied_layers), originals)


def _optimize_alpha_for_answer(
    *,
    model,
    image: torch.Tensor,
    question: str,
    target_answer: str,
    combo_pairs_sets: Sequence[Sequence[Sequence[int]]],
    override_rows: Dict[int, torch.Tensor],
    originals,
    steps: int,
    lr: float,
    init_alpha: float,
) -> float:
    raw_alpha = torch.nn.Parameter(_alpha_to_raw(init_alpha, image.device))
    opt = torch.optim.Adam([raw_alpha], lr=float(lr))
    num_sets = max(1, len(combo_pairs_sets))

    for _ in range(max(1, int(steps))):
        alpha = torch.sigmoid(raw_alpha)
        total_loss: Optional[torch.Tensor] = None
        for combo_pairs in combo_pairs_sets:
            loss = _loss_with_combo_override(
                model=model,
                image=image,
                question=question,
                target_answer=target_answer,
                combo_pairs=combo_pairs,
                override_rows=override_rows,
                originals=originals,
                alpha=alpha,
            )
            total_loss = loss if total_loss is None else (total_loss + loss)

        if total_loss is None:
            break
        total_loss = total_loss / float(num_sets)
        opt.zero_grad()
        total_loss.backward()
        opt.step()

    return float(torch.sigmoid(raw_alpha).detach().item())


def _write_pretty_jsonl(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    preferred_keys = [
        "index",
        "id",
        "image",
        "prompt",
        "question",
        "answer",
        "opt_answer",
        "opt_target",
        "best_alpha",
        "union_alpha",
        "union_topk_answers",
        "top3_alpha_answers_desc",
        "combo_count",
        "combo_seed",
        "rankings_source",
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec_idx, result in enumerate(results):
            keys = [k for k in preferred_keys if k in result]
            keys += sorted(k for k in result.keys() if k not in preferred_keys)
            f.write("{\n")
            for key_idx, key in enumerate(keys):
                f.write(f'  "{key}": {json.dumps(result[key], sort_keys=True)}')
                comma = "," if key_idx < len(keys) - 1 else ""
                f.write(comma + "\n")
            f.write("}")
            if rec_idx < len(results) - 1:
                f.write("\n\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize alpha like fine_tuning_L.py, but target answers come from the "
            "union of combo top-k answers used in combo_accuracy.py."
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
        default=Path("out/fine_tuning_combo_union_alpha.jsonl"),
        help="Output JSONL path.",
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
    parser.add_argument("--union-alpha", type=float, default=1.0, help="Alpha used to build union top-k answers")
    parser.add_argument(
        "--include-gold-if-missing",
        action="store_true",
        help="If set, append gold answer to optimization targets when missing from union top-k.",
    )
    parser.add_argument(
        "--max-union-answers",
        type=int,
        default=0,
        help="Cap number of union answers optimized per record (0 = all).",
    )
    parser.add_argument("--steps-per-answer", type=int, default=30, help="Optimization steps per target answer")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for alpha optimization")
    parser.add_argument("--init-alpha", type=float, default=0.5, help="Initial alpha value in [0,1]")
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
    if not (0.0 <= float(args.init_alpha) <= 1.0):
        raise ValueError("--init-alpha must be in [0, 1]")

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
        added_gold_to_targets = False
        if args.include_gold_if_missing and gold_answer_text.strip():
            union_norms = {normalize_answer(a) for a in union_targets}
            if gold_norm and gold_norm not in union_norms:
                union_targets.append(gold_answer_text.strip())
                added_gold_to_targets = True

        if args.max_union_answers > 0:
            union_targets = union_targets[: int(args.max_union_answers)]

        if not combo_pairs_sets:
            print(f"[warn] No combo pair sets for id={rec_id}; skipping", file=sys.stderr)
            continue
        if not union_targets:
            print(f"[warn] No union targets for id={rec_id}; skipping", file=sys.stderr)
            continue

        if not args.quiet:
            print("========================================")
            print(f"id: {rec_id}")
            print(f"record: {rec_idx}/{total_rows}")
            print(f"union targets: {len(union_targets)}")
            print(f"combo sets: {len(combo_pairs_sets)}")

        per_target_rows: List[Dict[str, Any]] = []
        for target_answer in union_targets:
            opt_target = "union_topk"
            if gold_norm and normalize_answer(target_answer) == gold_norm and added_gold_to_targets:
                opt_target = "gold_added"
            elif gold_norm and normalize_answer(target_answer) == gold_norm:
                opt_target = "gold_in_union"

            best_alpha = _optimize_alpha_for_answer(
                model=model,
                image=image,
                question=question_text,
                target_answer=str(target_answer),
                combo_pairs_sets=combo_pairs_sets,
                override_rows=override_rows,
                originals=originals,
                steps=int(args.steps_per_answer),
                lr=float(args.lr),
                init_alpha=float(args.init_alpha),
            )

            rec_result = {
                "index": int(rec_idx),
                "id": rec_id,
                "image": str(image_value),
                "prompt": prompt,
                "question": question_text,
                "answer": gold_answer_text,
                "opt_answer": str(target_answer),
                "opt_target": opt_target,
                "best_alpha": float(best_alpha),
                "union_alpha": float(args.union_alpha),
                "union_topk_answers": union_targets,
                "combo_count": int(len(combo_pairs_sets)),
                "combo_seed": int(rec_seed),
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
                "optimization": {
                    "steps_per_answer": int(args.steps_per_answer),
                    "lr": float(args.lr),
                    "init_alpha": float(args.init_alpha),
                },
            }
            results.append(rec_result)
            per_target_rows.append(rec_result)

            if not args.quiet:
                print(f"target={str(target_answer)!r} | best_alpha={best_alpha:.6f}")

        ranked = sorted(per_target_rows, key=lambda r: float(r["best_alpha"]), reverse=True)
        top3_alpha_answers_desc = [
            {
                "answer": str(item["opt_answer"]),
                "best_alpha": float(item["best_alpha"]),
                "opt_target": str(item["opt_target"]),
            }
            for item in ranked[:3]
        ]
        for item in per_target_rows:
            item["top3_alpha_answers_desc"] = top3_alpha_answers_desc

        if not args.quiet:
            print("[info] Top-3 answers by alpha:")
            for rank, item in enumerate(top3_alpha_answers_desc, 1):
                print(f"  {rank}. {item['answer']} (alpha={item['best_alpha']:.6f}, target={item['opt_target']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_pretty_jsonl(args.output, results)
    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
