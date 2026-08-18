import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import (
    apply_override,
    canonicalize_answer,
    data_path as DATA_PATH,
    ensure_mask,
    gaussian_from_mask,
    guess_focus_words,
    image_root,
    iter_jsonl,
    load_mask_from_dir,
    masks_root,
    normalize_answer,
    parse_heads,
    revert_override,
    select_override_indices,
)

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root
MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-answer object hiding strength for maximizing answer logprob."
    )
    parser.add_argument("--layers", default="all", help="Layer spec: 'all' or comma-separated indices.")
    parser.add_argument("--heads", default="all", help="Head spec: 'all' or comma-separated indices.")
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help=(
            "Restrict processing to records whose id/image stem/path matches the given value. "
            "Can be supplied multiple times or as comma-separated values."
        ),
    )
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Input JSONL dataset path.")
    parser.add_argument(
        "--output",
        default="out/fine_tuning_hide_object_results.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--opt-target",
        choices=("gt", "topk_gt_missing"),
        default="topk_gt_missing",
        help="Answer target mode (ground-truth only, or top-K beams + GT-if-missing).",
    )
    parser.add_argument(
        "--answers",
        action="append",
        default=None,
        help="Manual answer targets (repeat flag or comma-separated). Overrides --opt-target.",
    )
    parser.add_argument("--num-beams", type=int, default=3, help="Beam width used for topk_gt_missing.")
    parser.add_argument("--steps-per-answer", type=int, default=40, help="Optimization steps per target answer.")
    parser.add_argument("--lr-hide", type=float, default=0.08, help="Learning rate for hide_strength.")
    parser.add_argument("--lr-alpha", type=float, default=0.05, help="Learning rate for alpha when --learn-alpha.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.90,
        help="Fixed override alpha in [0,1] when --learn-alpha is off.",
    )
    parser.add_argument(
        "--learn-alpha",
        action="store_true",
        help="Jointly optimize alpha together with hide_strength.",
    )
    parser.add_argument(
        "--init-hide",
        type=float,
        default=0.20,
        help="Initial hide_strength in [0,1].",
    )
    parser.add_argument(
        "--hide-floor",
        type=float,
        default=0.25,
        help="Minimum relative attention retained on masked-object regions at full hide.",
    )
    parser.add_argument(
        "--naturalness-weight",
        type=float,
        default=0.05,
        help="Penalty weight on hide_strength^2 (higher -> subtler hiding).",
    )
    parser.add_argument(
        "--no-gaussian",
        action="store_true",
        help="Disable Gaussian blend on softened masks (use softened mask only).",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on processed records (0 = all).")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-step prints.")
    return parser.parse_args()


def _parse_values(raw_values: Optional[Sequence[str]]) -> List[str]:
    if not raw_values:
        return []
    items: List[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        for part in str(raw).replace(";", ",").split(","):
            text = part.strip()
            if text:
                items.append(text)
    return items


def _record_matches(entry: Dict[str, object], targets: Sequence[str]) -> bool:
    if not targets:
        return True
    rec_id = str(entry.get("id", "") or "")
    image = str(entry.get("image", "") or "")
    stem = Path(image).stem if image else ""
    target_set = {str(x) for x in targets}
    return (rec_id in target_set) or (image in target_set) or (stem in target_set)


def _as_text_answer(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _safe_logit(p: float) -> float:
    p_clamped = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p_clamped / (1.0 - p_clamped))


def _answer_key(text: str) -> str:
    canon = canonicalize_answer(text)
    if canon:
        return canon
    return normalize_answer(text)


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


def _generate_top_beams(
    model,
    tokenizer,
    image: torch.Tensor,
    question: str,
    *,
    num_beams: int,
    max_length: int = 10,
    min_length: int = 1,
) -> List[Tuple[str, float]]:
    question_inputs = _prepare_question_inputs(tokenizer, question, image.device)
    with torch.no_grad():
        image_embeds = model.visual_encoder(image)
        image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        question_output = model.text_encoder(
            input_ids=question_inputs["input_ids"],
            attention_mask=question_inputs["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_att_mask,
            return_dict=True,
        )

        question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long, device=question_states.device)
        model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}

        bos_ids = torch.full((image.size(0), 1), fill_value=tokenizer.bos_token_id, device=image.device)
        outputs = model.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            **model_kwargs,
        )

    sequences = outputs.sequences
    scores = outputs.sequences_scores
    beams: List[Tuple[str, float]] = []
    seen: set = set()
    for idx, seq in enumerate(sequences):
        answer_raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
        if not answer_raw:
            continue
        canon = canonicalize_answer(answer_raw)
        answer = canon if canon else answer_raw
        key = _answer_key(answer_raw)
        if key in seen:
            continue
        seen.add(key)
        score = float(scores[idx].item()) if scores is not None else float("nan")
        beams.append((answer, score))
        if len(beams) >= num_beams:
            break
    return beams


def _resolve_target_specs(
    *,
    manual_answers: Sequence[str],
    opt_target: str,
    model,
    tokenizer,
    image: torch.Tensor,
    question: str,
    gt_answer: str,
    num_beams: int,
) -> List[Dict[str, object]]:
    if manual_answers:
        targets: List[Dict[str, object]] = []
        seen: set = set()
        for idx, raw in enumerate(manual_answers, 1):
            cleaned = raw.strip()
            if not cleaned:
                continue
            canon = canonicalize_answer(cleaned)
            answer = canon if canon else cleaned
            key = _answer_key(answer)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                {
                    "opt_answer": answer,
                    "opt_target": f"manual{idx}",
                    "target_beam_rank": None,
                    "target_beam_score": None,
                    "baseline_topk": None,
                }
            )
        return targets

    if opt_target == "topk_gt_missing":
        beams = _generate_top_beams(
            model,
            tokenizer,
            image,
            question,
            num_beams=num_beams,
            max_length=10,
            min_length=1,
        )
        targets = []
        baseline_topk: List[str] = []
        seen_norm: set = set()
        ranked_beams: List[Tuple[int, str, float]] = []

        for rank, (answer, score) in enumerate(beams[:num_beams], 1):
            cleaned = answer.strip()
            if not cleaned:
                continue
            key = _answer_key(cleaned) or cleaned.lower()
            if key in seen_norm:
                continue
            seen_norm.add(key)
            baseline_topk.append(cleaned)
            ranked_beams.append((rank, cleaned, score))

        for rank, cleaned, score in ranked_beams:
            targets.append(
                {
                    "opt_answer": cleaned,
                    "opt_target": f"beam{rank}",
                    "target_beam_rank": rank,
                    "target_beam_score": score,
                    "baseline_topk": baseline_topk.copy(),
                }
            )

        gt_clean = (gt_answer or "").strip()
        if gt_clean:
            gt_canon = canonicalize_answer(gt_clean)
            gt_used = gt_canon if gt_canon else gt_clean
            gt_norm = _answer_key(gt_used)
            if gt_norm and gt_norm not in seen_norm:
                targets.append(
                    {
                        "opt_answer": gt_used,
                        "opt_target": "gt_missing_topk",
                        "target_beam_rank": None,
                        "target_beam_score": None,
                        "baseline_topk": baseline_topk.copy(),
                    }
                )
        if targets:
            return targets

    gt_used = (canonicalize_answer(gt_answer) or gt_answer or "").strip()
    return [
        {
            "opt_answer": gt_used,
            "opt_target": "gt",
            "target_beam_rank": None,
            "target_beam_score": None,
            "baseline_topk": None,
        }
    ]


def _compute_answer_logprob(
    model,
    tokenizer,
    image: torch.Tensor,
    question_inputs: Dict[str, torch.Tensor],
    answer: str,
    *,
    image_embeds: Optional[torch.Tensor] = None,
    image_att_mask: Optional[torch.Tensor] = None,
) -> float:
    cleaned = (answer or "").strip()
    if not cleaned:
        return float("-inf")

    if image_embeds is None:
        with torch.no_grad():
            image_embeds = model.visual_encoder(image)
    if image_att_mask is None:
        image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    input_ids = question_inputs["input_ids"]
    attention_mask = question_inputs["attention_mask"]

    with torch.no_grad():
        question_output = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_att_mask,
            return_dict=True,
        )

        answer_for_prob = canonicalize_answer(cleaned) or cleaned
        ans = tokenizer(answer_for_prob, add_special_tokens=False, return_tensors="pt").input_ids.to(image.device)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise RuntimeError("Tokenizer is missing BOS/EOS token ids.")
        if ans.numel() == 0:
            return float("-inf")

        bos = torch.tensor([[bos_id]], device=image.device)
        eos = torch.tensor([[eos_id]], device=image.device)
        target = torch.cat([bos, ans, eos], dim=1)
        decoder_in = target[:, :-1]
        labels = target[:, 1:]

        output = model.text_decoder(
            input_ids=decoder_in,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=question_inputs["attention_mask"],
            return_dict=True,
        )
        log_probs = torch.log_softmax(output.logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item())


def _build_object_map(
    mask_array: np.ndarray,
    *,
    gh: int,
    gw: int,
    device: torch.device,
    use_gaussian: bool,
) -> torch.Tensor:
    mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
    mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
    mask_small = F.interpolate(mask_tensor, size=(gh, gw), mode="bilinear", align_corners=False)
    mask_small = mask_small.squeeze(0).squeeze(0).clamp_(0.0, 1.0)
    mask_soft = soften_mask(mask_small, ksize=5, iters=2).clamp_(0.0, 1.0)
    if use_gaussian:
        return (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)
    return mask_soft


def _build_hide_grid(object_map: torch.Tensor, hide_strength: torch.Tensor, hide_floor: float) -> torch.Tensor:
    floor = float(min(max(hide_floor, 0.0), 1.0))
    hidden = 1.0 - hide_strength * (1.0 - floor) * object_map
    return hidden.clamp(min=floor, max=1.0)


def _summarize_hiding(object_map: torch.Tensor, hide_grid: torch.Tensor) -> Dict[str, float]:
    weights = object_map.clamp_min(0.0)
    weight_sum = float(weights.sum().item())
    if weight_sum <= 1e-8:
        object_keep = float(hide_grid.mean().item())
    else:
        object_keep = float(((hide_grid * weights).sum() / weights.sum()).item())
    object_hidden = float(max(0.0, min(1.0, 1.0 - object_keep)))
    global_hidden = float(max(0.0, min(1.0, (1.0 - hide_grid).mean().item())))
    return {
        "object_keep_ratio_weighted": object_keep,
        "object_hidden_ratio_weighted": object_hidden,
        "global_hidden_ratio_mean": global_hidden,
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_beams < 1:
        raise ValueError("--num-beams must be >= 1.")
    if args.steps_per_answer < 1:
        raise ValueError("--steps-per-answer must be >= 1.")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1].")
    if not (0.0 <= args.init_hide <= 1.0):
        raise ValueError("--init-hide must be in [0,1].")
    if not (0.0 <= args.hide_floor <= 1.0):
        raise ValueError("--hide-floor must be in [0,1].")
    if args.naturalness_weight < 0:
        raise ValueError("--naturalness-weight must be >= 0.")

    input_path = args.input.expanduser()
    if not input_path.exists():
        print(f"[warn] Input dataset not found: {input_path}", file=sys.stderr)
        return

    records = list(iter_jsonl(input_path))
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        print("No records to process.")
        return

    record_targets = _parse_values(args.record_ids)
    manual_answers = _parse_values(args.answers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = blip_vqa(pretrained=MODEL_URL, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    for p in model.parameters():
        p.requires_grad = False

    heads_spec = parse_heads(args.heads)
    layers_spec = parse_heads(args.layers)
    heads_arg = heads_spec[0] if len(heads_spec) == 1 else heads_spec

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    if layers_spec == (-1,):
        target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        target_layers = tuple(int(x) for x in layers_spec)

    gh = gw = IMAGE_SIZE // PATCH_SIZE
    results: List[Dict[str, object]] = []
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}

    for idx, entry in enumerate(records, 1):
        if record_targets and not _record_matches(entry, record_targets):
            continue

        image_value = entry.get("image")
        question = _as_text_answer(entry.get("question"))
        gt_answer = _as_text_answer(entry.get("answer"))
        prompt = _as_text_answer(entry.get("prompt"))
        rec_id = entry.get("id")

        if not image_value or not question:
            print(f"[warn] Missing image/question in record #{idx}; skipping.", file=sys.stderr)
            continue

        image_rel = Path(str(image_value))
        image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
        if not image_path.exists():
            print(f"[warn] Missing image: {image_path}; skipping.", file=sys.stderr)
            continue

        image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
        question_inputs = _prepare_question_inputs(tokenizer, question, device)

        with torch.no_grad():
            image_embeds = model.visual_encoder(image)
            image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        cache_key = (image_rel.as_posix(), prompt)
        if cache_key in mask_cache:
            mask_array = mask_cache[cache_key]
        else:
            mask_array = load_mask_from_dir(MASK_DIR, image_rel.stem, prompt, rec_id)
            if mask_array is not None:
                mask_cache[cache_key] = mask_array

        try:
            mask_array = ensure_mask(mask_array, gh, gw, stem=image_rel.stem)
        except RuntimeError:
            continue

        object_map = _build_object_map(
            mask_array,
            gh=gh,
            gw=gw,
            device=device,
            use_gaussian=not args.no_gaussian,
        )

        tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
        focus_words = guess_focus_words(question)
        override_indices = select_override_indices(tokens, focus_words, tokenizer)
        if not override_indices:
            override_indices = [0]

        target_specs = _resolve_target_specs(
            manual_answers=manual_answers,
            opt_target=args.opt_target,
            model=model,
            tokenizer=tokenizer,
            image=image,
            question=question,
            gt_answer=gt_answer,
            num_beams=args.num_beams,
        )
        if not target_specs:
            continue

        baseline_logprob_cache: Dict[str, float] = {}
        per_record_results: List[Dict[str, object]] = []

        for target_spec in target_specs:
            opt_answer = str(target_spec["opt_answer"])
            opt_target_used = str(target_spec["opt_target"])
            answer_key = _answer_key(opt_answer) or opt_answer.strip().lower()

            if answer_key in baseline_logprob_cache:
                before_logprob = baseline_logprob_cache[answer_key]
            else:
                before_logprob = _compute_answer_logprob(
                    model,
                    tokenizer,
                    image,
                    question_inputs,
                    opt_answer,
                    image_embeds=image_embeds,
                    image_att_mask=image_att_mask,
                )
                baseline_logprob_cache[answer_key] = before_logprob

            raw_hide = nn.Parameter(torch.tensor(_safe_logit(args.init_hide), device=device))
            params = [{"params": [raw_hide], "lr": float(args.lr_hide)}]

            if args.learn_alpha:
                raw_alpha = nn.Parameter(torch.tensor(_safe_logit(args.alpha), device=device))
                params.append({"params": [raw_alpha], "lr": float(args.lr_alpha)})
            else:
                raw_alpha = None
                fixed_alpha = torch.tensor(float(args.alpha), device=device)

            opt = torch.optim.Adam(params)

            best_total_loss = float("inf")
            best_hide = float(args.init_hide)
            best_alpha = float(args.alpha)
            best_raw_hide = float(raw_hide.detach().item())
            best_raw_alpha = float(raw_alpha.detach().item()) if raw_alpha is not None else None

            if not args.quiet:
                print(
                    f"### Record {idx} ({entry.get('id')}) target={opt_target_used} answer='{opt_answer}' "
                    f"steps={args.steps_per_answer}"
                )

            for step in range(args.steps_per_answer):
                hide_strength = torch.sigmoid(raw_hide)
                alpha_t = torch.sigmoid(raw_alpha) if raw_alpha is not None else fixed_alpha
                hide_grid = _build_hide_grid(object_map, hide_strength, args.hide_floor)
                override_rows = {int(token_idx): hide_grid for token_idx in override_indices}

                for layer_idx in target_layers:
                    apply_override(make_forward(heads_arg, override_rows, alpha_t), (layer_idx,), originals)

                try:
                    weights = torch.tensor([1.0], device=device)
                    ce_loss = model(image, [question], answer=[opt_answer], train=True, n=[1], weights=weights)
                    reg = float(args.naturalness_weight) * (hide_strength ** 2)
                    total_loss = ce_loss + reg

                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()
                finally:
                    revert_override(target_layers, originals)

                total_val = float(total_loss.detach().item())
                if total_val < best_total_loss:
                    best_total_loss = total_val
                    best_hide = float(torch.sigmoid(raw_hide).detach().item())
                    best_alpha = float(torch.sigmoid(raw_alpha).detach().item()) if raw_alpha is not None else float(
                        fixed_alpha.item()
                    )
                    best_raw_hide = float(raw_hide.detach().item())
                    best_raw_alpha = float(raw_alpha.detach().item()) if raw_alpha is not None else None

                if (not args.quiet) and (
                    step == 0 or step == args.steps_per_answer - 1 or (step + 1) % max(1, args.steps_per_answer // 4) == 0
                ):
                    print(
                        f"  step {step + 1:03d}/{args.steps_per_answer}: "
                        f"loss={total_val:+.6f} hide={float(torch.sigmoid(raw_hide).item()):.4f} "
                        f"alpha={float((torch.sigmoid(raw_alpha) if raw_alpha is not None else fixed_alpha).item()):.4f}"
                    )

            with torch.no_grad():
                best_hide_t = torch.sigmoid(torch.tensor(best_raw_hide, device=device))
                best_alpha_t = (
                    torch.sigmoid(torch.tensor(best_raw_alpha, device=device))
                    if best_raw_alpha is not None
                    else torch.tensor(best_alpha, device=device)
                )
                best_hide_grid = _build_hide_grid(object_map, best_hide_t, args.hide_floor)
                best_override_rows = {int(token_idx): best_hide_grid for token_idx in override_indices}

            for layer_idx in target_layers:
                apply_override(make_forward(heads_arg, best_override_rows, best_alpha_t), (layer_idx,), originals)
            try:
                after_logprob = _compute_answer_logprob(
                    model,
                    tokenizer,
                    image,
                    question_inputs,
                    opt_answer,
                    image_embeds=image_embeds,
                    image_att_mask=image_att_mask,
                )
                after_beams = _generate_top_beams(
                    model,
                    tokenizer,
                    image,
                    question,
                    num_beams=max(1, int(args.num_beams)),
                    max_length=10,
                    min_length=1,
                )
            finally:
                revert_override(target_layers, originals)

            hide_stats = _summarize_hiding(object_map, best_hide_grid)
            top_after_answers = [ans for ans, _score in after_beams]
            top_after_scores = [float(score) for _ans, score in after_beams]

            row = {
                "index": idx,
                "id": rec_id,
                "record_id": rec_id,
                "image": image_rel.as_posix(),
                "prompt": prompt,
                "question": question,
                "answer": gt_answer,
                "gold_answer": gt_answer,
                "gold_answer_norm": _answer_key(gt_answer),
                "opt_answer": opt_answer,
                "answer_candidate": opt_answer,
                "answer_norm": _answer_key(opt_answer),
                "opt_target": opt_target_used,
                "combo_index": 1,
                "combo_spec": "hide_object",
                "target_beam_rank": target_spec.get("target_beam_rank"),
                "target_beam_score": target_spec.get("target_beam_score"),
                "baseline_topk": target_spec.get("baseline_topk"),
                "before_logprob": float(before_logprob),
                "after_logprob": float(after_logprob),
                "delta_logprob": float(after_logprob - before_logprob),
                "best_hide_strength": float(best_hide),
                "best_hide": float(best_hide),
                "best_alpha": float(best_alpha),
                "naturalness_weight": float(args.naturalness_weight),
                "hide_floor": float(args.hide_floor),
                "focus_words": list(focus_words),
                "override_indices": [int(i) for i in override_indices],
                "layers": [int(v) for v in target_layers],
                "heads": heads_arg,
                "top_after_answers": top_after_answers,
                "top_after_scores": top_after_scores,
            }
            row.update(hide_stats)
            results.append(row)
            per_record_results.append(row)

            if not args.quiet:
                print(
                    f"  -> best hide={best_hide:.4f}, best alpha={best_alpha:.4f}, "
                    f"delta_logprob={after_logprob - before_logprob:+.6f}, "
                    f"object_hidden={hide_stats['object_hidden_ratio_weighted']:.4f}"
                )

        if per_record_results:
            ranked = sorted(per_record_results, key=lambda r: float(r["best_hide_strength"]), reverse=True)
            top3 = [
                {
                    "answer": str(r["opt_answer"]),
                    "best_hide_strength": float(r["best_hide_strength"]),
                    "best_alpha": float(r["best_alpha"]),
                    "delta_logprob": float(r["delta_logprob"]),
                }
                for r in ranked[:3]
            ]
            for r in per_record_results:
                r["top3_hide_answers_desc"] = top3

    out_path = Path(args.output)
    _write_jsonl(out_path, results)
    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
