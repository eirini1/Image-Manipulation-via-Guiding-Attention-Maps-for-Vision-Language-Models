import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root


#-----------------------Transformation of mask---------------------------------
_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}

def _to_numpy_uint8(mask_hw: torch.Tensor) -> np.ndarray:
    x = mask_hw.detach().float().clamp(0, 1).cpu().numpy()
    return (x * 255.0).astype(np.uint8)

def _to_torch_float(mask_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device)

def opencv_morph(mask_hw: torch.Tensor,
                 op: str = "none",
                 ksize: int = 3,
                 shape: str = "ellipse",
                 iters: int = 1,
                 device=None) -> torch.Tensor:
    if op == "none" or ksize <= 1 or iters <= 0:
        return mask_hw
    kernel = cv2.getStructuringElement(_SHAPES.get(shape, cv2.MORPH_ELLIPSE), (ksize, ksize))
    src = _to_numpy_uint8(mask_hw)
    if op == "erode":
        dst = cv2.erode(src, kernel, iterations=iters)
    elif op == "dilate":
        dst = cv2.dilate(src, kernel, iterations=iters)
    else:
        ops = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "grad": cv2.MORPH_GRADIENT}
        dst = cv2.morphologyEx(src, ops[op], kernel, iterations=iters)
    return _to_torch_float(dst, mask_hw.device if device is None else device)

def translate_mask(mask_hw: torch.Tensor, tx: int = 0, ty: int = 0,
                   border_value: float = 0.0) -> torch.Tensor:
    if tx == 0 and ty == 0:
        return mask_hw
    h, w = mask_hw.shape[-2], mask_hw.shape[-1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    src = _to_numpy_uint8(mask_hw)
    dst = cv2.warpAffine(
        src, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.clip(border_value, 0, 1) * 255),
    )
    return _to_torch_float(dst, mask_hw.device)
#-----------------------------------------------------------------------------


@dataclass
class AnswerProbabilityReport:
    answer: str
    prob: float
    beam_score: float
    decoded_answers: List[str]


def compute_answer_probability(
    model,
    tokenizer,
    image: torch.Tensor,
    question_inputs: Dict[str, torch.Tensor],
    answer: str,
    *,
    image_embeds: Optional[torch.Tensor] = None,
    image_att_mask: Optional[torch.Tensor] = None,
    num_beams: int = 3,
) -> AnswerProbabilityReport:
    cleaned_answer = (answer or "").strip()
    if cleaned_answer == "":
        return AnswerProbabilityReport(
            answer="",
            prob=0.0,
            beam_score=0.0,
            decoded_answers=[],
        )

    if image_embeds is None:
        with torch.no_grad():
            image_embeds = model.visual_encoder(image)

    if image_att_mask is None:
        image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    input_ids = question_inputs["input_ids"]
    att_mask = question_inputs["attention_mask"]

    with torch.no_grad():
        question_output = model.text_encoder(
            input_ids=input_ids,
            attention_mask=att_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_att_mask,
            output_attentions=True,
            return_dict=True,
        )

        question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
        question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
        model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
        
        bos_ids = torch.full((image.size(0),1),fill_value=tokenizer.bos_token_id,device=image.device)
        
        outputs = model.text_decoder.generate(input_ids=bos_ids,
                                                max_length=10,
                                                min_length=1,
                                                num_beams=num_beams,
                                                eos_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id, 
                                                num_return_sequences=3,
                                                return_dict_in_generate=True,
                                                output_scores=True,
                                                **model_kwargs)
        
        sequences = outputs.sequences
        sequence_scores = outputs.sequences_scores

        pad_id = tokenizer.pad_token_id
        bos_id = tokenizer.bos_token_id
        lengths: List[int] = []
        for seq in sequences:
            non_pad = seq[seq != pad_id] if pad_id is not None else seq
            if non_pad.numel() and bos_id is not None and non_pad[0].item() == bos_id:
                gen_len = int(non_pad.numel() - 1)
            else:
                gen_len = int(non_pad.numel())
            lengths.append(max(gen_len, 1))
        length_penalty = 1.0
        lengths_t = torch.tensor(lengths, device=sequence_scores.device, dtype=sequence_scores.dtype)
        reconstructed = sequence_scores * ((lengths_t-1) ** length_penalty)

    decoded_answers = []
    decoded_norms: List[str] = []
    target_norm = normalize_answer(cleaned_answer)
    beam_score = 0.0
    for i in range(sequences.size(0)):
        seq = sequences[i]
        decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
        decoded_answers.append(decoded)
        decoded_norm = normalize_answer(decoded)
        decoded_norms.append(decoded_norm)
        if decoded_norm == target_norm:
            beam_score = np.exp(float(reconstructed[i].item()))
            
        # elif target_norm and target_norm in decoded_norm:
        #     beam_score = np.exp(float(reconstructed[i].item()))

    if target_norm not in decoded_norms:
        print("Target answer is not in the top-3 beams.")


    ans = tokenizer(
        cleaned_answer,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(image.device)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing BOS or EOS token id needed for probability computation.")

    bos = torch.tensor([[bos_id]], device=image.device)
    eos = torch.tensor([[eos_id]], device=image.device)
    target = torch.cat([bos, ans, eos], dim=1)
    decoder_in = target[:, :-1]
    labels = target[:, 1:]

    with torch.no_grad():
        output = model.text_decoder(
            input_ids=decoder_in,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=question_inputs["attention_mask"],
            return_dict=True,
        )
        log_probs = torch.log_softmax(output.logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        log_prob_sum = token_log_probs.sum()
        prob = float(torch.exp(log_prob_sum).item())
    #prob = float(log_probs_sum.item())

    return AnswerProbabilityReport(
        answer=cleaned_answer,
        prob=prob,
        beam_score=beam_score,
        decoded_answers=decoded_answers,
    )

def _build_prob_rows(probability_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in probability_reports:
        before = report["before"]
        after = report["after"]
        rows.append(
            {
                "record_id": report.get("record_id", "") or "",
                "prob_before": before.prob,
                "prob_after": after.prob,
                "prob_delta": after.prob - before.prob,
                "beam_before": before.beam_score,
                "beam_after": after.beam_score,
                "beam_delta": after.beam_score - before.beam_score,
                "correct_after": 1 if report.get("hit") else 0,
            }
        )
    return rows


def _normalize_matrix_path(path_str: str) -> Path:
    out_path = Path(path_str)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".csv")
    return out_path


def _normalize_csv_path(path_str: str) -> Path:
    out_path = Path(path_str)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    return out_path


def _normalize_jsonl_path(path_str: str) -> Path:
    out_path = Path(path_str)
    if out_path.suffix.lower() != ".jsonl":
        out_path = out_path.with_suffix(".jsonl")
    return out_path


def _resolve_stage1_path(path_str: str, *, shard_idx: int, num_shards: int) -> Path:
    if "{shard}" in path_str or "{num_shards}" in path_str:
        resolved = path_str.format(shard=shard_idx, num_shards=num_shards)
        return _normalize_jsonl_path(resolved)
    out_path = _normalize_jsonl_path(path_str)
    if num_shards > 1:
        out_path = out_path.with_name(f"{out_path.stem}_shard-{shard_idx}{out_path.suffix}")
    return out_path


def _resolve_stage1_paths_all(path_str: str, *, num_shards: int) -> List[Path]:
    paths: List[Path] = []
    for shard_idx in range(num_shards):
        paths.append(_resolve_stage1_path(path_str, shard_idx=shard_idx, num_shards=num_shards))
    return paths


def _write_prob_matrix(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = [
        "record_id",
        "prob_before",
        "prob_after",
        "prob_delta",
        "beam_before",
        "beam_after",
        "beam_delta",
        "correct_after",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _split_specs(specs: Optional[str]) -> List[str]:
    if not specs:
        return []
    return [part.strip() for part in specs.split(";") if part.strip()]

def _parse_int_list(spec: Optional[str]) -> List[int]:
    if not spec:
        return []
    parts = [p for p in spec.replace(";", ",").split(",") if p.strip() != ""]
    vals: List[int] = []
    for part in parts:
        vals.append(int(part.strip()))
    return vals

def _format_combo_spec(combo: Tuple[int, ...]) -> str:
    return ",".join(str(x) for x in combo)


def _build_head_combos(total_heads: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for mask in range(1, 1 << total_heads):
        heads = tuple(i for i in range(total_heads) if (mask >> i) & 1)
        combos.append(heads)
    return combos


def _select_uniform_by_accuracy(
    rows: List[Dict[str, Any]],
    count: int,
    *,
    bins: int,
) -> List[Dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if bins < 1:
        bins = 1
    sort_key = lambda row: (row["accuracy"], row["correct_after"])
    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    accuracies = [row["accuracy"] for row in rows_sorted]
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    if max_acc <= min_acc:
        return rows_sorted[:count]

    bin_width = (max_acc - min_acc) / bins
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bins)]
    for row in rows_sorted:
        idx = int((row["accuracy"] - min_acc) / bin_width) if bin_width > 0 else 0
        if idx >= bins:
            idx = bins - 1
        buckets[idx].append(row)

    # Round-robin across non-empty bins to keep accuracy distribution uniform.
    for bucket in buckets:
        bucket.sort(key=sort_key, reverse=True)
    active = [i for i in range(bins) if buckets[i]]
    if not active:
        return []
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and active:
        bin_idx = active[cursor % len(active)]
        selected.append(buckets[bin_idx].pop(0))
        if not buckets[bin_idx]:
            active.remove(bin_idx)
            if not active:
                break
            cursor = cursor % len(active)
            continue
        cursor += 1
    return selected


def _get_total_heads(model) -> int:
    cfg = getattr(model, "text_encoder", None)
    cfg = getattr(cfg, "config", None)
    if cfg is not None:
        for name in ("num_attention_heads", "num_heads"):
            val = getattr(cfg, name, None)
            if isinstance(val, int) and val > 0:
                return val
    layers = model.text_encoder.encoder.layer
    if layers:
        sa = layers[0].crossattention.self
        for name in ("num_attention_heads", "num_heads"):
            val = getattr(sa, name, None)
            if isinstance(val, int) and val > 0:
                return val
    raise RuntimeError("Unable to determine number of attention heads.")


def _resolve_matrix_out_path(
    path_str: Optional[str],
    heads_spec: str,
    layers_spec: str,
    multi_combo: bool,
    tx: Optional[int] = None,
    ty: Optional[int] = None,
) -> Optional[Path]:
    if not path_str:
        return None
    if "{heads}" in path_str or "{layers}" in path_str or "{tx}" in path_str or "{ty}" in path_str:
        resolved = path_str.format(
            heads=sanitize_name(heads_spec),
            layers=sanitize_name(layers_spec),
            tx=tx if tx is not None else "",
            ty=ty if ty is not None else "",
        )
        return _normalize_matrix_path(resolved)
    out_path = _normalize_matrix_path(path_str)
    if multi_combo:
        suffix = f"heads-{sanitize_name(heads_spec)}_layers-{sanitize_name(layers_spec)}"
        out_path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")
    if tx is not None or ty is not None:
        tx_val = "" if tx is None else str(tx)
        ty_val = "" if ty is None else str(ty)
        out_path = out_path.with_name(f"{out_path.stem}_tx-{tx_val}_ty-{ty_val}{out_path.suffix}")
    return out_path


def _run_combo(
    *,
    heads_spec: str,
    layers_spec: str,
    records: List[Dict[str, Any]],
    model,
    tokenizer,
    originals,
    total_layers: int,
    device: torch.device,
    args: argparse.Namespace,
    target_id_norm: Optional[str],
    report_all_probs: bool,
    multi_combo: bool,
    prob_matrix_out_override: Optional[str] = None,
    print_matrix: bool = True,
    shift_grid: bool = False,
    tx_values: Optional[List[int]] = None,
    ty_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    heads = parse_heads(heads_spec)
    layers_spec_parsed = parse_heads(layers_spec)

    if layers_spec_parsed == (-1,):
        target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        target_layers = tuple(int(x) for x in layers_spec_parsed)

    if multi_combo:
        print("=" * 60)
        print(f"Combo heads={heads_spec} layers={layers_spec}")

    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    probability_reports: List[Dict[str, Any]] = []
    need_probability_report = bool(target_id_norm or report_all_probs)
    total_items = 0
    baseline_correct = 0
    changed_correct = 0
    retained_correct = 0
    changed_correct_ids: List[str] = []
    retained_correct_ids: List[str] = []

    if shift_grid:
        if not tx_values:
            tx_values = [args.tx]
        if not ty_values:
            ty_values = [args.ty]

        cached_records: List[Dict[str, Any]] = []
        for idx, entry in enumerate(records, 1):
            rec_id_val = entry.get("id")
            rec_id_str = "" if rec_id_val is None else str(rec_id_val)
            rec_id_norm = rec_id_str.strip().lower()

            image_value = entry.get("image")
            prompt = entry.get("prompt")
            if image_value is None or entry.get("question") is None or entry.get("answer") is None:
                print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
                continue

            if target_id_norm and rec_id_norm != target_id_norm:
                continue

            image_rel = Path(image_value)
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
                continue

            stem = image_rel.stem
            question = entry["question"]
            gold = entry["answer"]
            gold_norm = normalize_answer(gold)

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)

            mask_array: Optional[np.ndarray] = None
            cache_key = (image_rel.as_posix(), prompt)
            if cache_key in mask_cache:
                mask_array = mask_cache[cache_key]
            else:
                rec_id_for_mask = rec_id_str if rec_id_str else None
                mask_array = load_mask_from_dir(MASK_DIR, stem, prompt or "", rec_id_for_mask)
                if mask_array is not None:
                    mask_cache[cache_key] = mask_array

            gh = gw = IMAGE_SIZE // PATCH_SIZE
            try:
                mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
            except RuntimeError:
                continue

            mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
            mask_small = torch.nn.functional.interpolate(
                mask_tensor,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
            mask_soft = soften_mask(mask_small, ksize=5, iters=2)

            gauss = gaussian_from_mask(mask_soft)
            mask_soft = (gauss * mask_soft).clamp_(0.0, 1.0)
            mask_soft = opencv_morph(mask_soft, op=args.morph, ksize=args.ksize,
                                     shape=args.shape, iters=args.iters)

            question_inputs = tokenizer(
                question,
                padding="longest",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            )
            question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
            question_inputs["input_ids"][:, 0] = tokenizer.enc_token_id
            tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
            focus_words = guess_focus_words(question)
            override_indices = select_override_indices(tokens, focus_words, tokenizer)
            if not override_indices:
                override_indices = [0]

            image_embeds = None
            image_att_mask = None
            baseline_probs: Optional[AnswerProbabilityReport] = None
            if need_probability_report:
                with torch.no_grad():
                    image_embeds = model.visual_encoder(image)
                image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
                baseline_probs = compute_answer_probability(
                    model,
                    tokenizer,
                    image,
                    question_inputs,
                    gold,
                    image_embeds=image_embeds,
                    image_att_mask=image_att_mask,
                )

            with torch.no_grad():
                baseline_pred = model(image, question, train=False, inference="generate")[0]

            cached_records.append(
                {
                    "record_id": rec_id_str or "",
                    "question": question,
                    "gold": gold,
                    "gold_norm": gold_norm,
                    "image": image,
                    "mask_soft": mask_soft,
                    "question_inputs": question_inputs,
                    "override_indices": override_indices,
                    "image_embeds": image_embeds,
                    "image_att_mask": image_att_mask,
                    "baseline_probs": baseline_probs,
                    "baseline_pred": baseline_pred,
                }
            )

        for tx in tx_values:
            for ty in ty_values:
                print(f"[shift] tx={tx} ty={ty}")
                probability_reports = []
                total_items = 0
                baseline_correct = 0
                changed_correct = 0
                retained_correct = 0
                changed_correct_ids = []
                retained_correct_ids = []

                for record in cached_records:
                    gold = record["gold"]
                    gold_norm = record["gold_norm"]
                    image = record["image"]
                    mask_soft = translate_mask(record["mask_soft"], tx=tx, ty=ty, border_value=0.0)
                    override_rows = {i: mask_soft for i in record["override_indices"]}

                    heads_arg = heads[0] if len(heads) == 1 else heads
                    new_forward = make_forward(heads_arg, override_rows)

                    changed_probs: Optional[AnswerProbabilityReport] = None
                    apply_override(new_forward, target_layers, originals)
                    try:
                        with torch.no_grad():
                            pred = model(image, record["question"], train=False, inference="generate")[0]
                        if need_probability_report and record["image_embeds"] is not None and record["image_att_mask"] is not None:
                            changed_probs = compute_answer_probability(
                                model,
                                tokenizer,
                                image,
                                record["question_inputs"],
                                gold,
                                image_embeds=record["image_embeds"],
                                image_att_mask=record["image_att_mask"],
                            )
                    finally:
                        revert_override(target_layers, originals)

                    pred_norm = normalize_answer(pred)
                    hit = int(pred_norm == gold_norm)
                    baseline_hit = int(normalize_answer(record["baseline_pred"]) == gold_norm)
                    total_items += 1
                    baseline_correct += baseline_hit
                    changed_correct += hit
                    rec_label = record["record_id"] or "<unknown>"
                    if hit:
                        changed_correct_ids.append(rec_label)
                    if baseline_hit and hit:
                        retained_correct += 1
                        retained_correct_ids.append(rec_label)

                    if need_probability_report:
                        if record["baseline_probs"] is None or changed_probs is None:
                            print(f"[warn] Unable to compute probability report for record '{rec_label}'.", file=sys.stderr)
                        else:
                            probability_reports.append(
                                {
                                    "record_id": record["record_id"],
                                    "question": record["question"],
                                    "gold": gold,
                                    "before": record["baseline_probs"],
                                    "after": changed_probs,
                                    "prediction": pred,
                                    "hit": bool(hit),
                                }
                            )
                        if target_id_norm:
                            break

                if need_probability_report:
                    matrix_out = prob_matrix_out_override
                    if matrix_out is None:
                        matrix_out = args.prob_matrix_out
                    out_path = _resolve_matrix_out_path(matrix_out, heads_spec, layers_spec, multi_combo, tx=tx, ty=ty)
                    if out_path:
                        rows = _build_prob_rows(probability_reports)
                        if rows:
                            _write_prob_matrix(out_path, rows)
                            print(f"[info] Wrote probability matrix to {out_path}")

                accuracy = 0.0 if total_items == 0 else changed_correct / total_items
                print(f"Accuracy (after override): {accuracy:.6f}")

        return {
            "heads_spec": heads_spec,
            "layers_spec": layers_spec,
            "total_items": total_items,
            "correct_after": changed_correct,
            "accuracy": 0.0 if total_items == 0 else changed_correct / total_items,
        }

    for idx, entry in enumerate(records, 1):
        rec_id_val = entry.get("id")
        rec_id_str = "" if rec_id_val is None else str(rec_id_val)
        rec_id_norm = rec_id_str.strip().lower()

        image_value = entry.get("image")
        prompt = entry.get("prompt")
        if image_value is None or entry.get("question") is None or entry.get("answer") is None:
            print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
            continue

        if target_id_norm and rec_id_norm != target_id_norm:
            continue

        image_rel = Path(image_value)
        image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
        if not image_path.exists():
            print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
            continue

        stem = image_rel.stem
        question = entry["question"]
        gold = entry["answer"]
        gold_norm = normalize_answer(gold)

        image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)

        # Load or fallback mask
        mask_array: Optional[np.ndarray] = None
        cache_key = (image_rel.as_posix(), prompt)
        if cache_key in mask_cache:
            mask_array = mask_cache[cache_key]
        else:
            rec_id_for_mask = rec_id_str if rec_id_str else None
            mask_array = load_mask_from_dir(MASK_DIR, stem, prompt or "", rec_id_for_mask)
            if mask_array is not None:
                mask_cache[cache_key] = mask_array

        gh = gw = IMAGE_SIZE // PATCH_SIZE
        try:
            mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
        except RuntimeError:
            continue

        mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
        mask_small = torch.nn.functional.interpolate(
            mask_tensor,
            size=(gh, gw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
        mask_soft = soften_mask(mask_small, ksize=5, iters=2)

        gauss = gaussian_from_mask(mask_soft)
        mask_soft = (gauss * mask_soft).clamp_(0.0, 1.0)
        mask_soft = opencv_morph(mask_soft, op=args.morph, ksize=args.ksize,
                                 shape=args.shape, iters=args.iters)
        mask_soft = translate_mask(mask_soft, tx=args.tx, ty=args.ty, border_value=0.0)

        # Build token-level override rows
        question_inputs = tokenizer(
            question,
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
        question_inputs["input_ids"][:, 0] = tokenizer.enc_token_id
        tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
        focus_words = guess_focus_words(question)
        override_indices = select_override_indices(tokens, focus_words, tokenizer)
        if not override_indices:
            override_indices = [0]
        override_rows = {i: mask_soft for i in override_indices}

        compute_probs = need_probability_report
        image_embeds = None
        image_att_mask = None
        baseline_probs: Optional[AnswerProbabilityReport] = None
        if compute_probs:
            with torch.no_grad():
                image_embeds = model.visual_encoder(image)
            image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            baseline_probs = compute_answer_probability(
                model,
                tokenizer,
                image,
                question_inputs,
                gold,
                image_embeds=image_embeds,
                image_att_mask=image_att_mask,
            )

        # Prepare attention override for requested heads
        heads_arg = heads[0] if len(heads) == 1 else heads
        new_forward = make_forward(heads_arg, override_rows)

        # Apply override, run inference, revert
        with torch.no_grad():
            baseline_pred = model(image, question, train=False, inference="generate")[0]
        changed_probs: Optional[AnswerProbabilityReport] = None
        apply_override(new_forward, target_layers, originals)
        try:
            with torch.no_grad():
                pred = model(image, question, train=False, inference="generate")[0]
            if compute_probs and image_embeds is not None and image_att_mask is not None:
                changed_probs = compute_answer_probability(
                    model,
                    tokenizer,
                    image,
                    question_inputs,
                    gold,
                    image_embeds=image_embeds,
                    image_att_mask=image_att_mask,
                )
        finally:
            revert_override(target_layers, originals)

        pred_norm = normalize_answer(pred)
        hit = int(pred_norm == gold_norm)
        baseline_hit = int(normalize_answer(baseline_pred) == gold_norm)
        total_items += 1
        baseline_correct += baseline_hit
        changed_correct += hit
        rec_label = rec_id_str or "<unknown>"
        if hit:
            changed_correct_ids.append(rec_label)
        if baseline_hit and hit:
            retained_correct += 1
            retained_correct_ids.append(rec_label)

        if need_probability_report:
            if baseline_probs is None or changed_probs is None:
                rec_label = rec_id_str or "<unknown>"
                print(f"[warn] Unable to compute probability report for record '{rec_label}'.", file=sys.stderr)
            else:
                probability_reports.append(
                    {
                        "record_id": rec_id_str or "",
                        "question": question,
                        "gold": gold,
                        "before": baseline_probs,
                        "after": changed_probs,
                        "prediction": pred,
                        "hit": bool(hit),
                    }
                )
            if target_id_norm:
                break
            if report_all_probs:
                continue

    if need_probability_report:
        if not probability_reports:
            if target_id_norm:
                print(f"[warn] Record id '{args.target_id}' not found in dataset.", file=sys.stderr)
            else:
                print("[warn] No probability reports were generated.", file=sys.stderr)
            return {
                "heads_spec": heads_spec,
                "layers_spec": layers_spec,
                "total_items": total_items,
                "correct_after": changed_correct,
                "accuracy": 0.0 if total_items == 0 else changed_correct / total_items,
            }

        matrix_out = prob_matrix_out_override
        if matrix_out is None:
            matrix_out = args.prob_matrix_out
        out_path = _resolve_matrix_out_path(matrix_out, heads_spec, layers_spec, multi_combo)
        if out_path:
            rows = _build_prob_rows(probability_reports)
            if rows:
                _write_prob_matrix(out_path, rows)
                print(f"[info] Wrote probability matrix to {out_path}")
            else:
                print("[warn] Probability matrix requested but no rows to write.", file=sys.stderr)

        if target_id_norm:
            report = probability_reports[0]
            before = report["before"]
            after = report["after"]
            prob_delta = after.prob - before.prob
            beam_delta = after.beam_score - before.beam_score
            rec_label = report["record_id"] or "<unknown>"
            print(f"[record {rec_label}] Question: {report['question']}")
            print(f"  Gold answer: {report['gold']}")
            print("  Metric         Before        After        Delta")
            print(f"  prob       {before.prob:12.6f} {after.prob:12.6f} {prob_delta:+12.6f}")
            print(f"  beam       {before.beam_score:12.6f} {after.beam_score:12.6f} {beam_delta:+12.6f}")
            print(f"  Post-change prediction: {report['prediction']} (matches gold = {report['hit']})")
            print(f"  Top decodes before: {before.decoded_answers[:3]}")
            print(f"  Top decodes after : {after.decoded_answers[:3]}")
            print(f"  Right answers (baseline): {baseline_correct}")
            print(f"  Right answers (after override): {changed_correct}")
            print(f"  Remained right answers: {retained_correct}")
            print(f"  Right answers (after override) IDs: {changed_correct_ids}")
            print(f"  Remained right answers IDs: {retained_correct_ids}")
            accuracy = 0.0 if total_items == 0 else changed_correct / total_items
            print(f"Accuracy (after override): {accuracy:.6f}")
            return {
                "heads_spec": heads_spec,
                "layers_spec": layers_spec,
                "total_items": total_items,
                "correct_after": changed_correct,
                "accuracy": 0.0 if total_items == 0 else changed_correct / total_items,
            }

        # report_all_probs flow
        prob_deltas: List[float] = []
        beam_deltas: List[float] = []
        prob_improved = 0
        prob_worsened = 0
        prob_unchanged = 0
        beam_improved = 0
        beam_worsened = 0
        beam_unchanged = 0
        if print_matrix:
            print("record_id   prob_before   prob_after   prob_delta   beam_before   beam_after   beam_delta   correct_after")
        for report in probability_reports:
            before = report["before"]
            after = report["after"]
            prob_delta = after.prob - before.prob
            beam_delta = after.beam_score - before.beam_score
            prob_deltas.append(prob_delta)
            beam_deltas.append(beam_delta)
            if prob_delta > 0:
                prob_improved += 1
            elif prob_delta < 0:
                prob_worsened += 1
            else:
                prob_unchanged += 1
            if beam_delta > 0:
                beam_improved += 1
            elif beam_delta < 0:
                beam_worsened += 1
            else:
                beam_unchanged += 1
            if print_matrix:
                rec_label = report["record_id"] or "<unknown>"
                correct_after = "1" if report.get("hit") else "0"
                print(
                    f"{rec_label:>8} {before.prob:12.6f} {after.prob:11.6f} {prob_delta:+11.6f} "
                    f"{before.beam_score:12.6f} {after.beam_score:11.6f} {beam_delta:+11.6f} {correct_after:>13}"
                )

        if prob_deltas:
            avg_prob_delta = float(np.mean(prob_deltas))
            median_prob_delta = float(np.median(prob_deltas))
        else:
            avg_prob_delta = median_prob_delta = 0.0
        if beam_deltas:
            avg_beam_delta = float(np.mean(beam_deltas))
            median_beam_delta = float(np.median(beam_deltas))
        else:
            avg_beam_delta = median_beam_delta = 0.0

        print("-" * 40)
        print(f"Total records: {len(probability_reports)}")
        print(f"Prob improved / worse / same: {prob_improved} / {prob_worsened} / {prob_unchanged}")
        print(f"Beam improved / worse / same: {beam_improved} / {beam_worsened} / {beam_unchanged}")
        print(f"Average prob delta: {avg_prob_delta:+.6f}")
        print(f"Median  prob delta: {median_prob_delta:+.6f}")
        print(f"Average beam delta: {avg_beam_delta:+.6f}")
        print(f"Median  beam delta: {median_beam_delta:+.6f}")
        print(f"Right answers (after override): {changed_correct}/{total_items}")
        print(f"Remained right answers: {retained_correct}/{baseline_correct}")
        print(f"Right answers (after override) IDs: {changed_correct_ids}")
        print(f"Remained right answers IDs: {retained_correct_ids}")
    accuracy = 0.0 if total_items == 0 else changed_correct / total_items
    print(f"Accuracy (after override): {accuracy:.6f}")
    return {
        "heads_spec": heads_spec,
        "layers_spec": layers_spec,
        "total_items": total_items,
        "correct_after": changed_correct,
        "accuracy": accuracy,
    }

def _write_stage1_summary(
    out_path: Path,
    head_summaries: List[Dict[str, Any]],
    layer_summaries: List[Dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in head_summaries:
            payload = dict(row)
            payload["type"] = "head"
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        for row in layer_summaries:
            payload = dict(row)
            payload["type"] = "layer"
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_stage1_summary(out_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    head_summaries: List[Dict[str, Any]] = []
    layer_summaries: List[Dict[str, Any]] = []
    for row in iter_jsonl(out_path):
        row_type = str(row.get("type", "")).strip().lower()
        if row_type == "head":
            head_summaries.append(row)
        elif row_type == "layer":
            layer_summaries.append(row)
    return head_summaries, layer_summaries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Override attention on selected layers/heads and evaluate")
    parser.add_argument("--layers", default="all", help="Layer spec: 'all' or comma-separated indices")
    parser.add_argument("--heads", default="all", help="Head spec: 'all' or comma-separated indices")
    parser.add_argument(
        "--layers-set",
        default=None,
        help="Semicolon-separated list of layer specs to try (overrides --layers).",
    )
    parser.add_argument(
        "--heads-set",
        default=None,
        help="Semicolon-separated list of head specs to try (overrides --heads).",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Exhaustive grid search over all non-empty head/layer combos (use with shards).",
    )
    parser.add_argument(
        "--two-stage-search",
        action="store_true",
        help="Search heads with layers=all and layers with heads=all, then run top heads x top layers.",
    )
    parser.add_argument(
        "--top-heads",
        type=int,
        default=10,
        help="Number of top head combos to keep in two-stage search.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
        help="Number of top layer combos to keep in two-stage search.",
    )
    parser.add_argument(
        "--stage1-out",
        default=None,
        help="Stage-1 summary JSONL to read/write in two-stage search. Supports {shard}/{num_shards} placeholders.",
    )
    parser.add_argument(
        "--stage1-num-shards",
        type=int,
        default=1,
        help="Number of shards for stage-1 in two-stage search.",
    )
    parser.add_argument(
        "--stage1-shard-idx",
        type=int,
        default=0,
        help="Shard index for stage-1 in two-stage search (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards for exhaustive grid search.",
    )
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Shard index for exhaustive grid search (0-based).",
    )
    parser.add_argument(
        "--target-id",
        default=None,
        help="Dataset record id to report ground-truth probabilities before/after override (case-insensitive).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to dataset JSONL (defaults to utils2.data_path).",
    )
    parser.add_argument(
        "--report-all-probs",
        action="store_true",
        help="Report ground-truth probability deltas for every record in the dataset.",
    )
    parser.add_argument(
        "--prob-matrix-out",
        default=None,
        help="Write per-record probability metrics to a file (.csv or .jsonl). Use {heads}/{layers} placeholders for multi-combo outputs.",
    )
    parser.add_argument(
        "--combo-summary-out",
        default=None,
        help="Write per-combo accuracy summary to a CSV file.",
    )
    parser.add_argument(
        "--top-of-all",
        type=int,
        default=0,
        help="In two-stage search, use the top combinations from stage 1 without top heads x top layers. (it will have all heads or all layers)",
    )
    parser.add_argument(
        "--top-uniform",
        type=int,
        default=0,
        help="In two-stage search, select combos from stage 1 with a more uniform accuracy distribution (overrides --top-of-all).",
    )
    parser.add_argument(
        "--uniform-bins",
        type=int,
        default=10,
        help="Number of accuracy bins to use with --top-uniform.",
    )
    #--------------------transform mask-------------------------------------------
    parser.add_argument("--morph", default="none",
    choices=["none","erode","dilate","open","close","grad"])
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--shape", default="ellipse",
        choices=["rect","ellipse","cross"])
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--ty", type=int, default=0)
    parser.add_argument(
        "--shift-grid",
        action="store_true",
        help="Loop over tx/ty combinations and reuse per-record caches.",
    )
    parser.add_argument(
        "--tx-set",
        default=None,
        help="Comma/semicolon-separated list of tx values for --shift-grid.",
    )
    parser.add_argument(
        "--ty-set",
        default=None,
        help="Comma/semicolon-separated list of ty values for --shift-grid.",
    )
    #-----------------------------------------------------------------------------

    args = parser.parse_args(argv)
    target_id_norm = str(args.target_id).strip().lower() if args.target_id else None
    if target_id_norm and args.report_all_probs:
        parser.error("--target-id and --report-all-probs cannot be used together.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = data_path
    if args.data_path:
        dataset_path = Path(args.data_path)

    records = list(iter_jsonl(dataset_path))
    if not records:
        print(f"No records to evaluate in {dataset_path}.")
        return 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base")
    model.eval()
    model = model.to(device)

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    tokenizer = model.tokenizer

    tx_values = _parse_int_list(args.tx_set)
    ty_values = _parse_int_list(args.ty_set)

    if args.grid_search:
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError("--shard-idx must be within [0, num-shards)")
        total_heads = _get_total_heads(model)
        head_combos = _build_head_combos(total_heads)
        layer_combos = build_layer_combos(total_layers, 1, 0)
        heads_specs = [_format_combo_spec(h) for h in head_combos]
        layers_specs = [_format_combo_spec(l) for l in layer_combos]
        multi_combo = True
        shard_counter = 0
        combo_summaries: List[Dict[str, Any]] = []
        for heads_spec in heads_specs:
            for layers_spec in layers_specs:
                if shard_counter % args.num_shards != args.shard_idx:
                    shard_counter += 1
                    continue
                summary = _run_combo(
                    heads_spec=heads_spec,
                    layers_spec=layers_spec,
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    originals=originals,
                    total_layers=total_layers,
                    device=device,
                    args=args,
                    target_id_norm=target_id_norm,
                    report_all_probs=bool(args.report_all_probs),
                    multi_combo=multi_combo,
                    shift_grid=args.shift_grid,
                    tx_values=tx_values,
                    ty_values=ty_values,
                )
                combo_summaries.append(summary)
                shard_counter += 1
        if args.combo_summary_out:
            out_path = _normalize_csv_path(args.combo_summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["heads_spec", "layers_spec", "correct_after", "total_items", "accuracy"],
                )
                writer.writeheader()
                for row in combo_summaries:
                    writer.writerow(
                        {
                            "heads_spec": row["heads_spec"],
                            "layers_spec": row["layers_spec"],
                            "correct_after": row["correct_after"],
                            "total_items": row["total_items"],
                            "accuracy": row["accuracy"],
                        }
                    )
            print(f"[info] Wrote combo summary to {out_path}")
        return 0

    if args.two_stage_search:
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if args.num_shards > 1 and not (0 <= args.shard_idx < args.num_shards):
            raise ValueError("--shard-idx must be within [0, num-shards)")
        if args.stage1_num_shards < 1:
            raise ValueError("--stage1-num-shards must be >= 1")
        if args.stage1_num_shards > 1 and not (0 <= args.stage1_shard_idx < args.stage1_num_shards):
            raise ValueError("--stage1-shard-idx must be within [0, stage1-num-shards)")
        if args.stage1_num_shards > 1 and not args.stage1_out:
            print("[warn] stage-1 sharding without --stage1-out will yield partial rankings.", file=sys.stderr)
        total_heads = _get_total_heads(model)
        head_combos = _build_head_combos(total_heads)
        layer_combos = build_layer_combos(total_layers, 1, 0)

        head_summaries: List[Dict[str, Any]] = []
        layer_summaries: List[Dict[str, Any]] = []
        stage1_path = None
        if args.stage1_out:
            stage1_path = _resolve_stage1_path(
                args.stage1_out,
                shard_idx=args.stage1_shard_idx,
                num_shards=args.stage1_num_shards,
            )
        if stage1_path and stage1_path.exists():
            print(f"[stage 1] loading summaries from {stage1_path}")
            head_summaries, layer_summaries = _read_stage1_summary(stage1_path)
        else:
            print("[stage 1] evaluating head combos with layers=all")
            for idx, combo in enumerate(head_combos):
                if args.stage1_num_shards > 1 and (idx % args.stage1_num_shards) != args.stage1_shard_idx:
                    continue
                heads_spec = _format_combo_spec(combo)
                summary = _run_combo(
                    heads_spec=heads_spec,
                    layers_spec="all",
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    originals=originals,
                    total_layers=total_layers,
                    device=device,
                    args=args,
                    target_id_norm=None,
                    report_all_probs=False,
                    multi_combo=True,
                    prob_matrix_out_override="",
                )
                head_summaries.append(summary)

            print("[stage 1] evaluating layer combos with heads=all")
            for idx, combo in enumerate(layer_combos):
                if args.stage1_num_shards > 1 and (idx % args.stage1_num_shards) != args.stage1_shard_idx:
                    continue
                layers_spec = _format_combo_spec(combo)
                summary = _run_combo(
                    heads_spec="all",
                    layers_spec=layers_spec,
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    originals=originals,
                    total_layers=total_layers,
                    device=device,
                    args=args,
                    target_id_norm=None,
                    report_all_probs=False,
                    multi_combo=True,
                    prob_matrix_out_override="",
                )
                layer_summaries.append(summary)

            if stage1_path:
                _write_stage1_summary(stage1_path, head_summaries, layer_summaries)
                print(f"[stage 1] wrote summaries to {stage1_path}")

        if args.stage1_out:
            stage1_paths = _resolve_stage1_paths_all(args.stage1_out, num_shards=args.stage1_num_shards)
            combined_heads: List[Dict[str, Any]] = []
            combined_layers: List[Dict[str, Any]] = []
            missing_paths: List[Path] = []
            for path in stage1_paths:
                if not path.exists():
                    missing_paths.append(path)
                    continue
                heads_part, layers_part = _read_stage1_summary(path)
                combined_heads.extend(heads_part)
                combined_layers.extend(layers_part)
            if missing_paths:
                print(f"[warn] Missing stage-1 shard files: {missing_paths}", file=sys.stderr)
            if combined_heads or combined_layers:
                head_summaries = combined_heads
                layer_summaries = combined_layers

        sort_key = lambda row: (row["accuracy"], row["correct_after"])
        head_summaries.sort(key=sort_key, reverse=True)
        layer_summaries.sort(key=sort_key, reverse=True)

        combo_summaries: List[Dict[str, Any]] = []
        shard_counter = 0

        if args.top_uniform > 0:
            combined = head_summaries + layer_summaries
            top_combos = _select_uniform_by_accuracy(
                combined,
                args.top_uniform,
                bins=args.uniform_bins,
            )
            print(
                f"[stage 2] evaluating uniform-accuracy combos from stage 1 "
                f"(n={len(top_combos)}, bins={args.uniform_bins})"
            )
            for row in top_combos:
                if args.num_shards > 1:
                    if shard_counter % args.num_shards != args.shard_idx:
                        shard_counter += 1
                        continue
                summary = _run_combo(
                    heads_spec=row["heads_spec"],
                    layers_spec=row["layers_spec"],
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    originals=originals,
                    total_layers=total_layers,
                    device=device,
                    args=args,
                    target_id_norm=target_id_norm,
                    report_all_probs=bool(args.report_all_probs),
                    multi_combo=True,
                    print_matrix=False,
                    shift_grid=args.shift_grid,
                    tx_values=tx_values,
                    ty_values=ty_values,
                )
                combo_summaries.append(summary)
                shard_counter += 1
        elif args.top_of_all > 0:
            combined = head_summaries + layer_summaries
            combined.sort(key=sort_key, reverse=True)
            top_combos = combined[: args.top_of_all]
            print(f"[stage 2] evaluating top combos from stage 1 (n={len(top_combos)})")
            for row in top_combos:
                if args.num_shards > 1:
                    if shard_counter % args.num_shards != args.shard_idx:
                        shard_counter += 1
                        continue
                summary = _run_combo(
                    heads_spec=row["heads_spec"],
                    layers_spec=row["layers_spec"],
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    originals=originals,
                    total_layers=total_layers,
                    device=device,
                    args=args,
                    target_id_norm=target_id_norm,
                    report_all_probs=bool(args.report_all_probs),
                    multi_combo=True,
                    print_matrix=False,
                    shift_grid=args.shift_grid,
                    tx_values=tx_values,
                    ty_values=ty_values,
                )
                combo_summaries.append(summary)
                shard_counter += 1
        else:
            if args.shift_grid:
                # Keep a fixed number of head/layer combos for tx/ty shifts.
                top_combo_target = 25
                side = max(1, int(math.sqrt(top_combo_target)))
                top_heads_count = side
                top_layers_count = side
            else:
                top_heads_count = max(1, args.top_heads)
                top_layers_count = max(1, args.top_layers)
            top_heads = head_summaries[: top_heads_count]
            top_layers = layer_summaries[: top_layers_count]

            print(f"[stage 2] evaluating top heads x top layers ({len(top_heads)} x {len(top_layers)})")
            for head_row in top_heads:
                for layer_row in top_layers:
                    if args.num_shards > 1:
                        if shard_counter % args.num_shards != args.shard_idx:
                            shard_counter += 1
                            continue
                    summary = _run_combo(
                        heads_spec=head_row["heads_spec"],
                        layers_spec=layer_row["layers_spec"],
                        records=records,
                        model=model,
                        tokenizer=tokenizer,
                        originals=originals,
                        total_layers=total_layers,
                        device=device,
                        args=args,
                        target_id_norm=target_id_norm,
                        report_all_probs=bool(args.report_all_probs),
                        multi_combo=True,
                        print_matrix=False,
                        shift_grid=args.shift_grid,
                        tx_values=tx_values,
                        ty_values=ty_values,
                    )
                    combo_summaries.append(summary)
                    shard_counter += 1

        if args.combo_summary_out:
            out_path = _normalize_csv_path(args.combo_summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["heads_spec", "layers_spec", "correct_after", "total_items", "accuracy"],
                )
                writer.writeheader()
                for row in combo_summaries:
                    writer.writerow(
                        {
                            "heads_spec": row["heads_spec"],
                            "layers_spec": row["layers_spec"],
                            "correct_after": row["correct_after"],
                            "total_items": row["total_items"],
                            "accuracy": row["accuracy"],
                        }
                    )
            print(f"[info] Wrote combo summary to {out_path}")
        return 0

    heads_specs = _split_specs(args.heads_set)
    if not heads_specs:
        heads_specs = [args.heads]
    layers_specs = _split_specs(args.layers_set)
    if not layers_specs:
        layers_specs = [args.layers]

    multi_combo = len(heads_specs) > 1 or len(layers_specs) > 1
    report_all_probs = bool(args.report_all_probs)

    combo_summaries: List[Dict[str, Any]] = []
    for heads_spec in heads_specs:
        for layers_spec in layers_specs:
            summary = _run_combo(
                heads_spec=heads_spec,
                layers_spec=layers_spec,
                records=records,
                model=model,
                tokenizer=tokenizer,
                originals=originals,
                total_layers=total_layers,
                device=device,
                args=args,
                target_id_norm=target_id_norm,
                report_all_probs=report_all_probs,
                multi_combo=multi_combo,
                shift_grid=args.shift_grid,
                tx_values=tx_values,
                ty_values=ty_values,
            )
            combo_summaries.append(summary)
    if args.combo_summary_out:
        out_path = _normalize_csv_path(args.combo_summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["heads_spec", "layers_spec", "correct_after", "total_items", "accuracy"],
            )
            writer.writeheader()
            for row in combo_summaries:
                writer.writerow(
                    {
                        "heads_spec": row["heads_spec"],
                        "layers_spec": row["layers_spec"],
                        "correct_after": row["correct_after"],
                        "total_items": row["total_items"],
                        "accuracy": row["accuracy"],
                    }
                )
        print(f"[info] Wrote combo summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
