''' helper for saved_ranked_pairs.py and combo_accuracy.py'''
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils import make_forward
from utils2 import (
    apply_override,
    canonicalize_answer,
    masks_root,
    normalize_answer,
    parse_heads,
    revert_override,
)


IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root


def prepare_question_inputs(tokenizer, question: str, device: torch.device) -> Dict[str, torch.Tensor]:
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


def resolve_layers(spec: str, total_layers: int) -> Tuple[int, ...]:
    parsed = parse_heads(spec)
    if parsed == (-1,):
        return tuple(range(total_layers))
    resolved = tuple(int(v) for v in parsed)
    for layer_idx in resolved:
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(f"Layer index out of range: {layer_idx} (total={total_layers})")
    return resolved


def resolve_heads(spec: str, total_heads: int) -> Tuple[int, ...]:
    parsed = parse_heads(spec)
    if parsed == (-1,):
        return tuple(range(total_heads))
    resolved = tuple(int(v) for v in parsed)
    for head_idx in resolved:
        if head_idx < 0 or head_idx >= total_heads:
            raise ValueError(f"Head index out of range: {head_idx} (total={total_heads})")
    return resolved


def encode_question_state(
    model,
    question_inputs: Dict[str, torch.Tensor],
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
    with torch.no_grad():
        question_output = model.text_encoder(
            input_ids=question_inputs["input_ids"],
            attention_mask=question_inputs["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_att_mask,
            return_dict=True,
        )
    return question_output.last_hidden_state


def generate_topk_answers(
    model,
    tokenizer,
    question_state: torch.Tensor,
    *,
    top_k: int,
    max_length: int,
    min_length: int,
) -> List[str]:
    if top_k <= 0:
        return []

    question_states = question_state.repeat_interleave(top_k, dim=0)
    question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long, device=question_states.device)
    model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}
    bos_ids = torch.full((1, 1), fill_value=tokenizer.bos_token_id, device=question_states.device)

    with torch.no_grad():
        outputs = model.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=top_k,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=top_k,
            return_dict_in_generate=True,
            output_scores=False,
            **model_kwargs,
        )

    answers: List[str] = []
    seen = set()
    for seq in outputs.sequences:
        raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
        if not raw:
            continue
        canonical = canonicalize_answer(raw)
        key = canonical or normalize_answer(raw) or raw.lower()
        if key in seen:
            continue
        seen.add(key)
        answers.append(canonical if canonical else raw)
        if len(answers) >= top_k:
            break
    return answers


def answer_log_prob(
    model,
    tokenizer,
    question_state: torch.Tensor,
    question_attention_mask: torch.Tensor,
    answer: str,
) -> float:
    cleaned = (answer or "").strip()
    if not cleaned:
        return float("-inf")

    token_ids = tokenizer(
        cleaned,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(question_state.device)
    if token_ids.numel() == 0:
        return float("-inf")

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing BOS/EOS token IDs.")

    bos = torch.tensor([[bos_id]], device=question_state.device)
    eos = torch.tensor([[eos_id]], device=question_state.device)
    target = torch.cat([bos, token_ids, eos], dim=1)
    decoder_in = target[:, :-1]
    labels = target[:, 1:]

    with torch.no_grad():
        output = model.text_decoder(
            input_ids=decoder_in,
            encoder_hidden_states=question_state,
            encoder_attention_mask=question_attention_mask,
            return_dict=True,
        )
        log_probs = torch.log_softmax(output.logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def union_answers(before: Sequence[str], after: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for answer in list(before) + list(after):
        text = (answer or "").strip()
        if not text:
            continue
        canonical = canonicalize_answer(text)
        key = canonical or normalize_answer(text) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(canonical if canonical else text)
    return merged


def probs_from_log_probs(log_probs: Sequence[float]) -> List[float]:
    if not log_probs:
        return []
    values = torch.tensor(log_probs, dtype=torch.float64)
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return [1.0 / len(log_probs)] * len(log_probs)
    floor = values[finite].min() - 100.0
    values = torch.where(finite, values, floor)
    probs = torch.softmax(values, dim=0)
    return [float(v.item()) for v in probs]


def entropy_from_probs(probs: Sequence[float]) -> float:
    if not probs:
        return 0.0
    p = torch.tensor(probs, dtype=torch.float64)
    p = torch.clamp(p, min=1e-12)
    return float((-(p * p.log()).sum()).item())


def kl_divergence_from_probs(
    p_probs: Sequence[float],
    q_probs: Sequence[float],
) -> float:
    if not p_probs or not q_probs or len(p_probs) != len(q_probs):
        return 0.0
    p = torch.tensor(p_probs, dtype=torch.float64)
    q = torch.tensor(q_probs, dtype=torch.float64)
    p = torch.clamp(p, min=1e-12)
    q = torch.clamp(q, min=1e-12)
    p = p / p.sum().clamp_min(1e-12)
    q = q / q.sum().clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum().item())


def margin_from_probs(probs: Sequence[float]) -> float:
    if not probs:
        return 0.0
    sorted_probs = sorted(float(v) for v in probs)[::-1]
    if len(sorted_probs) == 1:
        return sorted_probs[0]
    return sorted_probs[0] - sorted_probs[1]


def margin_from_scores(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    sorted_scores = sorted(float(v) for v in scores)[::-1]
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    return sorted_scores[0] - sorted_scores[1]


def load_existing_cache(
    path: Path,
    *,
    target_layers: Sequence[int],
    target_heads: Sequence[int],
    top_k: int,
    max_length: int,
    min_length: int,
    require_kl_matrix: bool = False,
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
            if "delta_entropy_matrix" not in row or "delta_margin_matrix" not in row:
                continue
            if require_kl_matrix and "delta_kl_matrix" not in row:
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
            cache[str(rec_id)] = row
    return cache


def path_with_topk(path_value: str, top_k: int) -> Path:
    if "{k}" in path_value:
        return Path(path_value.format(k=int(top_k)))
    base = Path(path_value)
    suffix = base.suffix
    stem = base.stem
    if stem.endswith(f"_k{int(top_k)}"):
        return base
    return base.with_name(f"{stem}_k{int(top_k)}{suffix}")


def build_ranked_pairs_from_matrices(
    delta_entropy_matrix: Sequence[Sequence[Optional[float]]],
    delta_margin_matrix: Sequence[Sequence[Optional[float]]],
    *,
    delta_kl_matrix: Optional[Sequence[Sequence[Optional[float]]]] = None,
    layers: Sequence[int],
    heads: Sequence[int],
    b: float,
    c: float,
    d: float = 0.0,
) -> Tuple[List[List[Optional[Dict[str, float]]]], List[Dict[str, float]], int]:
    pair_matrix: List[List[Optional[Dict[str, float]]]] = []
    ranked_pairs: List[Dict[str, float]] = []
    valid_cells = 0
    for i, layer_idx in enumerate(layers):
        row_cells: List[Optional[Dict[str, float]]] = []
        for j, head_idx in enumerate(heads):
            d_h = None
            d_m = None
            d_kl = None
            if i < len(delta_entropy_matrix) and j < len(delta_entropy_matrix[i]):
                d_h = delta_entropy_matrix[i][j]
            if i < len(delta_margin_matrix) and j < len(delta_margin_matrix[i]):
                d_m = delta_margin_matrix[i][j]
            if (
                delta_kl_matrix is not None
                and i < len(delta_kl_matrix)
                and j < len(delta_kl_matrix[i])
            ):
                d_kl = delta_kl_matrix[i][j]
            if d_h is None or d_m is None:
                row_cells.append(None)
                continue
            d_h_f = float(d_h)
            d_m_f = float(d_m)
            d_kl_f = float(d_kl) if d_kl is not None else 0.0
            score = float(b) * d_m_f + float(c) * d_h_f + float(d) * d_kl_f
            cell = {"delta_H": d_h_f, "delta_m": d_m_f, "delta_KL": d_kl_f, "score": score}
            row_cells.append(cell)
            ranked_pairs.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "delta_H": d_h_f,
                    "delta_m": d_m_f,
                    "delta_KL": d_kl_f,
                    "score": score,
                }
            )
            valid_cells += 1
        pair_matrix.append(row_cells)
    ranked_pairs.sort(
        key=lambda item: (item["score"], item["delta_m"], item["delta_H"], item.get("delta_KL", 0.0)),
        reverse=True,
    )
    return pair_matrix, ranked_pairs, valid_cells


def print_top_pairs(record_id: str, ranked_pairs: Sequence[Dict[str, float]], top_n: int) -> None:
    if top_n <= 0:
        return
    limit = min(int(top_n), len(ranked_pairs))
    print(f"[top_pairs] id={record_id} showing {limit} of {len(ranked_pairs)}")
    for rank, item in enumerate(ranked_pairs[:limit], 1):
        print(
            f"  {rank:02d}. (L{int(item['layer'])},H{int(item['head'])}) "
            f"score={float(item['score']):+.6f} "
            f"Dm={float(item['delta_m']):+.6f} "
            f"DH={float(item['delta_H']):+.6f} "
            f"DKL={float(item.get('delta_KL', 0.0)):+.6f}"
        )


def _sample_combo_pair_sets(
    ranked_pairs: Sequence[Dict[str, float]],
    *,
    combo_count: int,
    top_pick_min: int,
    top_pick_max: int,
    next_window: int,
    next_pick_min: int,
    next_pick_max: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if combo_count <= 0:
        return []

    top_pool_size = min(max(top_pick_max, 0), len(ranked_pairs))
    top_pool = list(ranked_pairs[:top_pool_size])
    next_start = top_pool_size
    next_pool = list(ranked_pairs[next_start : next_start + max(next_window, 0)])

    top_hi = len(top_pool)
    next_hi = len(next_pool)
    if top_hi <= 0 or next_hi <= 0:
        return []

    top_lo = min(max(top_pick_min, 0), top_hi)
    next_lo = min(max(next_pick_min, 0), next_hi)
    if top_lo <= 0 or next_lo <= 0:
        return []

    next_hi = min(next_hi, max(next_pick_max, 0))
    top_hi = min(top_hi, max(top_pick_max, 0))
    if top_hi < top_lo or next_hi < next_lo:
        return []

    combos: List[Dict[str, Any]] = []
    seen = set()
    max_attempts = max(200, combo_count * 200)
    attempts = 0
    while len(combos) < combo_count and attempts < max_attempts:
        attempts += 1
        n_top = rng.randint(top_lo, top_hi)
        n_next = rng.randint(next_lo, next_hi)
        top_sel = rng.sample(top_pool, n_top)
        next_sel = rng.sample(next_pool, n_next)
        pairs = {(int(item["layer"]), int(item["head"])) for item in top_sel}
        pairs.update((int(item["layer"]), int(item["head"])) for item in next_sel)
        combo_key = tuple(sorted(pairs))
        if not combo_key or combo_key in seen:
            continue
        seen.add(combo_key)
        combos.append(
            {
                "top_count": int(n_top),
                "next_count": int(n_next),
                "pair_count": int(len(combo_key)),
                "pairs": [[int(l), int(h)] for (l, h) in combo_key],
            }
        )
    return combos


def _encode_state_with_pair_combo(
    *,
    model,
    question_inputs: Dict[str, torch.Tensor],
    image_embeds: torch.Tensor,
    combo_pairs: Sequence[Tuple[int, int]],
    override_rows: Dict[int, torch.Tensor],
    originals,
    alpha: float,
) -> torch.Tensor:
    layer_to_heads: Dict[int, set] = defaultdict(set)
    for layer_idx, head_idx in combo_pairs:
        layer_to_heads[int(layer_idx)].add(int(head_idx))

    applied_layers: List[int] = []
    try:
        for layer_idx in sorted(layer_to_heads.keys()):
            heads = tuple(sorted(int(h) for h in layer_to_heads[layer_idx]))
            if not heads:
                continue
            heads_arg = heads[0] if len(heads) == 1 else heads
            new_forward = make_forward(heads_arg, override_rows, alpha)
            apply_override(new_forward, (int(layer_idx),), originals)
            applied_layers.append(int(layer_idx))
        state = encode_question_state(model, question_inputs, image_embeds)
    finally:
        if applied_layers:
            revert_override(tuple(applied_layers), originals)
    return state


def evaluate_combo_runs_with_consensus(
    *,
    model,
    tokenizer,
    question_inputs: Dict[str, torch.Tensor],
    image_embeds: torch.Tensor,
    override_rows: Dict[int, torch.Tensor],
    originals,
    ranked_pairs: Sequence[Dict[str, float]],
    gold_norm: str,
    combo_count: int,
    combo_top_min: int,
    combo_top_max: int,
    combo_next_window: int,
    combo_next_min: int,
    combo_next_max: int,
    combo_seed: int,
    top_k: int,
    max_length: int,
    min_length: int,
    alpha: float,
    consensus_weight: float,
    mv_weight: float,
    mv_var_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(combo_seed))
    combo_specs = _sample_combo_pair_sets(
        ranked_pairs,
        combo_count=int(combo_count),
        top_pick_min=int(combo_top_min),
        top_pick_max=int(combo_top_max),
        next_window=int(combo_next_window),
        next_pick_min=int(combo_next_min),
        next_pick_max=int(combo_next_max),
        rng=rng,
    )
    if not combo_specs:
        return [], {
            "union_answers": [],
            "answer_stats": [],
            "selected_answer": "",
            "selected_answer_norm": "",
            "selected_hit": 0,
            "num_runs": 0,
            "logprob_cache_size": 0,
            "score_formula": "consensus_w*consensus_freq + mv_w*(mean_prob - mv_var_w*var_prob)",
            "score_weights": {
                "consensus_w": float(consensus_weight),
                "mv_w": float(mv_weight),
                "mv_var_w": float(mv_var_weight),
            },
        }

    run_states: List[torch.Tensor] = []
    run_topk_answers: List[List[str]] = []
    combo_results: List[Dict[str, Any]] = []
    for combo_idx, combo in enumerate(combo_specs, 1):
        combo_pairs = [(int(p[0]), int(p[1])) for p in combo.get("pairs", [])]
        state = _encode_state_with_pair_combo(
            model=model,
            question_inputs=question_inputs,
            image_embeds=image_embeds,
            combo_pairs=combo_pairs,
            override_rows=override_rows,
            originals=originals,
            alpha=float(alpha),
        )
        topk_after = generate_topk_answers(
            model,
            tokenizer,
            state,
            top_k=int(top_k),
            max_length=int(max_length),
            min_length=int(min_length),
        )
        pred_after = topk_after[0] if topk_after else ""
        pred_after_norm = normalize_answer(pred_after)
        run_states.append(state)
        run_topk_answers.append(topk_after)
        combo_results.append(
            {
                "combo_index": int(combo_idx),
                "top_count": int(combo.get("top_count", 0)),
                "next_count": int(combo.get("next_count", 0)),
                "pair_count": int(combo.get("pair_count", len(combo_pairs))),
                "pairs": combo.get("pairs", []),
                "topk_after": topk_after,
                "pred_after": pred_after,
                "after_hit": int(pred_after_norm == gold_norm),
            }
        )

    merged_answers: List[str] = []
    for answers in run_topk_answers:
        merged_answers = union_answers(merged_answers, answers)
    if not merged_answers:
        return combo_results, {
            "union_answers": [],
            "answer_stats": [],
            "selected_answer": "",
            "selected_answer_norm": "",
            "selected_hit": 0,
            "num_runs": len(combo_results),
            "logprob_cache_size": 0,
            "score_formula": "consensus_w*consensus_freq + mv_w*(mean_prob - mv_var_w*var_prob)",
            "score_weights": {
                "consensus_w": float(consensus_weight),
                "mv_w": float(mv_weight),
                "mv_var_w": float(mv_var_weight),
            },
        }

    tf_logprob_cache: Dict[Tuple[int, str], float] = {}
    consensus_count = defaultdict(int)
    prob_history: Dict[str, List[float]] = {answer: [] for answer in merged_answers}
    for run_idx, state in enumerate(run_states):
        run_logps: List[float] = []
        for answer in merged_answers:
            key = (int(run_idx), answer)
            if key not in tf_logprob_cache:
                tf_logprob_cache[key] = answer_log_prob(
                    model,
                    tokenizer,
                    state,
                    question_inputs["attention_mask"],
                    answer,
                )
            run_logps.append(float(tf_logprob_cache[key]))
        run_probs = probs_from_log_probs(run_logps)
        if run_probs:
            best_idx = int(np.argmax(np.array(run_probs, dtype=np.float64)))
            best_answer = merged_answers[best_idx]
            consensus_count[best_answer] += 1
        for answer, prob in zip(merged_answers, run_probs):
            prob_history[answer].append(float(prob))

    num_runs = max(1, len(run_states))
    answer_stats: List[Dict[str, Any]] = []
    for answer in merged_answers:
        values = prob_history.get(answer, [])
        if values:
            arr = np.array(values, dtype=np.float64)
            mean_prob = float(arr.mean())
            var_prob = float(arr.var())
        else:
            mean_prob = 0.0
            var_prob = 0.0
        consensus_freq = float(consensus_count.get(answer, 0)) / float(num_runs)
        mv_score = float(mean_prob) - float(mv_var_weight) * float(var_prob)
        combined_score = float(consensus_weight) * consensus_freq + float(mv_weight) * mv_score
        answer_stats.append(
            {
                "answer": answer,
                "consensus_count": int(consensus_count.get(answer, 0)),
                "consensus_freq": float(consensus_freq),
                "mean_prob": float(mean_prob),
                "var_prob": float(var_prob),
                "mean_variance_score": float(mv_score),
                "combined_score": float(combined_score),
            }
        )
    answer_stats.sort(
        key=lambda x: (
            float(x["combined_score"]),
            float(x["consensus_freq"]),
            float(x["mean_variance_score"]),
        ),
        reverse=True,
    )

    selected_answer = str(answer_stats[0]["answer"]) if answer_stats else ""
    selected_answer_norm = normalize_answer(selected_answer)
    selected_hit = int(selected_answer_norm == gold_norm)
    consensus_payload: Dict[str, Any] = {
        "union_answers": merged_answers,
        "answer_stats": answer_stats,
        "selected_answer": selected_answer,
        "selected_answer_norm": selected_answer_norm,
        "selected_hit": int(selected_hit),
        "num_runs": int(len(run_states)),
        "logprob_cache_size": int(len(tf_logprob_cache)),
        "score_formula": "consensus_w*consensus_freq + mv_w*(mean_prob - mv_var_w*var_prob)",
        "score_weights": {
            "consensus_w": float(consensus_weight),
            "mv_w": float(mv_weight),
            "mv_var_w": float(mv_var_weight),
        },
    }
    return combo_results, consensus_payload

