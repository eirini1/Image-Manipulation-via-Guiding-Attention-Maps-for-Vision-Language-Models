"""Override encoder self-attention on selected layers/heads for dataset records."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import types
import re

import torch
from torch import nn
import math
import numpy as np

from models.blip_vqa import blip_vqa
from utils import load_demo_image, make_forward, soften_mask
from utils2 import (
    TOKEN_STOP,
    data_path,
    image_root,
    iter_jsonl,
    normalize_answer,
    parse_heads,
    find_subsequence,
    select_override_indices,
    ensure_mask,
    masks_root,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root

STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "does",
    "did", "for", "from", "how", "in", "is", "it", "many", "of", "on",
    "state", "states", "that", "the", "their", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with"
}

def _sanitize_name(s: str) -> str:
    s = "" if s is None else str(s)
    cleaned = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in s)
    return cleaned[:80]


def _load_mask_from_dir(mask_dir: Path, image_stem: str, prompt: str, rec_id: Optional[str]) -> Optional[np.ndarray]:
    sanitized = _sanitize_name(prompt)
    patterns = []
    if rec_id:
        patterns.append(f"{rec_id}_{image_stem}_{sanitized}_*.npy")
    patterns.append(f"*_{image_stem}_{sanitized}_*.npy")

    for pat in patterns:
        candidates = sorted(mask_dir.glob(pat))
        if candidates:
            path = candidates[-1]
            try:
                arr = np.load(path)
                if arr.ndim > 2:
                    arr = arr.squeeze()
                return arr.astype(np.float32)
            except Exception:
                continue
    return None


def guess_focus_words(question: str) -> List[str]:
    clean = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = clean.split()
    content = [w for w in words if w not in STOP_WORDS]
    if not content:
        content = words
    if len(content) > 4:
        content = content[:4]
    return [w for w in content if w]

def _select_focus_indices(tokens: Sequence[str], focus_words: Sequence[str], tokenizer) -> List[int]:
    indices: List[int] = []
    for word in focus_words:
        subs = tokenizer.tokenize(word)
        if not subs:
            continue
        starts = find_subsequence(tokens, subs)
        for start in starts:
            for offset in range(len(subs)):
                idx = start + offset
                if 0 <= idx < len(tokens) and tokens[idx] not in TOKEN_STOP:
                    indices.append(idx)
    deduped = sorted(set(indices))
    return deduped


def _build_focus_row(seq_len: int, focus_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    if not focus_indices:
        row = torch.full((seq_len,), 1.0 / max(seq_len, 1), device=device)
        return row
    row = torch.zeros(seq_len, device=device)
    for idx in focus_indices:
        if 0 <= idx < seq_len:
            row[idx] = 1.0
    row = row / row.sum().clamp_min(1e-6)
    return row


def _resolve_head_indices(heads, total_heads: int, device: torch.device) -> Optional[torch.Tensor]:
    if heads is None:
        return None
    if isinstance(heads, int):
        if heads < 0:
            return None
        if heads >= total_heads:
            return torch.tensor([], device=device, dtype=torch.long)
        return torch.tensor([heads], device=device, dtype=torch.long)
    if not heads or heads == (-1,):
        return None
    valid = [h for h in heads if 0 <= h < total_heads]
    if not valid:
        return torch.tensor([], device=device, dtype=torch.long)
    return torch.tensor(valid, device=device, dtype=torch.long)


def _resolve_lambda_value(lambd, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(lambd, torch.Tensor):
        return torch.sigmoid(lambd).to(device=device, dtype=dtype)
    return torch.tensor(float(lambd), device=device, dtype=dtype)


def _resolve_gate_value(head_gates, layer_gate, head_indices, total_heads: int, device: torch.device, dtype: torch.dtype):
    gate = None
    if head_gates is not None:
        head_vec = torch.sigmoid(head_gates).to(device=device, dtype=dtype).view(-1)
        if head_indices is None:
            if head_vec.numel() != total_heads:
                raise RuntimeError(f"Head gates length {head_vec.numel()} != total heads {total_heads}")
        else:
            if head_indices.numel():
                head_vec = head_vec.index_select(0, head_indices)
        gate = head_vec
    if layer_gate is not None:
        layer_val = torch.sigmoid(layer_gate).to(device=device, dtype=dtype)
        gate = gate * layer_val if gate is not None else layer_val
    return gate


def make_self_forward(
    heads,
    override_cols: Dict[int, torch.Tensor],
    mode: str,
    beta: float,
    gamma: float,
    lambd: float,
    head_gates=None,
    layer_gate=None,
):
    def new_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
    ):
        is_cross_attention = encoder_hidden_states is not None
        src = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask_ = encoder_attention_mask if is_cross_attention else attention_mask

        mixed_query_layer = self.query(hidden_states)
        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask_ is not None:
            attention_scores = attention_scores + attention_mask_

        if override_cols and mode == "pre":
            q_len = attention_scores.size(-2)
            head_indices = _resolve_head_indices(heads, attention_scores.size(1), attention_scores.device)
            for idx, col in override_cols.items():
                col = col.to(attention_scores.device, dtype=attention_scores.dtype)
                if col.numel() != q_len:
                    raise RuntimeError(f"Column length {col.numel()} != query length {q_len}")
                bias = beta * col - gamma * (1.0 - col)
                gate = _resolve_gate_value(
                    head_gates,
                    layer_gate,
                    head_indices,
                    attention_scores.size(1),
                    attention_scores.device,
                    attention_scores.dtype,
                )
                if head_indices is None:
                    if gate is None:
                        attention_scores[:, :, :, idx] = attention_scores[:, :, :, idx] + bias.view(1, 1, q_len)
                    elif gate.dim() == 0:
                        attention_scores[:, :, :, idx] = attention_scores[:, :, :, idx] + bias.view(1, 1, q_len) * gate
                    else:
                        attention_scores[:, :, :, idx] = (
                            attention_scores[:, :, :, idx] + bias.view(1, 1, q_len) * gate.view(1, -1, 1)
                        )
                elif head_indices.numel():
                    if gate is None:
                        attention_scores[:, head_indices, :, idx] = (
                            attention_scores[:, head_indices, :, idx] + bias.view(1, 1, q_len)
                        )
                    elif gate.dim() == 0:
                        attention_scores[:, head_indices, :, idx] = (
                            attention_scores[:, head_indices, :, idx] + bias.view(1, 1, q_len) * gate
                        )
                    else:
                        attention_scores[:, head_indices, :, idx] = (
                            attention_scores[:, head_indices, :, idx] + bias.view(1, 1, q_len) * gate.view(1, -1, 1)
                        )

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if override_cols and mode == "post":
            attention_probs = attention_probs.clone()
            q_len = attention_probs.size(-2)
            head_indices = _resolve_head_indices(heads, attention_probs.size(1), attention_probs.device)
            for idx, col in override_cols.items():
                col = col.to(attention_probs.device, dtype=attention_probs.dtype)
                if col.numel() != q_len:
                    raise RuntimeError(f"Column length {col.numel()} != query length {q_len}")
                target = col.view(1, 1, q_len)
                gate = _resolve_gate_value(
                    head_gates,
                    layer_gate,
                    head_indices,
                    attention_probs.size(1),
                    attention_probs.device,
                    attention_probs.dtype,
                )
                lambd_val = _resolve_lambda_value(lambd, attention_probs.device, attention_probs.dtype)
                if gate is None:
                    eff_lambda = lambd_val
                else:
                    eff_lambda = lambd_val * gate
                if head_indices is None:
                    current = attention_probs[:, :, :, idx]
                    if isinstance(eff_lambda, torch.Tensor) and eff_lambda.dim() == 1:
                        eff = eff_lambda.view(1, -1, 1)
                        attention_probs[:, :, :, idx] = (1 - eff) * current + eff * target
                    else:
                        attention_probs[:, :, :, idx] = (1 - eff_lambda) * current + eff_lambda * target
                elif head_indices.numel():
                    current = attention_probs[:, head_indices, :, idx]
                    if isinstance(eff_lambda, torch.Tensor) and eff_lambda.dim() == 1:
                        eff = eff_lambda.view(1, -1, 1)
                        attention_probs[:, head_indices, :, idx] = (1 - eff) * current + eff * target
                    else:
                        attention_probs[:, head_indices, :, idx] = (1 - eff_lambda) * current + eff_lambda * target
            attention_probs = attention_probs / attention_probs.sum(-1, keepdim=True).clamp_min(1e-6)

        attention_probs.requires_grad_(True)

        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs

    return new_forward


def _apply_override(new_forward, layers: Tuple[int, ...], originals):
    for layer_idx in layers:
        module, _, _ = originals[layer_idx]
        module.forward = types.MethodType(new_forward, module)


def _revert_override(layers: Tuple[int, ...], originals):
    for layer_idx in layers:
        module, orig_forward, _ = originals[layer_idx]
        module.forward = orig_forward


def _parse_record_ids(raw: str) -> Tuple[str, ...]:
    if not raw:
        return tuple()
    items = [part.strip() for part in raw.split(",")]
    return tuple(item for item in items if item)


def _format_token_for_table(tok: str, max_len: int = 12) -> str:
    tok = tok.replace("\n", " ").replace("\r", " ")
    if len(tok) > max_len:
        return tok[: max_len - 3] + "..."
    return tok


def _format_attention_matrix(mat: torch.Tensor, tokens: Sequence[str], precision: int = 4) -> str:
    mat_np = mat.detach().float().cpu().numpy()
    seq_len = mat_np.shape[0]
    tok_labels = [_format_token_for_table(t) for t in tokens]
    width = precision + 4
    fmt = f"{{:>{width}.{precision}f}}"
    header = (" " * 16) + " ".join(f"{i:>{width}}" for i in range(seq_len))
    lines = [header]
    for i in range(seq_len):
        row_vals = " ".join(fmt.format(mat_np[i, j]) for j in range(seq_len))
        lines.append(f"{i:>2} {tok_labels[i]:<12} {row_vals}")
    return "\n".join(lines)


def _select_heads(heads_spec, total_heads: int) -> List[int]:
    if heads_spec == (-1,) or not heads_spec:
        return list(range(total_heads))
    return [h for h in heads_spec if 0 <= h < total_heads]

def _prepare_question_inputs(
    question: str,
    tokenizer,
    *,
    cls_only: bool,
    cls_plus: bool,
    all_queries: bool,
):
    question_inputs = tokenizer(
        question,
        padding="longest",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    )
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    question_inputs["input_ids"][:, 0] = tokenizer.enc_token_id
    question_inputs.pop("token_type_ids", None)

    tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
    focus_words = guess_focus_words(question)
    key_indices = _select_focus_indices(tokens, focus_words, tokenizer)
    if not key_indices:
        key_indices = select_override_indices(tokens, focus_words, tokenizer)
    override_indices = key_indices or [0]

    if cls_only:
        query_indices = [0]
    elif cls_plus:
        query_indices = sorted(set(key_indices + [0]))
    elif all_queries:
        query_indices = list(range(len(tokens)))
    else:
        query_indices = key_indices
    if not query_indices:
        query_indices = [0]

    focus_col = _build_focus_row(len(tokens), query_indices, device=device)
    override_cols = {i: focus_col for i in override_indices}
    return question_inputs, tokens, override_indices, override_cols, key_indices, query_indices


def _build_cross_override_grids(
    entry: Dict[str, object],
    image_rel: Path,
    question: str,
    query_indices: Sequence[int],
    mask_dir: Path,
    mask_cache: Dict[Tuple[str, str], np.ndarray],
) -> Dict[int, torch.Tensor]:
    prompt = entry.get("prompt") or question
    cache_key = (image_rel.as_posix(), str(prompt))
    mask_array = mask_cache.get(cache_key)
    if mask_array is None:
        mask_array = _load_mask_from_dir(mask_dir, image_rel.stem, str(prompt), entry.get("id"))
        if mask_array is not None:
            mask_cache[cache_key] = mask_array

    gh = gw = IMAGE_SIZE // PATCH_SIZE
    mask_array = ensure_mask(mask_array, gh, gw, stem=image_rel.stem)
    mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
    mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
    mask_small = torch.nn.functional.interpolate(
        mask_tensor,
        size=(gh, gw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0).clamp_(0.0, 1.0)
    mask_soft = soften_mask(mask_small, ksize=5, iters=2)
    return {i: mask_soft for i in query_indices}


def _get_total_heads(encoder_layers) -> int:
    if not encoder_layers:
        return 0
    sa = encoder_layers[0].attention.self
    for attr in ("num_attention_heads", "num_heads"):
        val = getattr(sa, attr, None)
        if isinstance(val, int):
            return val
    head_size = getattr(sa, "attention_head_size", None)
    all_head_size = getattr(sa, "all_head_size", None)
    if isinstance(head_size, int) and isinstance(all_head_size, int) and head_size > 0:
        return all_head_size // head_size
    return 12


def _init_logit(value: float, device: torch.device) -> torch.Tensor:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(clipped, device=device))


def _evaluate_records(
    records: Sequence[Dict[str, object]],
    model,
    tokenizer,
    target_layers: Tuple[int, ...],
    heads_arg,
    originals,
    heads_for_print,
    *,
    cls_only: bool,
    cls_plus: bool,
    all_queries: bool,
    mode: str,
    beta: float,
    gamma: float,
    lambd: float,
    cross_attn: bool,
    cross_target_layers: Tuple[int, ...],
    cross_heads_arg,
    cross_originals,
    mask_dir: Path,
    mask_cache: Dict[Tuple[str, str], np.ndarray],
    head_gates=None,
    layer_gates=None,
    record_ids: Sequence[str],
    print_record_ids: Sequence[str],
    verbose_records: bool,
    show_progress: bool,
    collect_ids: bool,
) -> Dict[str, object]:
    total_trials = 0
    total_hits = 0
    retain_trials = 0
    retain_hits = 0
    correct_ids: List[str] = []
    retained_correct_ids: List[str] = []
    start_time = time.time()
    PROG_STEP = 10

    for idx, entry in enumerate(records, 1):
        image_value = entry.get("image")
        if image_value is None or entry.get("question") is None or entry.get("answer") is None:
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
        if record_ids and rec_id_str not in record_ids:
            continue
        print_this = bool(print_record_ids) and rec_id_str in print_record_ids

        question = entry["question"]
        gold = entry["answer"]
        gold_norm = normalize_answer(gold)
        is_yes_labeled = str(entry.get("correct", "")).strip().lower() == "yes"

        image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)

        question_inputs, tokens, override_indices, override_cols, _, query_indices = _prepare_question_inputs(
            question,
            tokenizer,
            cls_only=cls_only,
            cls_plus=cls_plus,
            all_queries=all_queries,
        )
        new_cross_forward = None
        if cross_attn:
            cross_override_grids = _build_cross_override_grids(
                entry,
                image_rel,
                question,
                query_indices,
                mask_dir,
                mask_cache,
            )
            new_cross_forward = make_forward(cross_heads_arg, cross_override_grids)

        def run_pred(use_override: bool) -> str:
            if use_override:
                if new_cross_forward is not None:
                    _apply_override(new_cross_forward, cross_target_layers, cross_originals)
                for layer_idx in target_layers:
                    layer_gate = layer_gates[layer_idx] if layer_gates is not None else None
                    new_forward = make_self_forward(
                        heads_arg,
                        override_cols,
                        mode,
                        beta,
                        gamma,
                        lambd,
                        head_gates=head_gates,
                        layer_gate=layer_gate,
                    )
                    _apply_override(new_forward, (layer_idx,), originals)
            try:
                with torch.no_grad():
                    return model(image, question, train=False, inference="generate")[0]
            finally:
                if use_override:
                    _revert_override(target_layers, originals)
                    if new_cross_forward is not None:
                        _revert_override(cross_target_layers, cross_originals)

        pred = run_pred(True)

        pred_norm = normalize_answer(pred)
        hit = int(pred_norm == gold_norm)
        if verbose_records:
            print(f"id:{rec_id_str} pred_norm:{pred_norm}")
        total_trials += 1
        if hit:
            total_hits += 1
            if collect_ids:
                correct_ids.append(rec_id_str)
        if is_yes_labeled:
            retain_trials += 1
            if hit:
                retain_hits += 1
                if collect_ids:
                    retained_correct_ids.append(rec_id_str)

        if print_this:
            image_embeds = model.visual_encoder(image)
            image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            def run_text_encoder(use_override: bool):
                if use_override:
                    if new_cross_forward is not None:
                        _apply_override(new_cross_forward, cross_target_layers, cross_originals)
                    for layer_idx in target_layers:
                        layer_gate = layer_gates[layer_idx] if layer_gates is not None else None
                        new_forward = make_self_forward(
                            heads_arg,
                            override_cols,
                            mode,
                            beta,
                            gamma,
                            lambd,
                            head_gates=head_gates,
                            layer_gate=layer_gate,
                        )
                        _apply_override(new_forward, (layer_idx,), originals)
                try:
                    with torch.no_grad():
                        return model.text_encoder(
                            input_ids=question_inputs["input_ids"],
                            attention_mask=question_inputs["attention_mask"],
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_att_mask,
                            output_attentions=True,
                            return_dict=True,
                        )
                finally:
                    if use_override:
                        _revert_override(target_layers, originals)
                        if new_cross_forward is not None:
                            _revert_override(cross_target_layers, cross_originals)

            enc0 = run_text_encoder(False)
            enc1 = run_text_encoder(True)
            attn0 = enc0.attentions or []
            attn1 = enc1.attentions or []
            token_line = " | ".join(
                f"{i}:{_format_token_for_table(tok)}" for i, tok in enumerate(tokens)
            )
            print(f"################ Record {idx} ################")
            print(f"id: {rec_id_str}")
            print(f"image: {image_rel}")
            print(f"question: {question}")
            print(f"tokens: {token_line}")
            print(f"override_indices: {','.join(str(i) for i in override_indices)}")
            if not attn0 or not attn1:
                print("No encoder self-attention returned.")
            else:
                for layer_idx in target_layers:
                    if layer_idx >= len(attn0) or layer_idx >= len(attn1):
                        print(f"enc self layer {layer_idx}: not available")
                        continue
                    base = attn0[layer_idx][0]
                    over = attn1[layer_idx][0]
                    head_indices = _select_heads(heads_for_print, base.shape[0])
                    if not head_indices:
                        print(f"enc self layer {layer_idx}: no heads selected")
                        continue
                    base_sel = base[head_indices]
                    over_sel = over[head_indices]
                    mean_base = base_sel.mean(dim=0)
                    mean_over = over_sel.mean(dim=0)
                    print(f"enc self layer {layer_idx} n_heads={len(head_indices)}")
                    print("before:")
                    print(_format_attention_matrix(mean_base, tokens))
                    print("after:")
                    print(_format_attention_matrix(mean_over, tokens))
                    print("")
            print("##############################################")

        if show_progress and ((total_trials == 1) or (total_trials % PROG_STEP == 0)):
            elapsed = time.time() - start_time
            rate = total_trials / elapsed if elapsed > 0 else 0.0
            acc = (total_hits / total_trials * 100.0) if total_trials else 0.0
            print(
                f"[progress] {total_trials} | acc: {total_hits}/{total_trials} ({acc:.2f}%)"
                + f" | {rate:.2f} rec/s",
                flush=True,
            )

    accuracy = (total_hits / total_trials * 100.0) if total_trials else 0.0
    retention = (retain_hits / retain_trials * 100.0) if retain_trials else 0.0
    result: Dict[str, object] = {
        "total_trials": total_trials,
        "total_hits": total_hits,
        "retain_trials": retain_trials,
        "retain_hits": retain_hits,
        "accuracy": accuracy,
        "retention_accuracy": retention,
    }
    if collect_ids:
        result["correct_ids"] = correct_ids
        result["retained_correct_ids"] = retained_correct_ids
    return result


def _optimize_overrides(
    records: Sequence[Dict[str, object]],
    model,
    tokenizer,
    target_layers: Tuple[int, ...],
    heads_arg,
    originals,
    *,
    cls_only: bool,
    cls_plus: bool,
    all_queries: bool,
    mode: str,
    beta_init: float,
    gamma_init: float,
    lambd_init: float,
    optimize_layers: bool,
    optimize_heads: bool,
    optimize_beta: bool,
    optimize_gamma: bool,
    optimize_lambda: bool,
    total_layers: int,
    total_heads: int,
    cross_attn: bool,
    cross_target_layers: Tuple[int, ...],
    cross_heads_arg,
    cross_originals,
    mask_dir: Path,
    mask_cache: Dict[Tuple[str, str], np.ndarray],
    record_ids: Sequence[str],
    epochs: int,
    steps_per_record: int,
    lr: float,
) -> Dict[str, object]:
    for p in model.parameters():
        p.requires_grad = False

    layer_gates = torch.nn.Parameter(torch.zeros(total_layers, device=device)) if optimize_layers else None
    head_gates = torch.nn.Parameter(torch.zeros(total_heads, device=device)) if optimize_heads else None
    beta_param = torch.nn.Parameter(torch.tensor(float(beta_init), device=device)) if optimize_beta else None
    gamma_param = torch.nn.Parameter(torch.tensor(float(gamma_init), device=device)) if optimize_gamma else None
    lambd_param = torch.nn.Parameter(_init_logit(lambd_init, device)) if optimize_lambda else None

    params = [p for p in (layer_gates, head_gates, beta_param, gamma_param, lambd_param) if p is not None]
    if not params:
        raise RuntimeError("No optimization parameters selected.")

    opt = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        for idx, entry in enumerate(records, 1):
            image_value = entry.get("image")
            if image_value is None or entry.get("question") is None or entry.get("answer") is None:
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
            if record_ids and rec_id_str not in record_ids:
                continue

            question = entry["question"]
            gold = entry["answer"]
            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)

            _, _, _, override_cols, _, query_indices = _prepare_question_inputs(
                question,
                tokenizer,
                cls_only=cls_only,
                cls_plus=cls_plus,
                all_queries=all_queries,
            )
            new_cross_forward = None
            if cross_attn:
                cross_override_grids = _build_cross_override_grids(
                    entry,
                    image_rel,
                    question,
                    query_indices,
                    mask_dir,
                    mask_cache,
                )
                new_cross_forward = make_forward(cross_heads_arg, cross_override_grids)

            if (epoch == 0 and idx == 1) or (idx % 10 == 0):
                print(f"[opt] epoch {epoch + 1}/{epochs} record {idx}", flush=True)

            for _ in range(steps_per_record):
                if new_cross_forward is not None:
                    _apply_override(new_cross_forward, cross_target_layers, cross_originals)
                for layer_idx in target_layers:
                    layer_gate = layer_gates[layer_idx] if layer_gates is not None else None
                    beta_val = beta_param if beta_param is not None else beta_init
                    gamma_val = gamma_param if gamma_param is not None else gamma_init
                    lambd_val = lambd_param if lambd_param is not None else lambd_init
                    new_forward = make_self_forward(
                        heads_arg,
                        override_cols,
                        mode,
                        beta_val,
                        gamma_val,
                        lambd_val,
                        head_gates=head_gates,
                        layer_gate=layer_gate,
                    )
                    _apply_override(new_forward, (layer_idx,), originals)
                try:
                    weights = torch.tensor([1.0], device=device)
                    loss = model(image, [question], answer=[gold], train=True, n=[1], weights=weights)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                finally:
                    _revert_override(target_layers, originals)
                    if new_cross_forward is not None:
                        _revert_override(cross_target_layers, cross_originals)

    beta_value = beta_param.detach().item() if beta_param is not None else float(beta_init)
    gamma_value = gamma_param.detach().item() if gamma_param is not None else float(gamma_init)
    lambd_value = torch.sigmoid(lambd_param).detach().item() if lambd_param is not None else float(lambd_init)
    layer_gate_vals = torch.sigmoid(layer_gates).detach().cpu().tolist() if layer_gates is not None else None
    head_gate_vals = torch.sigmoid(head_gates).detach().cpu().tolist() if head_gates is not None else None

    return {
        "beta": beta_value,
        "gamma": gamma_value,
        "lambda": lambd_value,
        "layer_gates": layer_gate_vals,
        "head_gates": head_gate_vals,
        "layer_gates_tensor": layer_gates.detach() if layer_gates is not None else None,
        "head_gates_tensor": head_gates.detach() if head_gates is not None else None,
        "beta_tensor": beta_param.detach() if beta_param is not None else beta_init,
        "gamma_tensor": gamma_param.detach() if gamma_param is not None else gamma_init,
        "lambda_tensor": lambd_param.detach() if lambd_param is not None else lambd_init,
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Override self-attention on selected layers/heads")
    parser.add_argument("--layers", default="all", help="Layer spec: 'all' or comma-separated indices")
    parser.add_argument("--heads", default="all", help="Head spec: 'all' or comma-separated indices")
    parser.add_argument("--record-ids", default="", help="Comma-separated record ids to include; empty=all")
    parser.add_argument("--print-record-ids", default="", help="Comma-separated record ids to print attention for")
    parser.add_argument("--mode", default="post", choices=["pre", "post"], help="Override mode: pre|post")
    parser.add_argument("--cross-attn", action="store_true", help="Also override cross-attention using image masks")
    parser.add_argument("--mask-dir", default=str(MASK_DIR), help="Mask directory for cross-attention overrides")
    parser.add_argument("--cross-layers", default=None, help="Cross-attn layer spec; defaults to --layers")
    parser.add_argument("--cross-heads", default=None, help="Cross-attn head spec; defaults to --heads")
    parser.add_argument("--cls-only", action="store_true", help="Only CLS query attends to focused key tokens")
    parser.add_argument("--cls-plus", action="store_true", help="CLS and focus queries attend to focused key tokens")
    parser.add_argument("--all-queries", action="store_true", help="All queries attend to focused key tokens")
    parser.add_argument("--beta", type=float, default=12.0, help="Pre-softmax positive bias scale")
    parser.add_argument("--gamma", type=float, default=0.0, help="Pre-softmax negative bias scale")
    parser.add_argument("--lambda", dest="lambd", type=float, default=1.0, help="Post-softmax blend factor")
    parser.add_argument("--optimize", action="store_true", help="Optimize layer/head gates and beta/gamma")
    parser.add_argument("--opt-epochs", type=int, default=1, help="Optimization epochs")
    parser.add_argument("--opt-steps-per-record", type=int, default=10, help="Optimization steps per record")
    parser.add_argument("--opt-lr", type=float, default=0.05, help="Optimizer learning rate")
    parser.add_argument("--optimize-layers", action="store_true", help="Learn per-layer gates")
    parser.add_argument("--optimize-heads", action="store_true", help="Learn per-head gates")
    parser.add_argument("--optimize-beta", action="store_true", help="Learn beta")
    parser.add_argument("--optimize-gamma", action="store_true", help="Learn gamma")
    parser.add_argument("--optimize-lambda", action="store_true", help="Learn lambda (post mode)")
    parser.add_argument("--opt-out", default="", help="Write optimization summary JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record preds")
    args = parser.parse_args(argv)

    heads = parse_heads(args.heads)
    layers_spec = parse_heads(args.layers)
    cross_heads = parse_heads(args.cross_heads) if args.cross_heads is not None else heads
    cross_layers_spec = parse_heads(args.cross_layers) if args.cross_layers is not None else layers_spec
    record_ids = set(_parse_record_ids(args.record_ids))
    print_record_ids = set(_parse_record_ids(args.print_record_ids))
    mask_dir = Path(args.mask_dir)

    records = list(iter_jsonl(data_path))
    if not records:
        print("No records to evaluate.")
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
        sa = layer.attention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))
    cross_originals = []
    if args.cross_attn:
        for layer in encoder_layers:
            ca = layer.crossattention.self
            cross_originals.append((ca, ca.forward, getattr(ca, "save_attention", False)))

    if layers_spec == (-1,):
        target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        target_layers = tuple(int(x) for x in layers_spec)
    if cross_layers_spec == (-1,):
        cross_target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        cross_target_layers = tuple(int(x) for x in cross_layers_spec)

    tokenizer = model.tokenizer
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    heads_arg = heads[0] if len(heads) == 1 else heads
    cross_heads_arg = cross_heads[0] if len(cross_heads) == 1 else cross_heads

    if args.optimize:
        if not any(
            [
                args.optimize_layers,
                args.optimize_heads,
                args.optimize_beta,
                args.optimize_gamma,
                args.optimize_lambda,
            ]
        ):
            args.optimize_layers = True
            args.optimize_heads = True
            args.optimize_beta = True
            args.optimize_gamma = True
            if args.mode == "post":
                args.optimize_lambda = True
        if args.mode == "post" and (args.optimize_beta or args.optimize_gamma):
            print("[warn] beta/gamma have no effect in post mode.", file=sys.stderr)
        if args.mode == "pre" and args.optimize_lambda:
            print("[warn] lambda has no effect in pre mode.", file=sys.stderr)

        total_heads = _get_total_heads(encoder_layers)
        opt_result = _optimize_overrides(
            records,
            model,
            tokenizer,
            target_layers,
            heads_arg,
            originals,
            cls_only=args.cls_only,
            cls_plus=args.cls_plus,
            all_queries=args.all_queries,
            mode=args.mode,
            beta_init=args.beta,
            gamma_init=args.gamma,
            lambd_init=args.lambd,
            optimize_layers=args.optimize_layers,
            optimize_heads=args.optimize_heads,
            optimize_beta=args.optimize_beta,
            optimize_gamma=args.optimize_gamma,
            optimize_lambda=args.optimize_lambda,
            total_layers=total_layers,
            total_heads=total_heads,
            cross_attn=args.cross_attn,
            cross_target_layers=cross_target_layers,
            cross_heads_arg=cross_heads_arg,
            cross_originals=cross_originals,
            mask_dir=mask_dir,
            mask_cache=mask_cache,
            record_ids=record_ids,
            epochs=args.opt_epochs,
            steps_per_record=args.opt_steps_per_record,
            lr=args.opt_lr,
        )

        metrics = _evaluate_records(
            records,
            model,
            tokenizer,
            target_layers,
            heads_arg,
            originals,
            heads,
            cls_only=args.cls_only,
            cls_plus=args.cls_plus,
            all_queries=args.all_queries,
            mode=args.mode,
            beta=opt_result["beta_tensor"],
            gamma=opt_result["gamma_tensor"],
            lambd=opt_result["lambda_tensor"],
            cross_attn=args.cross_attn,
            cross_target_layers=cross_target_layers,
            cross_heads_arg=cross_heads_arg,
            cross_originals=cross_originals,
            mask_dir=mask_dir,
            mask_cache=mask_cache,
            head_gates=opt_result["head_gates_tensor"],
            layer_gates=opt_result["layer_gates_tensor"],
            record_ids=record_ids,
            print_record_ids=print_record_ids,
            verbose_records=not args.quiet,
            show_progress=True,
            collect_ids=True,
        )

        print("========================================")
        print(f"Layers: {list(target_layers)} | Heads: {list(heads) if len(heads) > 1 else heads[0]}")
        print(f"Optimized beta: {opt_result['beta']:.4f}")
        print(f"Optimized gamma: {opt_result['gamma']:.4f}")
        print(f"Optimized lambda: {opt_result['lambda']:.4f}")
        if opt_result["layer_gates"] is not None:
            print("Layer gates:", opt_result["layer_gates"])
        if opt_result["head_gates"] is not None:
            print("Head gates:", opt_result["head_gates"])
        print(
            f"Post-change accuracy: {metrics['total_hits']}/{metrics['total_trials']} "
            f"({metrics['accuracy']:.2f}%)"
        )
        if metrics["retain_trials"]:
            print(
                "Retention (originally-correct staying correct): "
                f"{metrics['retain_hits']}/{metrics['retain_trials']} "
                f"({metrics['retention_accuracy']:.2f}%)"
            )
        else:
            print("No items labeled as originally correct in dataset.")

        if args.opt_out:
            payload = {
                "beta": opt_result["beta"],
                "gamma": opt_result["gamma"],
                "lambda": opt_result["lambda"],
                "layer_gates": opt_result["layer_gates"],
                "head_gates": opt_result["head_gates"],
                "accuracy": metrics["accuracy"],
                "retention_accuracy": metrics["retention_accuracy"],
                "layers": list(target_layers),
                "heads": list(heads) if len(heads) > 1 else [heads[0]],
            }
            try:
                Path(args.opt_out).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.opt_out).open("w", encoding="utf-8") as f:
                    json.dump(payload, f, sort_keys=True, indent=2)
                print(f"Wrote optimization summary to {args.opt_out}")
            except OSError as exc:
                print(f"[warn] Failed to write optimization summary: {exc}", file=sys.stderr)
        return 0

    metrics = _evaluate_records(
        records,
        model,
        tokenizer,
        target_layers,
        heads_arg,
        originals,
        heads,
        cls_only=args.cls_only,
        cls_plus=args.cls_plus,
        all_queries=args.all_queries,
        mode=args.mode,
        beta=args.beta,
        gamma=args.gamma,
        lambd=args.lambd,
        cross_attn=args.cross_attn,
        cross_target_layers=cross_target_layers,
        cross_heads_arg=cross_heads_arg,
        cross_originals=cross_originals,
        mask_dir=mask_dir,
        mask_cache=mask_cache,
        head_gates=None,
        layer_gates=None,
        record_ids=record_ids,
        print_record_ids=print_record_ids,
        verbose_records=not args.quiet,
        show_progress=True,
        collect_ids=True,
    )

    print("========================================")
    print(f"Layers: {list(target_layers)} | Heads: {list(heads) if len(heads) > 1 else heads[0]}")
    print(
        f"Post-change accuracy: {metrics['total_hits']}/{metrics['total_trials']} "
        f"({metrics['accuracy']:.2f}%)"
    )
    if metrics["retain_trials"]:
        print(
            "Retention (originally-correct staying correct): "
            f"{metrics['retain_hits']}/{metrics['retain_trials']} "
            f"({metrics['retention_accuracy']:.2f}%)"
        )
    else:
        print("No items labeled as originally correct in dataset.")
    correct_ids = metrics.get("correct_ids", [])
    retained_correct_ids = metrics.get("retained_correct_ids", [])
    if correct_ids:
        print("\nCorrect IDs (all correct predictions):")
        for cid in correct_ids:
            print(cid)
    else:
        print("\nNo correct predictions.")
    if metrics["retain_trials"] and retained_correct_ids:
        print("\nRetained-Correct IDs (labeled correct and remained correct):")
        for cid in retained_correct_ids:
            print(cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
