"""Calculate probability deltas for baseline top beams after overriding heads/layers."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
import glob
import math
from math import comb
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

from models.blip_vqa import blip_vqa
from utils import *
from utils2 import * 

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root

@dataclass
class AnswerProbabilityReport:
    answer: str
    prob: float
    beam_score: float
    decoded_answers: List[str]


def _answer_key(text: str) -> str:
    canon = canonicalize_answer(text)
    if canon:
        return canon
    return normalize_answer(text)


def _answer_display(text: str) -> str:
    canon = canonicalize_answer(text)
    return canon if canon else (text or "").strip()

    
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
    """Return probability metrics for producing `answer` under the current model state. Expecting one word answers."""
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
                                                num_return_sequences=num_beams,
                                                return_dict_in_generate=True,
                                                output_scores=True,
                                                **model_kwargs)
        
        sequences = outputs.sequences
        sequence_scores = outputs.sequences_scores

        # Approximate sum log-prob from sequence_scores by undoing length penalty (default 1.0).
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

    target_display = _answer_display(cleaned_answer)
    target_key = _answer_key(cleaned_answer)

    decoded_answers = []
    decoded_norms: List[str] = []
    beam_score = 0.0
    beam_score_raw = 0.0
    for i in range(sequences.size(0)):
        seq = sequences[i]
        decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
        decoded_display = _answer_display(decoded)
        decoded_answers.append(decoded_display)
        decoded_norm = _answer_key(decoded)
        decoded_norms.append(decoded_norm)
        if decoded_norm == target_key:
            beam_score = np.exp(float(reconstructed[i].item()))
            beam_score_raw = float(sequence_scores[i].item())
            
        # elif target_norm and target_norm in decoded_norm:
        #     beam_score = np.exp(float(reconstructed[i].item()))

    if target_key not in decoded_norms:
        print("Target answer is not in the top-3 beams.")


    answer_for_prob = target_display if target_display else cleaned_answer
    ans = tokenizer(
        answer_for_prob,
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
        # prob = float(torch.exp(log_prob_sum).item())
    logprob = float(log_prob_sum.item())

    return AnswerProbabilityReport(
        answer=answer_for_prob,
        prob=logprob,
        beam_score=beam_score_raw,
        decoded_answers=decoded_answers,
    )

def _generate_top_beams(
    model,
    tokenizer,
    image: torch.Tensor,
    question: str,
    *,
    num_beams: int,
    max_length: int,
    min_length: int,
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
    seen = set()
    for idx, seq in enumerate(sequences):
        answer_raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
        if not answer_raw:
            continue
        answer = _answer_display(answer_raw)
        key = _answer_key(answer_raw)
        if key in seen:
            continue
        seen.add(key)
        score = float(scores[idx].item()) if scores is not None else float("nan")
        beams.append((answer, score))
        if len(beams) >= num_beams:
            break
    return beams

def _resolve_output_dir(path_str: str, heads_spec: str, layers_spec: str, target_id: str) -> Path:
    resolved = path_str.format(
        heads=sanitize_name(heads_spec),
        layers=sanitize_name(layers_spec),
        target_id=sanitize_name(target_id),
    )
    out_path = Path(resolved)
    if out_path.suffix.lower() == ".csv":
        return out_path.parent
    return out_path

def _plot_hist_on_ax(ax, values: List[float], title: str, x_label: str, color: str, *, show_zero_line: bool) -> None:
    if values:
        ax.hist(values, bins=30, color=color, alpha=0.6, edgecolor="black", linewidth=0.5)
        if show_zero_line:
            ax.axvline(0.0, color="black", linewidth=1.0)
        avg_val = sum(values) / len(values)
        ax.set_title(f"{title} (n={len(values)}, mean={avg_val:+.4f})")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title(f"{title} (n=0)")
        if show_zero_line:
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _plot_rank_combined_hist(
    *,
    prob_after: List[float],
    prob_delta: List[float],
    beam_after: List[float],
    beam_delta: List[float],
    out_path: Path,
    rank: int,
    answer_label: str,
    overall_note: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    _plot_hist_on_ax(
        axes[0, 0],
        prob_after,
        title=f"Logprob after (beam {rank}{answer_label})",
        x_label="Logprob after",
        color="#2a6fbb",
        show_zero_line=False,
    )
    _plot_hist_on_ax(
        axes[0, 1],
        prob_delta,
        title=f"Logprob delta (beam {rank}{answer_label})",
        x_label="Delta logprob (after - before)",
        color="#d17b0f",
        show_zero_line=True,
    )
    _plot_hist_on_ax(
        axes[1, 0],
        beam_after,
        title=f"Beam score after (beam {rank}{answer_label})",
        x_label="Beam score after",
        color="#2a6fbb",
        show_zero_line=False,
    )
    _plot_hist_on_ax(
        axes[1, 1],
        beam_delta,
        title=f"Beam score delta (beam {rank}{answer_label})",
        x_label="Delta beam score (after - before)",
        color="#d17b0f",
        show_zero_line=True,
    )

    if overall_note:
        # Keep a small "comment" note in the exported figure.
        fig.text(0.01, 0.01, f"# {overall_note}", ha="left", va="bottom", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scores_to_probs(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    values = torch.tensor(scores, dtype=torch.float64)
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return [1.0 / len(scores)] * len(scores)
    floor = values[finite].min() - 100.0
    values = torch.where(finite, values, floor)
    probs = torch.softmax(values, dim=0)
    return [float(v.item()) for v in probs]


def _compute_candidate_stability_features(
    *,
    before_logprob: float,
    after_logprobs: Sequence[float],
    after_probs: Sequence[float],
) -> Dict[str, float]:
    after_logprob_list = [float(v) for v in after_logprobs]
    after_prob_list = [float(v) for v in after_probs]
    deltas = [v - float(before_logprob) for v in after_logprob_list]
    delta_std = float(np.std(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0
    after_prob_std = float(np.std(np.asarray(after_prob_list, dtype=np.float64))) if after_prob_list else 0.0
    delta_mean = float(np.mean(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0
    after_prob_mean = float(np.mean(np.asarray(after_prob_list, dtype=np.float64))) if after_prob_list else 0.0
    p_pos = float(np.mean(np.asarray([1.0 if d > 0.0 else 0.0 for d in deltas], dtype=np.float64))) if deltas else 0.0
    return {
        "delta_logprob_mean_across_combos": float(delta_mean),
        "delta_logprob_std_across_combos": float(delta_std),
        "after_prob_mean_across_combos": float(after_prob_mean),
        "after_prob_std_across_combos": float(after_prob_std),
        "p_pos_across_combos": float(p_pos),
    }


BASELINE_STABILITY_METRIC_SPECS: List[Tuple[str, str, str]] = [
    (
        "delta_logprob_mean_across_combos",
        "Candidate stability: delta logprob mean",
        "mean(after_logprob - before_logprob) across combos",
    ),
    (
        "delta_logprob_std_across_combos",
        "Candidate stability: delta logprob std",
        "std(after_logprob - before_logprob) across combos",
    ),
    (
        "after_prob_mean_across_combos",
        "Candidate stability: after prob mean",
        "mean(softmax(after_logprob)) across combos",
    ),
    (
        "after_prob_std_across_combos",
        "Candidate stability: after prob std",
        "std(softmax(after_logprob)) across combos",
    ),
    (
        "p_pos_across_combos",
        "Candidate directional stability: p_pos",
        "P(delta_logprob > 0) across combos",
    ),
]

DYNAMIC_STABILITY_METRIC_SPECS: List[Tuple[str, str, str]] = BASELINE_STABILITY_METRIC_SPECS + [
    (
        "mean_after_rank_across_combos",
        "Dynamic candidates: mean after-rank",
        "mean rank in after_beams across combos",
    ),
]

TOP1_METRIC_SPECS: List[Tuple[str, str, str]] = [
    (
        "top1_after_is_gt_rate_across_combos",
        "Top1 correctness after override",
        "P(top1_after == gold) across combos",
    ),
    (
        "top1_changed_from_baseline_rate",
        "Top1 behavior: changed from baseline",
        "P(top1_after != baseline_top1) across combos",
    ),
    (
        "top1_switch_rate_between_combos",
        "Top1 behavior: switch rate",
        "P(top1 changes vs previous combo)",
    ),
    (
        "top1_top2_margin_beam_mean_across_combos",
        "Top1 confidence: beam margin mean",
        "mean(beam_score_top1 - beam_score_top2)",
    ),
    (
        "top1_top2_margin_beam_std_across_combos",
        "Top1 confidence: beam margin std",
        "std(beam_score_top1 - beam_score_top2)",
    ),
    (
        "top1_top2_margin_prob_mean_across_combos",
        "Top1 confidence: prob margin mean",
        "mean(prob_top1 - prob_top2)",
    ),
    (
        "top1_top2_margin_prob_std_across_combos",
        "Top1 confidence: prob margin std",
        "std(prob_top1 - prob_top2)",
    ),
]


def _derive_baseline_top1_is_gt(row: Dict[str, Any]) -> Optional[int]:
    raw = row.get("baseline_top1_is_gt")
    if raw in (0, 1):
        return int(raw)
    try:
        parsed = int(raw)
        if parsed in (0, 1):
            return parsed
    except Exception:
        pass

    baseline_norm = str(row.get("baseline_top1_answer_norm", "") or "").strip()
    gold_norm = str(row.get("gold_answer_norm", "") or "").strip()

    if not baseline_norm:
        baseline_text = str(row.get("baseline_top1_answer", "") or "")
        baseline_norm = _answer_key(baseline_text)
    if not gold_norm:
        gold_text = str(row.get("gold_answer", "") or "")
        gold_norm = _answer_key(gold_text)

    if not baseline_norm or not gold_norm:
        return None
    return int(baseline_norm == gold_norm)


def _derive_top1_after_is_gt_majority(row: Dict[str, Any]) -> Optional[int]:
    raw = row.get("top1_after_is_gt_majority")
    if raw in (0, 1):
        return int(raw)
    try:
        parsed = int(raw)
        if parsed in (0, 1):
            return parsed
    except Exception:
        pass

    raw_rate = row.get("top1_after_is_gt_rate_across_combos", None)
    if raw_rate is not None:
        try:
            rate = float(raw_rate)
            if np.isfinite(rate):
                return int(rate >= 0.5)
        except Exception:
            pass

    raw_count = row.get("top1_after_is_gt_count", None)
    raw_n = row.get("num_combo_observations", None)
    try:
        count = int(raw_count)
        n_obs = int(raw_n)
        if n_obs > 0:
            return int((float(count) / float(n_obs)) >= 0.5)
    except Exception:
        pass
    return None


def _plot_gt_vs_non_gt_metric_grid(
    *,
    rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[Tuple[str, str, str]],
    out_path: Path,
    plot_style: str = "violin",
    overall_note: Optional[str] = None,
    group_key: str = "is_gt_candidate",
    gt_label: str = "GT",
    non_gt_label: str = "non-GT",
) -> None:
    if not metric_specs:
        return

    def _extract_vals(metric_key: str) -> Tuple[List[float], List[float]]:
        gt_vals: List[float] = []
        non_gt_vals: List[float] = []
        for row in rows:
            raw_group = row.get(group_key)
            if raw_group not in (0, 1):
                try:
                    raw_group = int(raw_group)
                except Exception:
                    if group_key == "baseline_top1_is_gt":
                        raw_group = _derive_baseline_top1_is_gt(row)
                    elif group_key == "top1_after_is_gt_majority":
                        raw_group = _derive_top1_after_is_gt_majority(row)
                    else:
                        continue
            if raw_group not in (0, 1):
                continue
            raw_value = row.get(metric_key)
            try:
                value = float(raw_value)
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            if int(raw_group) == 1:
                gt_vals.append(value)
            else:
                non_gt_vals.append(value)
        return gt_vals, non_gt_vals

    def _plot_group(
        ax,
        *,
        gt_vals: Sequence[float],
        non_gt_vals: Sequence[float],
        title: str,
        metric_label: str,
    ) -> None:
        gt_list = [float(v) for v in gt_vals]
        non_gt_list = [float(v) for v in non_gt_vals]
        if not gt_list and not non_gt_list:
            ax.text(0.5, 0.5, "No labeled data", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(f"{title} (n=0)")
            if plot_style == "hist":
                ax.set_xlabel(metric_label)
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel(metric_label)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            return

        if plot_style == "hist":
            combined = non_gt_list + gt_list
            bins = 30
            if combined:
                try:
                    bins = np.histogram_bin_edges(np.asarray(combined, dtype=np.float64), bins=30)
                except Exception:
                    bins = 30
            if non_gt_list:
                non_gt_mean = float(np.mean(np.asarray(non_gt_list, dtype=np.float64)))
                ax.hist(
                    non_gt_list,
                    bins=bins,
                    color="#d55e00",
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"{non_gt_label} n={len(non_gt_list)} mean={non_gt_mean:+.3f}",
                )
            if gt_list:
                gt_mean = float(np.mean(np.asarray(gt_list, dtype=np.float64)))
                ax.hist(
                    gt_list,
                    bins=bins,
                    color="#009e73",
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"{gt_label} n={len(gt_list)} mean={gt_mean:+.3f}",
                )
            ax.set_title(title)
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            return

        data: List[List[float]] = []
        positions: List[int] = []
        labels: List[str] = []
        colors: List[str] = []
        if non_gt_list:
            data.append(non_gt_list)
            positions.append(1)
            non_gt_mean = float(np.mean(np.asarray(non_gt_list, dtype=np.float64)))
            labels.append(f"{non_gt_label}\nn={len(non_gt_list)}\nmean={non_gt_mean:+.3f}")
            colors.append("#d55e00")
        if gt_list:
            data.append(gt_list)
            positions.append(2 if non_gt_list else 1)
            gt_mean = float(np.mean(np.asarray(gt_list, dtype=np.float64)))
            labels.append(f"{gt_label}\nn={len(gt_list)}\nmean={gt_mean:+.3f}")
            colors.append("#009e73")

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")

        ax.set_title(title)
        ax.set_ylabel(metric_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    cols = 3
    n_metrics = len(metric_specs)
    rows_n = int(math.ceil(float(n_metrics) / float(cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.8 * cols, 4.8 * rows_n))
    axes_arr = np.atleast_1d(axes).reshape(rows_n, cols)
    flat_axes = [ax for row in axes_arr for ax in row]

    for idx, (metric_key, title, metric_label) in enumerate(metric_specs):
        gt_vals, non_gt_vals = _extract_vals(metric_key)
        _plot_group(
            flat_axes[idx],
            gt_vals=gt_vals,
            non_gt_vals=non_gt_vals,
            title=title,
            metric_label=metric_label,
        )
    for ax in flat_axes[n_metrics:]:
        ax.axis("off")

    if overall_note:
        fig.text(0.01, 0.01, f"# {overall_note}", ha="left", va="bottom", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gt_vs_non_gt_stability(
    *,
    gt_delta_logprob_std: Sequence[float],
    non_gt_delta_logprob_std: Sequence[float],
    gt_after_prob_std: Sequence[float],
    non_gt_after_prob_std: Sequence[float],
    gt_p_pos: Sequence[float],
    non_gt_p_pos: Sequence[float],
    out_path: Path,
    plot_style: str = "violin",
    overall_note: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    def _plot_group(
        ax,
        *,
        gt_vals: Sequence[float],
        non_gt_vals: Sequence[float],
        title: str,
        metric_label: str,
    ) -> None:
        gt_list = [float(v) for v in gt_vals]
        non_gt_list = [float(v) for v in non_gt_vals]
        if not gt_list and not non_gt_list:
            ax.text(0.5, 0.5, "No labeled candidates", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(f"{title} (n=0)")
            if plot_style == "hist":
                ax.set_xlabel(metric_label)
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel(metric_label)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            return

        if plot_style == "hist":
            combined = non_gt_list + gt_list
            bins = 30
            if combined:
                try:
                    bins = np.histogram_bin_edges(np.asarray(combined, dtype=np.float64), bins=30)
                except Exception:
                    bins = 30
            if non_gt_list:
                non_gt_mean = float(np.mean(np.asarray(non_gt_list, dtype=np.float64)))
                ax.hist(
                    non_gt_list,
                    bins=bins,
                    color="#d55e00",
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"non-GT n={len(non_gt_list)} mean={non_gt_mean:+.3f}",
                )
            if gt_list:
                gt_mean = float(np.mean(np.asarray(gt_list, dtype=np.float64)))
                ax.hist(
                    gt_list,
                    bins=bins,
                    color="#009e73",
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"GT n={len(gt_list)} mean={gt_mean:+.3f}",
                )
            ax.set_title(title)
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            return

        data: List[List[float]] = []
        positions: List[int] = []
        labels: List[str] = []
        colors: List[str] = []
        if non_gt_list:
            data.append(non_gt_list)
            positions.append(1)
            non_gt_mean = float(np.mean(np.asarray(non_gt_list, dtype=np.float64)))
            labels.append(f"non-GT\nn={len(non_gt_list)}\nmean={non_gt_mean:+.3f}")
            colors.append("#d55e00")
        if gt_list:
            data.append(gt_list)
            positions.append(2 if non_gt_list else 1)
            gt_mean = float(np.mean(np.asarray(gt_list, dtype=np.float64)))
            labels.append(f"GT\nn={len(gt_list)}\nmean={gt_mean:+.3f}")
            colors.append("#009e73")

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")

        ax.set_title(title)
        ax.set_ylabel(metric_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    _plot_group(
        axes[0],
        gt_vals=gt_delta_logprob_std,
        non_gt_vals=non_gt_delta_logprob_std,
        title="Candidate stability across combos: delta logprob std",
        metric_label="std(after_logprob - before_logprob) across combos",
    )
    _plot_group(
        axes[1],
        gt_vals=gt_after_prob_std,
        non_gt_vals=non_gt_after_prob_std,
        title="Candidate stability across combos: after prob std",
        metric_label="std(softmax(after_logprob)) across combos",
    )
    _plot_group(
        axes[2],
        gt_vals=gt_p_pos,
        non_gt_vals=non_gt_p_pos,
        title="Candidate directional stability: p_pos",
        metric_label="P(delta_logprob > 0) across combos",
    )

    if overall_note:
        fig.text(0.01, 0.01, f"# {overall_note}", ha="left", va="bottom", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_layers(spec: str, total_layers: int) -> Tuple[int, ...]:
    parsed = parse_heads(spec)
    if parsed == (-1,):
        return tuple(range(total_layers))
    return tuple(int(x) for x in parsed)


def _resolve_heads(spec: str) -> Sequence[int]:
    parsed = parse_heads(spec)
    if parsed == (-1,):
        return tuple(range(12))
    return parsed


def _parse_pair_combo_spec(spec: str) -> Dict[int, Tuple[int, ...]]:
    if not spec.startswith("pairs:"):
        return {}
    body = spec[len("pairs:") :].strip()
    if not body:
        return {}
    layer_to_heads: Dict[int, set] = defaultdict(set)
    for chunk in body.split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if "h" not in token or not token.startswith("l"):
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


def _iter_jsonl_rows(path: Path) -> Sequence[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                rows.append(row)
    return rows


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _resolve_aggregate_feature_paths(
    jsonl_args: Sequence[str],
    glob_pattern: Optional[str],
) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for raw in jsonl_args:
        p = Path(str(raw))
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)
    if glob_pattern:
        for raw in sorted(glob.glob(str(glob_pattern), recursive=True)):
            p = Path(raw)
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
    return paths


def _run_stability_aggregation_from_files(
    *,
    feature_paths: Sequence[Path],
    out_dir: Path,
    plot_style: str,
) -> int:
    loaded_rows: List[Dict[str, Any]] = []
    existing_paths: List[Path] = []
    for path in feature_paths:
        if not path.exists():
            print(f"[warn] Missing feature JSONL: {path}; skipping")
            continue
        rows = _iter_jsonl_rows(path)
        if not rows:
            print(f"[warn] No rows in feature JSONL: {path}; skipping")
            continue
        for row in rows:
            row_copy = dict(row)
            row_copy["_source_file"] = str(path)
            loaded_rows.append(row_copy)
        existing_paths.append(path)

    if not loaded_rows:
        print("[error] No feature rows loaded for aggregation.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_jsonl = out_dir / "candidate_stability_features_merged.jsonl"
    with merged_jsonl.open("w", encoding="utf-8") as f:
        for row in loaded_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gt_delta_std_vals: List[float] = []
    non_gt_delta_std_vals: List[float] = []
    gt_after_prob_std_vals: List[float] = []
    non_gt_after_prob_std_vals: List[float] = []
    gt_p_pos_vals: List[float] = []
    non_gt_p_pos_vals: List[float] = []
    labeled_rows = 0
    for row in loaded_rows:
        raw_is_gt = row.get("is_gt_candidate")
        if raw_is_gt not in (0, 1):
            try:
                raw_is_gt = int(raw_is_gt)
            except Exception:
                continue
        if raw_is_gt not in (0, 1):
            continue

        delta_std = _to_float_or_none(row.get("delta_logprob_std_across_combos"))
        after_prob_std = _to_float_or_none(row.get("after_prob_std_across_combos"))
        p_pos = _to_float_or_none(row.get("p_pos_across_combos"))
        if delta_std is None and after_prob_std is None and p_pos is None:
            continue
        labeled_rows += 1
        if int(raw_is_gt) == 1:
            if delta_std is not None:
                gt_delta_std_vals.append(delta_std)
            if after_prob_std is not None:
                gt_after_prob_std_vals.append(after_prob_std)
            if p_pos is not None:
                gt_p_pos_vals.append(p_pos)
        else:
            if delta_std is not None:
                non_gt_delta_std_vals.append(delta_std)
            if after_prob_std is not None:
                non_gt_after_prob_std_vals.append(after_prob_std)
            if p_pos is not None:
                non_gt_p_pos_vals.append(p_pos)

    note = (
        f"aggregate_rows={len(loaded_rows)} labeled_rows={labeled_rows} "
        f"files={len(existing_paths)}"
    )
    plot_path = out_dir / "candidate_stability_gt_vs_non_gt_aggregate.pdf"
    _plot_gt_vs_non_gt_stability(
        gt_delta_logprob_std=gt_delta_std_vals,
        non_gt_delta_logprob_std=non_gt_delta_std_vals,
        gt_after_prob_std=gt_after_prob_std_vals,
        non_gt_after_prob_std=non_gt_after_prob_std_vals,
        gt_p_pos=gt_p_pos_vals,
        non_gt_p_pos=non_gt_p_pos_vals,
        out_path=plot_path,
        plot_style=plot_style,
        overall_note=note,
    )
    plot_path_extended = out_dir / "candidate_stability_gt_vs_non_gt_aggregate_extended.pdf"
    _plot_gt_vs_non_gt_metric_grid(
        rows=loaded_rows,
        metric_specs=BASELINE_STABILITY_METRIC_SPECS,
        out_path=plot_path_extended,
        plot_style=plot_style,
        overall_note=note,
        group_key="is_gt_candidate",
        gt_label="GT",
        non_gt_label="non-GT",
    )

    gt_count = len(gt_delta_std_vals)
    non_gt_count = len(non_gt_delta_std_vals)
    gt_delta_std_mean = float(np.mean(np.asarray(gt_delta_std_vals, dtype=np.float64))) if gt_delta_std_vals else float("nan")
    non_gt_delta_std_mean = (
        float(np.mean(np.asarray(non_gt_delta_std_vals, dtype=np.float64))) if non_gt_delta_std_vals else float("nan")
    )
    gt_after_prob_std_mean = (
        float(np.mean(np.asarray(gt_after_prob_std_vals, dtype=np.float64))) if gt_after_prob_std_vals else float("nan")
    )
    non_gt_after_prob_std_mean = (
        float(np.mean(np.asarray(non_gt_after_prob_std_vals, dtype=np.float64))) if non_gt_after_prob_std_vals else float("nan")
    )
    gt_p_pos_mean = float(np.mean(np.asarray(gt_p_pos_vals, dtype=np.float64))) if gt_p_pos_vals else float("nan")
    non_gt_p_pos_mean = float(np.mean(np.asarray(non_gt_p_pos_vals, dtype=np.float64))) if non_gt_p_pos_vals else float("nan")

    print(f"[info] Aggregated feature files: {len(existing_paths)}")
    print(f"[info] Wrote merged features: {merged_jsonl}")
    style_label = "histogram" if plot_style == "hist" else "violin+box"
    print(f"[info] Wrote aggregate GT vs non-GT {style_label} plot: {plot_path}")
    print(f"[info] Wrote aggregate GT vs non-GT extended {style_label} plot: {plot_path_extended}")
    print(
        "[stability][aggregate] "
        f"GT n={gt_count}, non-GT n={non_gt_count} | "
        f"delta_logprob_std mean GT={gt_delta_std_mean:+.6f} vs non-GT={non_gt_delta_std_mean:+.6f} | "
        f"after_prob_std mean GT={gt_after_prob_std_mean:+.6f} vs non-GT={non_gt_after_prob_std_mean:+.6f} | "
        f"p_pos mean GT={gt_p_pos_mean:+.6f} vs non-GT={non_gt_p_pos_mean:+.6f}"
    )

    # --- Run per-record paired statistical tests (GT vs wrong answers) ---
    try:
        from analyze_stability_statistics import (
            _group_by_record,
            _analyze_one_record,
            _print_aggregate_summary,
            _write_csv as _write_stat_csv,
            _write_record_summary_csv,
        )
        grouped = _group_by_record(loaded_rows)
        record_results = []
        for rid in sorted(grouped.keys()):
            result = _analyze_one_record(
                rid, grouped[rid],
                n_permutations=10_000,
                n_bootstrap=10_000,
                alpha=0.05,
                min_combos=5,
            )
            if result is not None:
                record_results.append(result)
        if record_results:
            _print_aggregate_summary(record_results, 0.05)
            stat_csv = out_dir / "statistical_tests_per_pair.csv"
            _write_stat_csv(record_results, stat_csv, 0.05)
            summary_csv = out_dir / "statistical_tests_per_record.csv"
            _write_record_summary_csv(record_results, summary_csv)
        else:
            print("[info] No records with both GT and wrong candidates + raw arrays for paired tests.")
    except Exception as exc:
        print(f"[warn] Could not run per-record statistical tests: {exc}")

    # Automatically aggregate sibling per-record artifacts if present.
    dynamic_paths: List[Path] = []
    top1_paths: List[Path] = []
    seen_dynamic = set()
    seen_top1 = set()
    for base_path in existing_paths:
        dyn_path = base_path.parent / "candidate_stability_features_new_answers.jsonl"
        top1_path = base_path.parent / "top1_margin_change_features.jsonl"
        if dyn_path.exists():
            key = str(dyn_path.resolve())
            if key not in seen_dynamic:
                seen_dynamic.add(key)
                dynamic_paths.append(dyn_path)
        if top1_path.exists():
            key = str(top1_path.resolve())
            if key not in seen_top1:
                seen_top1.add(key)
                top1_paths.append(top1_path)

    dynamic_rows: List[Dict[str, Any]] = []
    for path in dynamic_paths:
        rows = _iter_jsonl_rows(path)
        if not rows:
            continue
        for row in rows:
            row_copy = dict(row)
            row_copy["_source_file"] = str(path)
            dynamic_rows.append(row_copy)

    if dynamic_rows:
        dynamic_merged_jsonl = out_dir / "candidate_stability_features_new_answers_merged.jsonl"
        with dynamic_merged_jsonl.open("w", encoding="utf-8") as f:
            for row in dynamic_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        dynamic_plot = out_dir / "candidate_stability_new_answers_gt_vs_non_gt_aggregate_extended.pdf"
        dynamic_note = f"aggregate_rows={len(dynamic_rows)} files={len(dynamic_paths)} source=dynamic_answers"
        _plot_gt_vs_non_gt_metric_grid(
            rows=dynamic_rows,
            metric_specs=DYNAMIC_STABILITY_METRIC_SPECS,
            out_path=dynamic_plot,
            plot_style=plot_style,
            overall_note=dynamic_note,
            group_key="is_gt_candidate",
            gt_label="GT",
            non_gt_label="non-GT",
        )

        dyn_gt_rows = 0
        dyn_non_gt_rows = 0
        for row in dynamic_rows:
            raw_is_gt = row.get("is_gt_candidate")
            if raw_is_gt not in (0, 1):
                try:
                    raw_is_gt = int(raw_is_gt)
                except Exception:
                    continue
            if raw_is_gt == 1:
                dyn_gt_rows += 1
            elif raw_is_gt == 0:
                dyn_non_gt_rows += 1
        print(f"[info] Aggregated dynamic new-answer feature files: {len(dynamic_paths)}")
        print(f"[info] Wrote merged dynamic new-answer features: {dynamic_merged_jsonl}")
        print(f"[info] Wrote aggregate dynamic new-answer GT vs non-GT {style_label} plot: {dynamic_plot}")
        print(
            "[stability][aggregate][new-answers] "
            f"rows={len(dynamic_rows)} | GT rows={dyn_gt_rows}, non-GT rows={dyn_non_gt_rows}"
        )

    top1_rows: List[Dict[str, Any]] = []
    for path in top1_paths:
        rows = _iter_jsonl_rows(path)
        if not rows:
            continue
        for row in rows:
            row_copy = dict(row)
            derived_group = _derive_baseline_top1_is_gt(row_copy)
            if derived_group in (0, 1):
                row_copy["baseline_top1_is_gt"] = int(derived_group)
            derived_after_group = _derive_top1_after_is_gt_majority(row_copy)
            if derived_after_group in (0, 1):
                row_copy["top1_after_is_gt_majority"] = int(derived_after_group)
            row_copy["_source_file"] = str(path)
            top1_rows.append(row_copy)

    if top1_rows:
        top1_merged_jsonl = out_dir / "top1_margin_change_features_merged.jsonl"
        with top1_merged_jsonl.open("w", encoding="utf-8") as f:
            for row in top1_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        top1_plot = out_dir / "top1_metrics_after_gt_vs_non_gt_aggregate_extended.pdf"
        top1_note = f"aggregate_rows={len(top1_rows)} files={len(top1_paths)} source=top1_metrics"
        _plot_gt_vs_non_gt_metric_grid(
            rows=top1_rows,
            metric_specs=TOP1_METRIC_SPECS,
            out_path=top1_plot,
            plot_style=plot_style,
            overall_note=top1_note,
            group_key="top1_after_is_gt_majority",
            gt_label="after top1 mostly=GT",
            non_gt_label="after top1 mostly!=GT",
        )

        top1_gt_rows = 0
        top1_non_gt_rows = 0
        top1_after_gt_rates: List[float] = []
        for row in top1_rows:
            raw_group = _derive_top1_after_is_gt_majority(row)
            if raw_group == 1:
                top1_gt_rows += 1
            elif raw_group == 0:
                top1_non_gt_rows += 1
            raw_rate = row.get("top1_after_is_gt_rate_across_combos", None)
            if raw_rate is not None:
                try:
                    rate_val = float(raw_rate)
                    if np.isfinite(rate_val):
                        top1_after_gt_rates.append(rate_val)
                except Exception:
                    pass
        print(f"[info] Aggregated top1 feature files: {len(top1_paths)}")
        print(f"[info] Wrote merged top1 features: {top1_merged_jsonl}")
        print(f"[info] Wrote aggregate top1 (after-top1 grouped) GT-vs-non-GT {style_label} plot: {top1_plot}")
        print(
            "[top1][aggregate] "
            f"rows={len(top1_rows)} | after top1 mostly=GT rows={top1_gt_rows}, after top1 mostly!=GT rows={top1_non_gt_rows} | "
            f"after_top1_is_gt mean={(float(np.mean(np.asarray(top1_after_gt_rates, dtype=np.float64))) if top1_after_gt_rates else float('nan')):.6f}"
        )

    return 0


def _resolve_topk_path_template(path_value: str, top_k: int) -> Path:
    if "{k}" in path_value:
        return Path(path_value.format(k=int(top_k)))
    return Path(path_value)


def _encode_pair_combo_spec(pairs: Sequence[Tuple[int, int]]) -> str:
    ordered = sorted((int(layer), int(head)) for layer, head in pairs)
    return "pairs:" + ",".join(f"l{layer}h{head}" for layer, head in ordered)


def _load_ranked_pairs_for_target(
    rankings_path: Path,
    *,
    target_id: str,
) -> List[Dict[str, Any]]:
    target_norm = str(target_id).strip().lower()
    if not rankings_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")
    for row in _iter_jsonl_rows(rankings_path):
        rec_id = str(row.get("id", "")).strip().lower()
        if rec_id != target_norm:
            continue
        ranked_pairs = row.get("ranked_pairs", [])
        if not isinstance(ranked_pairs, list):
            break
        return [item for item in ranked_pairs if isinstance(item, dict)]
    raise ValueError(f"No ranking row with id={target_id!r} found in {rankings_path}.")


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
            layer = int(item.get("layer"))  # type: ignore[arg-type]
            head = int(item.get("head"))  # type: ignore[arg-type]
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


def _dedupe_ints(values: Sequence[int]) -> Tuple[int, ...]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return tuple(ordered)


def _indices_to_spec(indices: Sequence[int]) -> str:
    return ",".join(str(i) for i in indices)


def _sample_random_head_layer_specs(
    head_pool: Sequence[int],
    layer_pool: Sequence[int],
    *,
    head_count: int,
    layer_count: int,
    combo_count: int,
    seed: Optional[int],
) -> List[Tuple[str, str]]:
    if combo_count <= 0:
        return []
    if head_count <= 0:
        raise ValueError("--random-head-count must be > 0 when --random-combos is enabled.")
    if layer_count <= 0:
        raise ValueError("--random-layer-count must be > 0 when --random-combos is enabled.")

    heads = _dedupe_ints(head_pool)
    layers = _dedupe_ints(layer_pool)
    if head_count > len(heads):
        raise ValueError(f"--random-head-count ({head_count}) exceeds available head pool size ({len(heads)}).")
    if layer_count > len(layers):
        raise ValueError(f"--random-layer-count ({layer_count}) exceeds available layer pool size ({len(layers)}).")

    total_unique = comb(len(heads), head_count) * comb(len(layers), layer_count)
    if combo_count > total_unique:
        raise ValueError(
            f"Requested --random-combos={combo_count}, but only {total_unique} unique combinations exist "
            f"for head_count={head_count}, layer_count={layer_count}."
        )

    rng = random.Random(seed)
    head_combos = list(itertools.combinations(heads, head_count))
    layer_combos = list(itertools.combinations(layers, layer_count))
    layer_combo_count = len(layer_combos)

    # If the requested set is dense, sample by index without replacement directly.
    sampled: List[Tuple[str, str]] = []
    if combo_count > (total_unique // 2):
        sampled_indices = rng.sample(range(total_unique), combo_count)
        for flat_idx in sampled_indices:
            head_idx, layer_idx = divmod(flat_idx, layer_combo_count)
            sampled.append(
                (
                    _indices_to_spec(head_combos[head_idx]),
                    _indices_to_spec(layer_combos[layer_idx]),
                )
            )
        return sampled

    # Sparse request path: rejection sampling of pair selections.
    selected = set()
    while len(sampled) < combo_count:
        head_choice = head_combos[rng.randrange(len(head_combos))]
        layer_choice = layer_combos[rng.randrange(layer_combo_count)]
        key = (head_choice, layer_choice)
        if key in selected:
            continue
        selected.add(key)
        sampled.append((_indices_to_spec(head_choice), _indices_to_spec(layer_choice)))
    return sampled


def _build_cached_record(
    entry: Dict[str, Any],
    *,
    model,
    tokenizer,
    device: torch.device,
    num_beams: int,
    max_length: int,
    min_length: int,
) -> Optional[Dict[str, Any]]:
    rec_id_val = entry.get("id")
    rec_id = "" if rec_id_val is None else str(rec_id_val)
    image_value = entry.get("image")
    question = entry.get("question")
    prompt = entry.get("prompt")
    if image_value is None or question is None:
        print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
        return None

    image_rel = Path(image_value)
    image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
    if not image_path.exists():
        print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
        return None

    image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
    question_inputs = _prepare_question_inputs(tokenizer, question, image.device)

    mask_array: Optional[np.ndarray] = None
    stem = image_rel.stem
    rec_id_for_mask = rec_id if rec_id else None
    if prompt:
        mask_array = load_mask_from_dir(MASK_DIR, stem, prompt, rec_id_for_mask)
    gh = gw = IMAGE_SIZE // PATCH_SIZE
    mask_array = ensure_mask(mask_array, gh, gw, stem=stem)
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

    tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
    focus_words = guess_focus_words(question)
    override_indices = select_override_indices(tokens, focus_words, tokenizer)
    if not override_indices:
        override_indices = [0]
    override_rows = {i: mask_soft for i in override_indices}

    with torch.no_grad():
        image_embeds = model.visual_encoder(image)
    image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    beams = _generate_top_beams(
        model,
        tokenizer,
        image,
        question,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
    )

    baseline_beams: List[Dict[str, Any]] = []
    for idx, (answer, beam_score) in enumerate(beams, 1):
        report_baseline = compute_answer_probability(
            model,
            tokenizer,
            image,
            question_inputs,
            answer,
            image_embeds=image_embeds,
            image_att_mask=image_att_mask,
            num_beams=num_beams,
        )
        baseline_beams.append(
            {
                "rank": idx,
                "answer": answer,
                "beam_score": beam_score,
                "prob_before": report_baseline.prob,
            }
        )

    picked_answer = baseline_beams[0]["answer"] if baseline_beams else ""
    return {
        "record_id": rec_id,
        "question": question,
        "image": image,
        "gold_answer": str(entry.get("answer", "") or ""),
        "gold_answer_norm": _answer_key(str(entry.get("answer", "") or "")),
        "picked_answer": picked_answer,
        "question_inputs": question_inputs,
        "image_embeds": image_embeds,
        "image_att_mask": image_att_mask,
        "override_rows": override_rows,
        "baseline_beams": baseline_beams,
    }


def _build_hybrid_eval_combos(
    *,
    args,
    records: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    total_layers: int,
    originals,
) -> List[Tuple[str, str]]:
    combo_count = int(args.hybrid_combos)
    pairs_per_combo = int(args.hybrid_pairs_per_combo)
    top_pairs_limit = int(args.hybrid_top_pairs)
    high_candidate_count = int(args.hybrid_high_candidates)
    high_frac = float(args.hybrid_high_frac)
    var_weight = float(args.hybrid_var_weight)
    calib_size = max(int(args.hybrid_calib_size), 1)
    seed = args.hybrid_seed

    if combo_count <= 0:
        return []
    if pairs_per_combo <= 0:
        raise ValueError("--hybrid-pairs-per-combo must be > 0 when --hybrid-combos is enabled.")
    if high_frac < 0.0 or high_frac > 1.0:
        raise ValueError("--hybrid-high-frac must be within [0, 1].")

    head_pool = _dedupe_ints(_resolve_heads(args.heads))
    layer_pool = _dedupe_ints(_resolve_layers(args.layers, total_layers))
    pair_pool: List[Tuple[int, int]] = [(layer_idx, head_idx) for layer_idx in layer_pool for head_idx in head_pool]
    if pairs_per_combo > len(pair_pool):
        raise ValueError(
            f"--hybrid-pairs-per-combo ({pairs_per_combo}) exceeds available pair pool size ({len(pair_pool)})."
        )

    total_unique = comb(len(pair_pool), pairs_per_combo)
    if combo_count > total_unique:
        raise ValueError(
            f"Requested --hybrid-combos={combo_count}, but only {total_unique} unique combinations exist "
            f"for pairs_per_combo={pairs_per_combo}."
        )

    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []
    for entry in records:
        if entry.get("image") is None or entry.get("question") is None or entry.get("answer") is None:
            continue
        if int(args.hybrid_fail_only) != 0:
            correct_tag = str(entry.get("correct", "")).strip().lower()
            if correct_tag not in {"no", "false", "0"}:
                continue
        candidates.append(entry)
    if not candidates and int(args.hybrid_fail_only) != 0:
        print("[warn] No failure-tagged calibration records found; falling back to all records with answers.")
        for entry in records:
            if entry.get("image") is None or entry.get("question") is None or entry.get("answer") is None:
                continue
            candidates.append(entry)
    if not candidates:
        raise ValueError("No calibration records available for --hybrid-combos.")

    rng.shuffle(candidates)
    calib_records: List[Dict[str, Any]] = []
    for entry in candidates:
        cached = _build_cached_record(
            entry,
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
        )
        if not cached:
            continue
        gold_answer = str(cached.get("gold_answer", "") or "").strip()
        picked_answer = str(cached.get("picked_answer", "") or "").strip()
        if not gold_answer or not picked_answer:
            continue
        report_gold_before = compute_answer_probability(
            model,
            tokenizer,
            cached["image"],
            cached["question_inputs"],
            gold_answer,
            image_embeds=cached["image_embeds"],
            image_att_mask=cached["image_att_mask"],
            num_beams=args.num_beams,
        )
        report_pick_before = compute_answer_probability(
            model,
            tokenizer,
            cached["image"],
            cached["question_inputs"],
            picked_answer,
            image_embeds=cached["image_embeds"],
            image_att_mask=cached["image_att_mask"],
            num_beams=args.num_beams,
        )
        margin_before = report_gold_before.prob - report_pick_before.prob
        cached["margin_before"] = margin_before
        calib_records.append(cached)
        if len(calib_records) >= calib_size:
            break

    if not calib_records:
        raise ValueError("Failed to build any calibration records for hybrid selection.")
    print(f"[info] Hybrid calibration records: {len(calib_records)}")
    print(f"[info] Hybrid pair scan size: {len(pair_pool)} (layers={len(layer_pool)}, heads={len(head_pool)})")

    pair_effects: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for layer_idx, head_idx in pair_pool:
        for record in calib_records:
            new_forward = make_forward(head_idx, record["override_rows"], LAMBDA=1.0)
            apply_override(new_forward, (layer_idx,), originals)
            try:
                report_gold_after = compute_answer_probability(
                    model,
                    tokenizer,
                    record["image"],
                    record["question_inputs"],
                    record["gold_answer"],
                    image_embeds=record["image_embeds"],
                    image_att_mask=record["image_att_mask"],
                    num_beams=args.num_beams,
                )
                report_pick_after = compute_answer_probability(
                    model,
                    tokenizer,
                    record["image"],
                    record["question_inputs"],
                    record["picked_answer"],
                    image_embeds=record["image_embeds"],
                    image_att_mask=record["image_att_mask"],
                    num_beams=args.num_beams,
                )
            finally:
                revert_override((layer_idx,), originals)
            margin_after = report_gold_after.prob - report_pick_after.prob
            effect = margin_after - float(record["margin_before"])
            pair_effects[(layer_idx, head_idx)].append(effect)


    pair_means: Dict[Tuple[int, int], float] = {}
    pair_scores: Dict[Tuple[int, int], float] = {}
    for pair in pair_pool:
        effects = pair_effects.get(pair, [])
        arr = np.asarray(effects, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.mean() < -0.2 and (arr < 0).mean() > 0.75:
            continue
        pair_scores[pair] = float(arr.mean()) + var_weight * float(arr.std())
        pair_means[pair] = float(np.mean(effects)) if effects else 0.0
    pairs_ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    top_print = pairs_ranked[: min(12, len(pairs_ranked))]
    if top_print:
        print(
            "[info] Hybrid top pairs:",
            ", ".join(f"L{layer}H{head}:{score:.4f}" for (layer, head), score in top_print),
        )
    if top_pairs_limit <= 0:
        top_pairs_limit = min(24, len(pair_pool))
    top_pairs_limit = max(top_pairs_limit, pairs_per_combo)
    top_pairs = [pair for pair, _ in pairs_ranked[:top_pairs_limit]]
    if len(top_pairs) < pairs_per_combo:
        raise ValueError("Not enough top pairs to form leverage combos; reduce --hybrid-pairs-per-combo.")

    high_count = max(0, min(combo_count, int(round(combo_count * high_frac))))
    random_count = combo_count - high_count

    def _encode_pair_combo(pairs: Sequence[Tuple[int, int]]) -> str:
        ordered = sorted((int(layer), int(head)) for layer, head in pairs)
        return "pairs:" + ",".join(f"l{layer}h{head}" for layer, head in ordered)

    def _generate_candidate_pair_combos(
        pool: Sequence[Tuple[int, int]],
        *,
        k: int,
        max_count: int,
        rng_obj: random.Random,
    ) -> List[Tuple[Tuple[int, int], ...]]:
        if max_count <= 0:
            return []
        total = comb(len(pool), k)
        if total <= max_count:
            return [tuple(combo) for combo in itertools.combinations(pool, k)]
        sampled: List[Tuple[Tuple[int, int], ...]] = []
        seen_idx = set()
        attempts = 0
        max_attempts = max(20000, max_count * 40)
        while len(sampled) < max_count and attempts < max_attempts:
            attempts += 1
            idxs = tuple(sorted(rng_obj.sample(range(len(pool)), k)))
            if idxs in seen_idx:
                continue
            seen_idx.add(idxs)
            sampled.append(tuple(pool[i] for i in idxs))
        return sampled

    def _pairs_from_spec(spec: str) -> List[Tuple[int, int]]:
        layer_heads = _parse_pair_combo_spec(spec)
        pairs: List[Tuple[int, int]] = []
        for layer_idx, heads_for_layer in layer_heads.items():
            for head_idx in heads_for_layer:
                pairs.append((int(layer_idx), int(head_idx)))
        return pairs

    def _scaled_target_counts(
        source_counts: Dict[int, int],
        *,
        total_target: int,
        keys: Sequence[int],
    ) -> Dict[int, int]:
        keys_list = [int(k) for k in keys]
        if not keys_list:
            return {}
        if total_target <= 0:
            return {k: 0 for k in keys_list}
        total_source = sum(int(source_counts.get(k, 0)) for k in keys_list)
        if total_source <= 0:
            base = total_target // len(keys_list)
            rem = total_target - base * len(keys_list)
            out = {k: base for k in keys_list}
            for idx in range(rem):
                out[keys_list[idx % len(keys_list)]] += 1
            return out

        raw: Dict[int, float] = {}
        floors: Dict[int, int] = {}
        fracs: List[Tuple[float, int]] = []
        floor_sum = 0
        for k in keys_list:
            value = total_target * (float(source_counts.get(k, 0)) / float(total_source))
            raw[k] = value
            floor_v = int(math.floor(value))
            floors[k] = floor_v
            floor_sum += floor_v
            fracs.append((value - float(floor_v), k))
        remaining = total_target - floor_sum
        fracs.sort(key=lambda x: x[0], reverse=True)
        for i in range(max(0, remaining)):
            floors[fracs[i % len(fracs)][1]] += 1
        return floors

    if high_candidate_count <= 0:
        high_candidate_count = max(2000, high_count * 50)
    high_candidate_count = max(high_candidate_count, high_count)
    candidate_pair_combos = _generate_candidate_pair_combos(
        top_pairs,
        k=pairs_per_combo,
        max_count=high_candidate_count,
        rng_obj=rng,
    )
    scored_candidates: List[Tuple[float, str]] = []
    for combo in candidate_pair_combos:
        combo_score = sum(pair_means.get(pair, 0.0) for pair in combo)
        scored_candidates.append((combo_score, _encode_pair_combo(combo)))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    high_specs: List[str] = []
    seen_high = set()
    for _, spec in scored_candidates:
        if spec in seen_high:
            continue
        seen_high.add(spec)
        high_specs.append(spec)
        if len(high_specs) >= high_count:
            break
    print(
        f"[info] Hybrid high candidates: {len(candidate_pair_combos)} "
        f"-> selected top {len(high_specs)} by sum(pair_mean)"
    )

    random_specs: List[str] = []
    excluded_specs = set(high_specs)
    if random_count > 0:
        random_rng = random.Random(None if seed is None else (seed + 17))
        random_candidate_count = max(high_candidate_count, max(3000, random_count * 120))
        candidate_random_combos = _generate_candidate_pair_combos(
            pair_pool,
            k=pairs_per_combo,
            max_count=random_candidate_count,
            rng_obj=random_rng,
        )

        candidate_items: List[Dict[str, Any]] = []
        seen_candidate_specs = set()
        for combo in candidate_random_combos:
            spec = _encode_pair_combo(combo)
            if spec in excluded_specs or spec in seen_candidate_specs:
                continue
            seen_candidate_specs.add(spec)
            layer_counts: Dict[int, int] = defaultdict(int)
            head_counts: Dict[int, int] = defaultdict(int)
            for layer_idx, head_idx in combo:
                layer_counts[int(layer_idx)] += 1
                head_counts[int(head_idx)] += 1
            candidate_items.append(
                {
                    "spec": spec,
                    "layer_counts": dict(layer_counts),
                    "head_counts": dict(head_counts),
                }
            )

        if high_specs:
            high_layer_counts: Dict[int, int] = defaultdict(int)
            high_head_counts: Dict[int, int] = defaultdict(int)
            for spec in high_specs:
                for layer_idx, head_idx in _pairs_from_spec(spec):
                    high_layer_counts[int(layer_idx)] += 1
                    high_head_counts[int(head_idx)] += 1

            random_total_pairs = random_count * pairs_per_combo
            target_layer_counts = _scaled_target_counts(
                dict(high_layer_counts),
                total_target=random_total_pairs,
                keys=layer_pool,
            )
            target_head_counts = _scaled_target_counts(
                dict(high_head_counts),
                total_target=random_total_pairs,
                keys=head_pool,
            )

            current_layer_counts: Dict[int, int] = defaultdict(int)
            current_head_counts: Dict[int, int] = defaultdict(int)
            used_idx = set()

            def _match_loss(item: Dict[str, Any]) -> int:
                layer_part = 0
                head_part = 0
                for layer_idx in layer_pool:
                    after = current_layer_counts.get(int(layer_idx), 0) + item["layer_counts"].get(int(layer_idx), 0)
                    layer_part += abs(after - target_layer_counts.get(int(layer_idx), 0))
                for head_idx in head_pool:
                    after = current_head_counts.get(int(head_idx), 0) + item["head_counts"].get(int(head_idx), 0)
                    head_part += abs(after - target_head_counts.get(int(head_idx), 0))
                return layer_part + head_part

            for _ in range(random_count):
                best_idx = -1
                best_loss: Optional[int] = None
                for idx, item in enumerate(candidate_items):
                    if idx in used_idx:
                        continue
                    loss = _match_loss(item)
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_idx = idx
                if best_idx < 0:
                    break
                used_idx.add(best_idx)
                chosen = candidate_items[best_idx]
                random_specs.append(str(chosen["spec"]))
                for layer_idx, add_v in chosen["layer_counts"].items():
                    current_layer_counts[int(layer_idx)] += int(add_v)
                for head_idx, add_v in chosen["head_counts"].items():
                    current_head_counts[int(head_idx)] += int(add_v)

            if random_specs:
                layer_gap = sum(abs(current_layer_counts.get(int(k), 0) - target_layer_counts.get(int(k), 0)) for k in layer_pool)
                head_gap = sum(abs(current_head_counts.get(int(k), 0) - target_head_counts.get(int(k), 0)) for k in head_pool)
                print(
                    f"[info] Random match gaps vs high-set histogram: "
                    f"layer_L1={layer_gap}, head_L1={head_gap}"
                )
        else:
            print("[warn] High set is empty; falling back to pure random for matched-random set.")

        if len(random_specs) < random_count:
            used_specs = set(random_specs) | excluded_specs
            remaining_specs = [str(item["spec"]) for item in candidate_items if str(item["spec"]) not in used_specs]
            random_rng.shuffle(remaining_specs)
            need = random_count - len(random_specs)
            random_specs.extend(remaining_specs[:need])

    eval_combos = [(spec, "pairs") for spec in (high_specs + random_specs)]
    if len(eval_combos) < combo_count:
        print(f"[warn] Hybrid selected {len(eval_combos)} combos (requested {combo_count}).")
    print(
        f"[info] Hybrid selected {len(eval_combos)} combos "
        f"({len(high_specs)} high-leverage + {len(random_specs)} matched-random)."
    )
    for idx, (heads_spec, layers_spec) in enumerate(eval_combos, 1):
        tag = "high" if idx <= len(high_specs) else "rand"
        print(f"  {idx:03d}. [{tag}] {heads_spec}")
    return eval_combos


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute probability deltas for baseline top beams.")
    parser.add_argument("--data-path", default=None, help="Path to dataset JSONL.")
    parser.add_argument("--target-id", default=None, help="Only process the record with this id (required).")
    parser.add_argument("--num-beams", type=int, default=3, help="Number of beam answers to return.")
    parser.add_argument("--max-length", type=int, default=10, help="Max generation length.")
    parser.add_argument("--min-length", type=int, default=1, help="Min generation length.")
    parser.add_argument("--heads", default="all", help="Head spec for override.")
    parser.add_argument("--layers", default="all", help="Layer spec for override.")
    parser.add_argument("--random-combos", type=int, default=0, help="Randomly sample N unique (heads,layers) combinations.")
    parser.add_argument("--random-head-count", type=int, default=0, help="Heads per random combo (drawn from --heads pool).")
    parser.add_argument("--random-layer-count", type=int, default=0, help="Layers per random combo (drawn from --layers pool).")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional RNG seed for random combo sampling.")
    parser.add_argument("--hybrid-combos", type=int, default=0, help="Hybrid sample N combos: high-leverage + matched-random.")
    parser.add_argument("--hybrid-high-frac", type=float, default=0.8, help="Fraction of hybrid combos chosen as high-leverage.")
    parser.add_argument("--hybrid-pairs-per-combo", type=int, default=4, help="Number of (layer,head) pairs in each hybrid combo.")
    parser.add_argument("--hybrid-top-pairs", type=int, default=24, help="Top crucial pairs considered for leverage combo construction.")
    parser.add_argument("--hybrid-high-candidates", type=int, default=5000, help="Candidate combo count built from top pairs before top-N selection.")
    parser.add_argument("--hybrid-seed", type=int, default=None, help="Optional RNG seed for hybrid combo sampling.")
    parser.add_argument("--hybrid-calib-size", type=int, default=8, help="Calibration record count for single head/layer scan.")
    parser.add_argument("--hybrid-fail-only", type=int, default=1, help="Use only records tagged as failures for calibration (1/0).")
    parser.add_argument("--hybrid-var-weight", type=float, default=0.5, help="Weight of effect std-dev in leverage score.")
    parser.add_argument(
        "--rankings-jsonl",
        default=None,
        help="Stage-1 rankings JSONL from save_ranked_pairs.py (supports {k} template).",
    )
    parser.add_argument(
        "--rankings-top-k",
        type=int,
        default=3,
        help="Value used to resolve {k} in --rankings-jsonl.",
    )
    parser.add_argument(
        "--ranked-combos",
        type=int,
        default=0,
        help="Sample N pair-combos from ranked_pairs for the target record.",
    )
    parser.add_argument("--ranked-top-min", type=int, default=25, help="Min pairs sampled from top ranked pool.")
    parser.add_argument("--ranked-top-max", type=int, default=30, help="Max pairs sampled from top ranked pool.")
    parser.add_argument("--ranked-next-window", type=int, default=60, help="Secondary pool size following top pool.")
    parser.add_argument("--ranked-next-min", type=int, default=30, help="Min pairs sampled from secondary pool.")
    parser.add_argument("--ranked-next-max", type=int, default=40, help="Max pairs sampled from secondary pool.")
    parser.add_argument("--ranked-seed", type=int, default=0, help="Optional RNG seed for ranked combo sampling.")
    parser.add_argument(
        "--hist-mode",
        default="both",
        choices=["delta", "after", "both"],
        help="Deprecated. Ignored; script always plots both after and delta together.",
    )
    parser.add_argument(
        "--matrix-out",
        default="probs/dataset/rand/{target_id}/matrix_heads-{heads}_layers-{layers}.csv",
        help="Output path template for probability matrix CSV (dir or filename).",
    )
    parser.add_argument(
        "--aggregate-stability-only",
        action="store_true",
        help="Skip model inference and aggregate stability from existing candidate_stability_features JSONL files.",
    )
    parser.add_argument(
        "--aggregate-stability-jsonl",
        action="append",
        default=[],
        help="Input candidate_stability_features JSONL file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--aggregate-stability-glob",
        default=None,
        help="Glob pattern for candidate_stability_features JSONL files (supports recursive '**').",
    )
    parser.add_argument(
        "--aggregate-stability-out",
        default="probs/stability_aggregate",
        help="Output directory for aggregate-only stability artifacts.",
    )
    parser.add_argument(
        "--stability-plot-style",
        default="violin",
        choices=["violin", "hist"],
        help="Plot style for GT vs non-GT stability diagrams.",
    )
    parser.add_argument(
        "--include-new-answer-stability",
        action="store_true",
        help=(
            "Also compute candidate-stability features for answers that newly appear in post-override "
            "top beams (not only baseline candidates)."
        ),
    )
    args = parser.parse_args(argv)

    if args.aggregate_stability_only:
        feature_paths = _resolve_aggregate_feature_paths(
            jsonl_args=[str(v) for v in (args.aggregate_stability_jsonl or [])],
            glob_pattern=args.aggregate_stability_glob,
        )
        if not feature_paths:
            parser.error(
                "--aggregate-stability-only requires --aggregate-stability-jsonl and/or --aggregate-stability-glob."
            )
        return _run_stability_aggregation_from_files(
            feature_paths=feature_paths,
            out_dir=Path(args.aggregate_stability_out),
            plot_style=str(args.stability_plot_style),
        )

    if not args.target_id:
        parser.error("--target-id is required and must match exactly one record.")
    if args.hist_mode != "both":
        print(f"[info] --hist-mode={args.hist_mode} is ignored; plotting both after and delta in one figure.")
    hybrid_mode = args.hybrid_combos > 0
    random_mode = args.random_combos > 0
    ranked_mode = args.ranked_combos > 0
    if args.rankings_jsonl and not ranked_mode:
        parser.error("--rankings-jsonl requires --ranked-combos > 0.")
    enabled_modes = int(hybrid_mode) + int(random_mode) + int(ranked_mode)
    if enabled_modes > 1:
        parser.error("--hybrid-combos, --random-combos, and --ranked-combos are mutually exclusive.")
    if hybrid_mode:
        if args.hybrid_pairs_per_combo <= 0:
            parser.error("--hybrid-pairs-per-combo must be > 0 when --hybrid-combos is set.")
        if args.hybrid_high_frac < 0.0 or args.hybrid_high_frac > 1.0:
            parser.error("--hybrid-high-frac must be within [0, 1].")
    if random_mode:
        if args.random_head_count <= 0:
            parser.error("--random-head-count must be > 0 when --random-combos is set.")
        if args.random_layer_count <= 0:
            parser.error("--random-layer-count must be > 0 when --random-combos is set.")
    if ranked_mode:
        if not args.rankings_jsonl:
            parser.error("--rankings-jsonl is required when --ranked-combos is set.")
        if args.ranked_top_min <= 0 or args.ranked_top_max <= 0:
            parser.error("--ranked-top-min and --ranked-top-max must be > 0.")
        if args.ranked_next_min <= 0 or args.ranked_next_max <= 0:
            parser.error("--ranked-next-min and --ranked-next-max must be > 0.")

    dataset_path = data_path
    if args.data_path:
        dataset_path = Path(args.data_path)

    records = list(iter_jsonl(dataset_path))
    if not records:
        print(f"No records to evaluate in {dataset_path}.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    if hybrid_mode:
        eval_combos = _build_hybrid_eval_combos(
            args=args,
            records=records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            total_layers=total_layers,
            originals=originals,
        )
    elif random_mode:
        head_pool = _resolve_heads(args.heads)
        layer_pool = _resolve_layers(args.layers, total_layers)
        eval_combos = _sample_random_head_layer_specs(
            head_pool,
            layer_pool,
            head_count=args.random_head_count,
            layer_count=args.random_layer_count,
            combo_count=args.random_combos,
            seed=args.random_seed,
        )
        print(f"[info] Randomly selected {len(eval_combos)} (heads,layers) combos:")
        for idx, (heads_spec, layers_spec) in enumerate(eval_combos, 1):
            print(f"  {idx}. heads={heads_spec} | layers={layers_spec}")
    elif ranked_mode:
        rankings_path = _resolve_topk_path_template(str(args.rankings_jsonl), int(args.rankings_top_k))
        ranked_pairs = _load_ranked_pairs_for_target(
            rankings_path,
            target_id=args.target_id,
        )
        combo_specs = _sample_ranked_pair_combo_specs(
            ranked_pairs,
            combo_count=int(args.ranked_combos),
            top_pick_min=int(args.ranked_top_min),
            top_pick_max=int(args.ranked_top_max),
            next_window=int(args.ranked_next_window),
            next_pick_min=int(args.ranked_next_min),
            next_pick_max=int(args.ranked_next_max),
            seed=args.ranked_seed,
        )
        if not combo_specs:
            raise ValueError(
                "No ranked combos sampled. Increase ranked pool settings or check ranked_pairs availability."
            )
        eval_combos = [(spec, "pairs") for spec in combo_specs]
        print(f"[info] Ranked-pair selected {len(eval_combos)} combos from {rankings_path}:")
        for idx, (heads_spec, _) in enumerate(eval_combos, 1):
            print(f"  {idx:03d}. {heads_spec}")
    else:
        eval_combos = [(args.heads, args.layers)]

    target_id_norm = str(args.target_id).strip().lower()

    cached_records: List[Dict[str, Any]] = []
    for entry in records:
        rec_id_val = entry.get("id")
        rec_id = "" if rec_id_val is None else str(rec_id_val)
        rec_id_norm = rec_id.strip().lower()
        if target_id_norm and rec_id_norm != target_id_norm:
            continue
        cached = _build_cached_record(
            entry,
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
        )
        if cached:
            cached_records.append(cached)


    if not cached_records:
        raise ValueError(f"No record found for --target-id {args.target_id!r}.")
    cached_records_by_id: Dict[str, Dict[str, Any]] = {
        str(rec.get("record_id", "")): rec for rec in cached_records
    }

    prob_deltas_by_rank: Dict[int, List[float]] = {}
    beam_deltas_by_rank: Dict[int, List[float]] = {}
    prob_after_by_rank: Dict[int, List[float]] = {}
    beam_after_by_rank: Dict[int, List[float]] = {}
    sampled_mode = random_mode or hybrid_mode or ranked_mode
    random_total_trials = 0
    random_total_hits = 0
    overall_accuracy_note: Optional[str] = None
    gt_delta_logprob_std_vals: List[float] = []
    non_gt_delta_logprob_std_vals: List[float] = []
    gt_after_prob_std_vals: List[float] = []
    non_gt_after_prob_std_vals: List[float] = []
    gt_p_pos_vals: List[float] = []
    non_gt_p_pos_vals: List[float] = []
    candidate_feature_rows: List[Dict[str, Any]] = []
    candidate_series: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    dynamic_candidate_feature_rows: List[Dict[str, Any]] = []
    dynamic_candidate_series: Dict[Tuple[str, str], Dict[str, Any]] = {}
    top1_change_feature_rows: List[Dict[str, Any]] = []
    top1_series_by_record: Dict[str, Dict[str, Any]] = {}

    for combo_idx, (heads_spec, layers_spec) in enumerate(eval_combos, 1):
        if hybrid_mode:
            if not heads_spec.startswith("pairs:"):
                raise ValueError(f"Hybrid mode expected pair-combo spec, got: {heads_spec!r}")
            pair_layer_heads = _parse_pair_combo_spec(heads_spec)
            if not pair_layer_heads:
                raise ValueError(f"Hybrid mode got empty/invalid pair-combo spec: {heads_spec!r}")
            target_layers = tuple(sorted(pair_layer_heads.keys()))
            heads_arg = ()
            is_pair_combo = True
        else:
            is_pair_combo = heads_spec.startswith("pairs:")
            pair_layer_heads = _parse_pair_combo_spec(heads_spec) if is_pair_combo else {}
            target_layers = _resolve_layers(layers_spec, total_layers) if not is_pair_combo else tuple(sorted(pair_layer_heads.keys()))
            heads = _resolve_heads(heads_spec) if not is_pair_combo else ()
            heads_arg = heads[0] if (not is_pair_combo and isinstance(heads, (list, tuple)) and len(heads) == 1) else heads
        combo_acc_trials = 0
        combo_acc_hits = 0

        for record in cached_records:
            applied_layers: Tuple[int, ...] = ()
            candidate_prob_before_scores: List[float] = []
            candidate_prob_after_scores: List[float] = []
            after_logprob_by_norm: Dict[str, float] = {}
            if is_pair_combo:
                selected_layers: List[int] = []
                for layer_idx in sorted(pair_layer_heads.keys()):
                    heads_for_layer = pair_layer_heads.get(layer_idx, ())
                    if not heads_for_layer:
                        continue
                    layer_heads_arg: Any = heads_for_layer[0] if len(heads_for_layer) == 1 else heads_for_layer
                    new_forward = make_forward(layer_heads_arg, record["override_rows"], LAMBDA=1.0)
                    apply_override(new_forward, (layer_idx,), originals)
                    selected_layers.append(layer_idx)
                applied_layers = tuple(selected_layers)
            else:
                new_forward = make_forward(heads_arg, record["override_rows"], LAMBDA=1.0)
                apply_override(new_forward, target_layers, originals)
                applied_layers = target_layers
            try:
                for beam in record["baseline_beams"]:
                    report_after = compute_answer_probability(
                        model,
                        tokenizer,
                        record["image"],
                        record["question_inputs"],
                        beam["answer"],
                        image_embeds=record["image_embeds"],
                        image_att_mask=record["image_att_mask"],
                        num_beams=args.num_beams,
                    )
                    rank = int(beam["rank"])
                    prob_after_by_rank.setdefault(rank, []).append(report_after.prob)
                    beam_after_by_rank.setdefault(rank, []).append(report_after.beam_score)
                    prob_delta = report_after.prob - beam["prob_before"]
                    beam_delta = report_after.beam_score - beam["beam_score"]
                    prob_deltas_by_rank.setdefault(rank, []).append(prob_delta)
                    beam_deltas_by_rank.setdefault(rank, []).append(beam_delta)
                    candidate_prob_before_scores.append(float(beam["prob_before"]))
                    candidate_prob_after_scores.append(float(report_after.prob))
                    beam_answer_norm = _answer_key(str(beam.get("answer", "")))
                    if beam_answer_norm:
                        after_logprob_by_norm[beam_answer_norm] = float(report_after.prob)

                gold_norm = str(record.get("gold_answer_norm", "") or "")
                after_beams = _generate_top_beams(
                    model,
                    tokenizer,
                    record["image"],
                    record["question"],
                    num_beams=args.num_beams,
                    max_length=args.max_length,
                    min_length=args.min_length,
                )
                pred_after = str(after_beams[0][0]) if after_beams else ""
                pred_norm = _answer_key(pred_after)
                if sampled_mode and gold_norm:
                    combo_acc_trials += 1
                    if pred_norm == gold_norm:
                        combo_acc_hits += 1

                rec_key = str(record.get("record_id", ""))
                baseline_top1_answer = str(record["baseline_beams"][0].get("answer", "")) if record["baseline_beams"] else ""
                baseline_top1_norm = _answer_key(baseline_top1_answer)
                if rec_key not in top1_series_by_record:
                    top1_series_by_record[rec_key] = {
                        "record_id": rec_key,
                        "gold_answer_norm": gold_norm,
                        "baseline_top1_answer": baseline_top1_answer,
                        "baseline_top1_answer_norm": baseline_top1_norm,
                        "top1_after_answer_values": [],
                        "top1_after_answer_norm_values": [],
                        "top1_top2_margin_beam_values": [],
                        "top1_top2_margin_prob_values": [],
                        "combo_indices": [],
                    }
                rec_obs = top1_series_by_record[rec_key]
                rec_obs["top1_after_answer_values"].append(pred_after)
                rec_obs["top1_after_answer_norm_values"].append(pred_norm)
                rec_obs["combo_indices"].append(int(combo_idx))
                after_beam_scores = [float(score) for _, score in after_beams]
                if len(after_beam_scores) >= 2:
                    top1_top2_margin_beam = float(after_beam_scores[0] - after_beam_scores[1])
                    after_beam_probs = _scores_to_probs(after_beam_scores)
                    top1_top2_margin_prob = float(after_beam_probs[0] - after_beam_probs[1]) if len(after_beam_probs) >= 2 else 0.0
                    rec_obs["top1_top2_margin_beam_values"].append(top1_top2_margin_beam)
                    rec_obs["top1_top2_margin_prob_values"].append(top1_top2_margin_prob)

                if args.include_new_answer_stability and after_beams:
                    per_combo_entries: List[Tuple[int, str, str, float]] = []
                    seen_after_norms: set = set()
                    for after_rank, (after_answer_raw, _after_score) in enumerate(after_beams, 1):
                        after_answer = str(after_answer_raw)
                        after_norm = _answer_key(after_answer)
                        if not after_norm or after_norm in seen_after_norms:
                            continue
                        seen_after_norms.add(after_norm)
                        if after_norm in after_logprob_by_norm:
                            after_logprob = float(after_logprob_by_norm[after_norm])
                        else:
                            report_new_after = compute_answer_probability(
                                model,
                                tokenizer,
                                record["image"],
                                record["question_inputs"],
                                after_answer,
                                image_embeds=record["image_embeds"],
                                image_att_mask=record["image_att_mask"],
                                num_beams=args.num_beams,
                            )
                            after_logprob = float(report_new_after.prob)
                            after_logprob_by_norm[after_norm] = after_logprob
                        per_combo_entries.append((int(after_rank), after_answer, after_norm, after_logprob))

                    per_combo_probs = _scores_to_probs([entry[3] for entry in per_combo_entries])
                    for entry_idx, (after_rank, after_answer, after_norm, after_logprob) in enumerate(per_combo_entries):
                        key_dyn = (rec_key, after_norm)
                        is_gt_candidate_dyn: Optional[int] = None
                        if gold_norm:
                            is_gt_candidate_dyn = int(after_norm == gold_norm)
                        if key_dyn not in dynamic_candidate_series:
                            dynamic_candidate_series[key_dyn] = {
                                "record_id": rec_key,
                                "candidate_rank": -1,
                                "candidate_answer": after_answer,
                                "candidate_answer_norm": after_norm,
                                "gold_answer_norm": str(gold_norm),
                                "is_gt_candidate": is_gt_candidate_dyn,
                                "before_logprob": None,
                                "num_baseline_beams": int(len(record.get("baseline_beams", []))),
                                "after_logprob_values": [],
                                "after_prob_values": [],
                                "after_rank_values": [],
                                "combo_indices": [],
                            }
                        dyn_obs = dynamic_candidate_series[key_dyn]
                        dyn_obs["after_logprob_values"].append(float(after_logprob))
                        dyn_obs["after_prob_values"].append(
                            float(per_combo_probs[entry_idx]) if entry_idx < len(per_combo_probs) else 0.0
                        )
                        dyn_obs["after_rank_values"].append(int(after_rank))
                        dyn_obs["combo_indices"].append(int(combo_idx))

                candidate_after_probs = _scores_to_probs(candidate_prob_after_scores)
                for idx, beam in enumerate(record["baseline_beams"]):
                    before_score = float(candidate_prob_before_scores[idx]) if idx < len(candidate_prob_before_scores) else 0.0
                    after_score = float(candidate_prob_after_scores[idx]) if idx < len(candidate_prob_after_scores) else 0.0
                    after_prob = float(candidate_after_probs[idx]) if idx < len(candidate_after_probs) else 0.0
                    answer_text = str(beam.get("answer", ""))
                    answer_norm = _answer_key(answer_text)
                    is_gt_candidate: Optional[bool] = None
                    if gold_norm:
                        is_gt_candidate = bool(answer_norm == gold_norm)
                    candidate_rank = int(beam.get("rank", idx + 1))
                    key = (rec_key, candidate_rank, answer_norm)
                    if key not in candidate_series:
                        candidate_series[key] = {
                            "record_id": rec_key,
                            "candidate_rank": candidate_rank,
                            "candidate_answer": answer_text,
                            "candidate_answer_norm": answer_norm,
                            "gold_answer_norm": str(gold_norm),
                            "is_gt_candidate": None if is_gt_candidate is None else int(is_gt_candidate),
                            "before_logprob": float(before_score),
                            "num_baseline_beams": int(len(candidate_prob_after_scores)),
                            "after_logprob_values": [],
                            "after_prob_values": [],
                            "combo_indices": [],
                        }
                    candidate_series[key]["after_logprob_values"].append(float(after_score))
                    candidate_series[key]["after_prob_values"].append(float(after_prob))
                    candidate_series[key]["combo_indices"].append(int(combo_idx))
            finally:
                if applied_layers:
                    revert_override(applied_layers, originals)
        if sampled_mode:
            if combo_acc_trials > 0:
                random_total_trials += combo_acc_trials
                random_total_hits += combo_acc_hits

    for obs in candidate_series.values():
        stability = _compute_candidate_stability_features(
            before_logprob=float(obs.get("before_logprob", 0.0)),
            after_logprobs=obs.get("after_logprob_values", []),
            after_probs=obs.get("after_prob_values", []),
        )
        candidate_row = {
            "record_id": str(obs.get("record_id", "")),
            "candidate_rank": int(obs.get("candidate_rank", 0)),
            "candidate_answer": str(obs.get("candidate_answer", "")),
            "candidate_answer_norm": str(obs.get("candidate_answer_norm", "")),
            "gold_answer_norm": str(obs.get("gold_answer_norm", "")),
            "is_gt_candidate": obs.get("is_gt_candidate", None),
            "before_logprob": float(obs.get("before_logprob", 0.0)),
            "num_baseline_beams": int(obs.get("num_baseline_beams", 0)),
            "num_combo_observations": int(len(obs.get("after_logprob_values", []))),
            "combo_indices": list(obs.get("combo_indices", [])),
            "after_logprob_values": [float(v) for v in obs.get("after_logprob_values", [])],
            "after_prob_values": [float(v) for v in obs.get("after_prob_values", [])],
            "delta_logprob_mean_across_combos": float(stability["delta_logprob_mean_across_combos"]),
            "delta_logprob_std_across_combos": float(stability["delta_logprob_std_across_combos"]),
            "after_prob_mean_across_combos": float(stability["after_prob_mean_across_combos"]),
            "after_prob_std_across_combos": float(stability["after_prob_std_across_combos"]),
            "p_pos_across_combos": float(stability["p_pos_across_combos"]),
        }
        candidate_feature_rows.append(candidate_row)

        is_gt = candidate_row.get("is_gt_candidate")
        if is_gt == 1:
            gt_delta_logprob_std_vals.append(float(stability["delta_logprob_std_across_combos"]))
            gt_after_prob_std_vals.append(float(stability["after_prob_std_across_combos"]))
            gt_p_pos_vals.append(float(stability["p_pos_across_combos"]))
        elif is_gt == 0:
            non_gt_delta_logprob_std_vals.append(float(stability["delta_logprob_std_across_combos"]))
            non_gt_after_prob_std_vals.append(float(stability["after_prob_std_across_combos"]))
            non_gt_p_pos_vals.append(float(stability["p_pos_across_combos"]))

    if args.include_new_answer_stability and dynamic_candidate_series:
        for obs in dynamic_candidate_series.values():
            rec_id = str(obs.get("record_id", ""))
            rec = cached_records_by_id.get(rec_id)
            if rec is None:
                continue
            candidate_answer = str(obs.get("candidate_answer", ""))
            report_before_dyn = compute_answer_probability(
                model,
                tokenizer,
                rec["image"],
                rec["question_inputs"],
                candidate_answer,
                image_embeds=rec["image_embeds"],
                image_att_mask=rec["image_att_mask"],
                num_beams=args.num_beams,
            )
            obs["before_logprob"] = float(report_before_dyn.prob)

        for obs in dynamic_candidate_series.values():
            before_logprob_val = obs.get("before_logprob", None)
            if before_logprob_val is None:
                continue
            stability_dyn = _compute_candidate_stability_features(
                before_logprob=float(before_logprob_val),
                after_logprobs=obs.get("after_logprob_values", []),
                after_probs=obs.get("after_prob_values", []),
            )
            after_rank_values = [int(v) for v in obs.get("after_rank_values", [])]
            mean_after_rank = (
                float(np.mean(np.asarray(after_rank_values, dtype=np.float64))) if after_rank_values else float("nan")
            )
            min_after_rank = int(min(after_rank_values)) if after_rank_values else -1
            dynamic_row = {
                "record_id": str(obs.get("record_id", "")),
                "candidate_rank": int(obs.get("candidate_rank", -1)),
                "candidate_answer": str(obs.get("candidate_answer", "")),
                "candidate_answer_norm": str(obs.get("candidate_answer_norm", "")),
                "gold_answer_norm": str(obs.get("gold_answer_norm", "")),
                "is_gt_candidate": obs.get("is_gt_candidate", None),
                "candidate_source": "after_beams_dynamic",
                "before_logprob": float(before_logprob_val),
                "num_baseline_beams": int(obs.get("num_baseline_beams", 0)),
                "num_combo_observations": int(len(obs.get("after_logprob_values", []))),
                "combo_indices": list(obs.get("combo_indices", [])),
                "after_logprob_values": [float(v) for v in obs.get("after_logprob_values", [])],
                "after_prob_values": [float(v) for v in obs.get("after_prob_values", [])],
                "mean_after_rank_across_combos": mean_after_rank,
                "min_after_rank_across_combos": min_after_rank,
                "delta_logprob_mean_across_combos": float(stability_dyn["delta_logprob_mean_across_combos"]),
                "delta_logprob_std_across_combos": float(stability_dyn["delta_logprob_std_across_combos"]),
                "after_prob_mean_across_combos": float(stability_dyn["after_prob_mean_across_combos"]),
                "after_prob_std_across_combos": float(stability_dyn["after_prob_std_across_combos"]),
                "p_pos_across_combos": float(stability_dyn["p_pos_across_combos"]),
            }
            dynamic_candidate_feature_rows.append(dynamic_row)

    for obs in top1_series_by_record.values():
        top1_norm_values = [str(v) for v in obs.get("top1_after_answer_norm_values", [])]
        baseline_top1_norm = str(obs.get("baseline_top1_answer_norm", ""))
        gold_norm_for_record = str(obs.get("gold_answer_norm", ""))
        baseline_top1_is_gt: Optional[int] = None
        top1_after_is_gt_count: Optional[int] = None
        top1_after_is_gt_rate: Optional[float] = None
        top1_after_is_gt_majority: Optional[int] = None
        if gold_norm_for_record:
            baseline_top1_is_gt = int(baseline_top1_norm == gold_norm_for_record)
            top1_after_is_gt_count = int(sum(1 for v in top1_norm_values if v == gold_norm_for_record))
            n_obs_tmp = int(len(top1_norm_values))
            top1_after_is_gt_rate = float(top1_after_is_gt_count / n_obs_tmp) if n_obs_tmp > 0 else 0.0
            top1_after_is_gt_majority = int(top1_after_is_gt_rate >= 0.5) if n_obs_tmp > 0 else None
        n_obs = int(len(top1_norm_values))
        changed_from_baseline_count = (
            int(sum(1 for v in top1_norm_values if v != baseline_top1_norm)) if baseline_top1_norm else 0
        )
        changed_from_baseline_rate = float(changed_from_baseline_count / n_obs) if n_obs > 0 else 0.0
        switch_count = int(sum(1 for i in range(1, n_obs) if top1_norm_values[i] != top1_norm_values[i - 1]))
        switch_rate = float(switch_count / (n_obs - 1)) if n_obs > 1 else 0.0

        margin_beam_values = [float(v) for v in obs.get("top1_top2_margin_beam_values", [])]
        margin_prob_values = [float(v) for v in obs.get("top1_top2_margin_prob_values", [])]
        margin_beam_mean = float(np.mean(np.asarray(margin_beam_values, dtype=np.float64))) if margin_beam_values else 0.0
        margin_beam_std = float(np.std(np.asarray(margin_beam_values, dtype=np.float64))) if margin_beam_values else 0.0
        margin_prob_mean = float(np.mean(np.asarray(margin_prob_values, dtype=np.float64))) if margin_prob_values else 0.0
        margin_prob_std = float(np.std(np.asarray(margin_prob_values, dtype=np.float64))) if margin_prob_values else 0.0

        top1_change_feature_rows.append(
            {
                "record_id": str(obs.get("record_id", "")),
                "gold_answer_norm": str(obs.get("gold_answer_norm", "")),
                "baseline_top1_answer": str(obs.get("baseline_top1_answer", "")),
                "baseline_top1_answer_norm": baseline_top1_norm,
                "baseline_top1_is_gt": baseline_top1_is_gt,
                "num_combo_observations": n_obs,
                "top1_after_is_gt_count": top1_after_is_gt_count,
                "top1_after_is_gt_rate_across_combos": top1_after_is_gt_rate,
                "top1_after_is_gt_majority": top1_after_is_gt_majority,
                "top1_changed_from_baseline_count": changed_from_baseline_count,
                "top1_changed_from_baseline_rate": changed_from_baseline_rate,
                "top1_switch_count_between_combos": switch_count,
                "top1_switch_rate_between_combos": switch_rate,
                "num_margin_observations": int(len(margin_beam_values)),
                "top1_top2_margin_beam_mean_across_combos": margin_beam_mean,
                "top1_top2_margin_beam_std_across_combos": margin_beam_std,
                "top1_top2_margin_prob_mean_across_combos": margin_prob_mean,
                "top1_top2_margin_prob_std_across_combos": margin_prob_std,
                "combo_indices": list(obs.get("combo_indices", [])),
            }
        )

    if sampled_mode:
        if hybrid_mode:
            mode_tag = "hybrid"
        elif random_mode:
            mode_tag = "random"
        else:
            mode_tag = "ranked"
        if random_total_trials > 0:
            random_overall_acc = 100.0 * random_total_hits / random_total_trials
            overall_accuracy_note = (
                f"overall_accuracy: {random_total_hits}/{random_total_trials} "
                f"({random_overall_acc:.2f}%) across {len(eval_combos)} combos"
            )
            print(
                f"[{mode_tag}][overall] accuracy across {len(eval_combos)} combos: "
                f"{random_total_hits}/{random_total_trials} ({random_overall_acc:.2f}%)"
            )
        else:
            overall_accuracy_note = f"overall_accuracy: N/A across {len(eval_combos)} combos (missing gold answers)"
            print(f"[{mode_tag}][overall] accuracy across {len(eval_combos)} combos: N/A (missing gold answers)")

    if hybrid_mode:
        out_heads_label = f"hybrid-{args.hybrid_combos}x{args.hybrid_pairs_per_combo}"
        out_layers_label = f"pairs-top{args.hybrid_top_pairs}"
    elif ranked_mode:
        out_heads_label = f"ranked-{args.ranked_combos}"
        out_layers_label = (
            f"top{args.ranked_top_min}-{args.ranked_top_max}"
            f"_next{args.ranked_next_min}-{args.ranked_next_max}_w{args.ranked_next_window}"
        )
    elif random_mode:
        out_heads_label = f"random-{args.random_combos}x{args.random_head_count}"
        out_layers_label = f"random-{args.random_combos}x{args.random_layer_count}"
    else:
        out_heads_label = eval_combos[0][0]
        out_layers_label = eval_combos[0][1]
    out_dir = _resolve_output_dir(args.matrix_out, out_heads_label, out_layers_label, args.target_id)
    baseline_answers_by_rank: Dict[int, str] = {}
    gt_answer_norm = ""
    if cached_records:
        gt_answer_norm = str(cached_records[0].get("gold_answer_norm", "") or "")
        for beam in cached_records[0]["baseline_beams"]:
            baseline_answers_by_rank[int(beam["rank"])] = str(beam["answer"])

    ranks = sorted(
        set(prob_after_by_rank.keys())
        | set(prob_deltas_by_rank.keys())
        | set(beam_after_by_rank.keys())
        | set(beam_deltas_by_rank.keys())
    )

    for rank in ranks:
        answer_text = baseline_answers_by_rank.get(rank, "")
        if answer_text:
            is_gt = bool(gt_answer_norm) and (_answer_key(answer_text) == gt_answer_norm)
            gt_mark = " *GT" if is_gt else ""
            answer_label = f" | {answer_text}{gt_mark}"
        else:
            answer_label = ""
        if hybrid_mode:
            combined_name = (
                f"seed_{args.hybrid_seed}_beam{rank}_pairs{args.hybrid_pairs_per_combo}_hybrid.pdf"
            )
        elif ranked_mode:
            combined_name = (
                f"seed_{args.ranked_seed}_beam{rank}_ranked.pdf"
            )
        elif random_mode:
            combined_name = (
                f"seed_{args.random_seed}_beam{rank}_h{args.random_head_count}_l{args.random_layer_count}_random.pdf"
            )
        else:
            combined_name = f"beam{rank}_combined_after_delta.pdf"
        combined_path = out_dir / combined_name
        _plot_rank_combined_hist(
            prob_after=prob_after_by_rank.get(rank, []),
            prob_delta=prob_deltas_by_rank.get(rank, []),
            beam_after=beam_after_by_rank.get(rank, []),
            beam_delta=beam_deltas_by_rank.get(rank, []),
            out_path=combined_path,
            rank=rank,
            answer_label=answer_label,
            overall_note=overall_accuracy_note,
        )

    if candidate_feature_rows:
        features_jsonl = out_dir / "candidate_stability_features.jsonl"
        with features_jsonl.open("w", encoding="utf-8") as f:
            for row in candidate_feature_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[info] Wrote per-candidate stability features to {features_jsonl}")

    if dynamic_candidate_feature_rows:
        dynamic_features_jsonl = out_dir / "candidate_stability_features_new_answers.jsonl"
        with dynamic_features_jsonl.open("w", encoding="utf-8") as f:
            for row in dynamic_candidate_feature_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[info] Wrote dynamic new-answer stability features to {dynamic_features_jsonl}")

    if top1_change_feature_rows:
        top1_features_jsonl = out_dir / "top1_margin_change_features.jsonl"
        with top1_features_jsonl.open("w", encoding="utf-8") as f:
            for row in top1_change_feature_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[info] Wrote top1-change and top1-top2 margin features to {top1_features_jsonl}")

    stability_plot_path = out_dir / "candidate_stability_gt_vs_non_gt.pdf"
    _plot_gt_vs_non_gt_stability(
        gt_delta_logprob_std=gt_delta_logprob_std_vals,
        non_gt_delta_logprob_std=non_gt_delta_logprob_std_vals,
        gt_after_prob_std=gt_after_prob_std_vals,
        non_gt_after_prob_std=non_gt_after_prob_std_vals,
        gt_p_pos=gt_p_pos_vals,
        non_gt_p_pos=non_gt_p_pos_vals,
        out_path=stability_plot_path,
        plot_style=str(args.stability_plot_style),
        overall_note=overall_accuracy_note,
    )
    style_label = "histogram" if str(args.stability_plot_style) == "hist" else "violin+box"
    print(f"[info] Wrote GT vs non-GT stability {style_label} distributions to {stability_plot_path}")

    if candidate_feature_rows:
        stability_plot_extended_path = out_dir / "candidate_stability_gt_vs_non_gt_extended.pdf"
        _plot_gt_vs_non_gt_metric_grid(
            rows=candidate_feature_rows,
            metric_specs=BASELINE_STABILITY_METRIC_SPECS,
            out_path=stability_plot_extended_path,
            plot_style=str(args.stability_plot_style),
            overall_note=overall_accuracy_note,
            group_key="is_gt_candidate",
            gt_label="GT",
            non_gt_label="non-GT",
        )
        print(
            f"[info] Wrote extended GT vs non-GT candidate stability {style_label} distributions to "
            f"{stability_plot_extended_path}"
        )

    if dynamic_candidate_feature_rows:
        stability_dynamic_plot_extended_path = out_dir / "candidate_stability_new_answers_gt_vs_non_gt_extended.pdf"
        _plot_gt_vs_non_gt_metric_grid(
            rows=dynamic_candidate_feature_rows,
            metric_specs=DYNAMIC_STABILITY_METRIC_SPECS,
            out_path=stability_dynamic_plot_extended_path,
            plot_style=str(args.stability_plot_style),
            overall_note=overall_accuracy_note,
            group_key="is_gt_candidate",
            gt_label="GT",
            non_gt_label="non-GT",
        )
        print(
            f"[info] Wrote extended GT vs non-GT dynamic-answer stability {style_label} distributions to "
            f"{stability_dynamic_plot_extended_path}"
        )

    if top1_change_feature_rows:
        top1_plot_extended_path = out_dir / "top1_metrics_after_gt_vs_non_gt_extended.pdf"
        _plot_gt_vs_non_gt_metric_grid(
            rows=top1_change_feature_rows,
            metric_specs=TOP1_METRIC_SPECS,
            out_path=top1_plot_extended_path,
            plot_style=str(args.stability_plot_style),
            overall_note=overall_accuracy_note,
            group_key="top1_after_is_gt_majority",
            gt_label="after top1 mostly=GT",
            non_gt_label="after top1 mostly!=GT",
        )
        print(
            f"[info] Wrote extended GT vs non-GT top1 metric (grouped by after-top1 correctness) {style_label} distributions to "
            f"{top1_plot_extended_path}"
        )

    gt_count = len(gt_delta_logprob_std_vals)
    non_gt_count = len(non_gt_delta_logprob_std_vals)
    if gt_count or non_gt_count:
        gt_delta_std_mean = (
            float(np.mean(np.asarray(gt_delta_logprob_std_vals, dtype=np.float64))) if gt_count else float("nan")
        )
        non_gt_delta_std_mean = (
            float(np.mean(np.asarray(non_gt_delta_logprob_std_vals, dtype=np.float64))) if non_gt_count else float("nan")
        )
        gt_after_prob_std_mean = (
            float(np.mean(np.asarray(gt_after_prob_std_vals, dtype=np.float64))) if gt_count else float("nan")
        )
        non_gt_after_prob_std_mean = (
            float(np.mean(np.asarray(non_gt_after_prob_std_vals, dtype=np.float64))) if non_gt_count else float("nan")
        )
        gt_p_pos_mean = float(np.mean(np.asarray(gt_p_pos_vals, dtype=np.float64))) if gt_p_pos_vals else float("nan")
        non_gt_p_pos_mean = float(np.mean(np.asarray(non_gt_p_pos_vals, dtype=np.float64))) if non_gt_p_pos_vals else float("nan")
        print(
            "[stability] "
            f"GT n={gt_count}, non-GT n={non_gt_count} | "
            f"delta_logprob_std mean GT={gt_delta_std_mean:+.6f} vs non-GT={non_gt_delta_std_mean:+.6f} | "
            f"after_prob_std mean GT={gt_after_prob_std_mean:+.6f} vs non-GT={non_gt_after_prob_std_mean:+.6f} | "
            f"p_pos mean GT={gt_p_pos_mean:+.6f} vs non-GT={non_gt_p_pos_mean:+.6f}"
        )
    if dynamic_candidate_feature_rows:
        dyn_gt = [row for row in dynamic_candidate_feature_rows if row.get("is_gt_candidate", None) == 1]
        dyn_non_gt = [row for row in dynamic_candidate_feature_rows if row.get("is_gt_candidate", None) == 0]
        dyn_delta_std_mean = float(
            np.mean(np.asarray([float(r["delta_logprob_std_across_combos"]) for r in dynamic_candidate_feature_rows], dtype=np.float64))
        )
        dyn_after_prob_std_mean = float(
            np.mean(np.asarray([float(r["after_prob_std_across_combos"]) for r in dynamic_candidate_feature_rows], dtype=np.float64))
        )
        dyn_mean_after_rank = float(
            np.nanmean(np.asarray([float(r["mean_after_rank_across_combos"]) for r in dynamic_candidate_feature_rows], dtype=np.float64))
        )
        print(
            "[stability][new-answers] "
            f"rows={len(dynamic_candidate_feature_rows)} | "
            f"GT rows={len(dyn_gt)}, non-GT rows={len(dyn_non_gt)} | "
            f"delta_logprob_std mean={dyn_delta_std_mean:+.6f} | "
            f"after_prob_std mean={dyn_after_prob_std_mean:+.6f} | "
            f"mean_after_rank={dyn_mean_after_rank:+.6f}"
        )
    if top1_change_feature_rows:
        top1_after_is_gt_rates = np.asarray(
            [
                float(r["top1_after_is_gt_rate_across_combos"])
                for r in top1_change_feature_rows
                if r.get("top1_after_is_gt_rate_across_combos", None) is not None
            ],
            dtype=np.float64,
        )
        changed_rates = np.asarray(
            [float(r.get("top1_changed_from_baseline_rate", 0.0)) for r in top1_change_feature_rows], dtype=np.float64
        )
        switch_rates = np.asarray(
            [float(r.get("top1_switch_rate_between_combos", 0.0)) for r in top1_change_feature_rows], dtype=np.float64
        )
        margin_beam_means = np.asarray(
            [float(r.get("top1_top2_margin_beam_mean_across_combos", 0.0)) for r in top1_change_feature_rows], dtype=np.float64
        )
        margin_prob_means = np.asarray(
            [float(r.get("top1_top2_margin_prob_mean_across_combos", 0.0)) for r in top1_change_feature_rows], dtype=np.float64
        )
        print(
            "[top1] "
            f"records={len(top1_change_feature_rows)} | "
            f"after_top1_is_gt mean={(float(np.mean(top1_after_is_gt_rates)) if top1_after_is_gt_rates.size else float('nan')):.6f} | "
            f"changed_from_baseline mean={float(np.mean(changed_rates)):.6f} | "
            f"switch_rate mean={float(np.mean(switch_rates)):.6f} | "
            f"margin_beam(top1-top2) mean={float(np.mean(margin_beam_means)):+.6f} | "
            f"margin_prob(top1-top2) mean={float(np.mean(margin_prob_means)):+.6f}"
        )
    print(f"[info] Wrote combined after+delta plots to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
