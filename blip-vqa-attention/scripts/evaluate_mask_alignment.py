"""Compute per-(layer, head) baseline attention closeness to record masks."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.blip_vqa import blip_vqa
from pair_pipeline_common import (
    IMAGE_SIZE,
    MASK_DIR,
    PATCH_SIZE,
    prepare_question_inputs,
    resolve_heads,
    resolve_layers,
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
    select_override_indices,
)

MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
EPS = 1e-12


def _normalize_id(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").lower()


def _parse_record_ids(raw_values: Optional[Sequence[str]]) -> Set[str]:
    wanted: Set[str] = set()
    if not raw_values:
        return wanted
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            text = _normalize_id(part)
            if text:
                wanted.add(text)
    return wanted


def _record_matches(entry: Dict[str, Any], wanted_ids: Set[str]) -> bool:
    if not wanted_ids:
        return True

    image_value = entry.get("image")
    image_stem = Path(str(image_value)).stem if image_value is not None else ""
    candidates = {
        _normalize_id(entry.get("id")),
        _normalize_id(image_value),
        _normalize_id(image_stem),
    }
    return any(candidate in wanted_ids for candidate in candidates if candidate)


def _normalize_dist(values: torch.Tensor) -> torch.Tensor:
    x = values.float().clamp_min(0.0)
    total = float(x.sum().item())
    if total <= 0.0:
        return torch.full_like(x, 1.0 / max(1, x.numel()))
    return x / x.sum().clamp_min(EPS)


def _pair_metrics(attn_dist: torch.Tensor, mask_dist: torch.Tensor) -> Dict[str, float]:
    p = _normalize_dist(attn_dist)
    q = _normalize_dist(mask_dist)
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    p = p / p.sum().clamp_min(EPS)
    q = q / q.sum().clamp_min(EPS)

    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    jsd_nats = float((0.5 * kl_pm + 0.5 * kl_qm).item())
    jsd_norm = float(jsd_nats / math.log(2.0))
    jsd_similarity = float(max(0.0, 1.0 - jsd_norm))
    cosine = float(F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0), dim=1).item())
    intersection = float(torch.minimum(p, q).sum().item())
    return {
        "jsd_nats": jsd_nats,
        "jsd_norm": jsd_norm,
        "jsd_similarity": jsd_similarity,
        "cosine_similarity": cosine,
        "intersection": intersection,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline cross-attention maps to masks for each (layer, head) pair."
    )
    parser.add_argument("--layers", default="all", help="Layer pool: 'all' or comma-separated indices.")
    parser.add_argument("--heads", default="all", help="Head pool: 'all' or comma-separated indices.")
    parser.add_argument(
        "--out-jsonl",
        default="out/baseline_mask_closeness.jsonl",
        help="Output JSONL with per-record ranked pair closeness.",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help="Optional record id/image/image-stem filter (repeat or pass comma-separated values).",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on processed records (0 = all).")
    parser.add_argument(
        "--print-top-n",
        type=int,
        default=30,
        help="Print top-N pairs by JSD similarity per record (0 disables printing).",
    )
    parser.add_argument(
        "--no-gaussian",
        action="store_true",
        help="Do not blend Gaussian weighting into softened mask.",
    )
    args = parser.parse_args(argv)

    wanted_ids = _parse_record_ids(args.record_ids)
    records = [row for row in iter_jsonl(data_path) if _record_matches(row, wanted_ids)]
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        print("No matching records found.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = blip_vqa(pretrained=MODEL_URL, image_size=IMAGE_SIZE, vit="base").to(device)
    model.eval()
    tokenizer = model.tokenizer

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    total_heads = int(getattr(encoder_layers[0].crossattention.self, "num_attention_heads", 12))
    target_layers = resolve_layers(args.layers, total_layers)
    target_heads = resolve_heads(args.heads, total_heads)
    if not target_layers or not target_heads:
        raise ValueError("No layers/heads selected.")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gh = gw = IMAGE_SIZE // PATCH_SIZE
    start_time = time.time()
    processed = 0
    skipped = 0
    prog_step = 5

    with out_path.open("w", encoding="utf-8") as out_file:
        for rec_idx, entry in enumerate(records, 1):
            image_value = entry.get("image")
            question = entry.get("question")
            prompt = entry.get("prompt", "")
            rec_id = entry.get("id")
            rec_id_str = str(rec_id) if rec_id is not None else ""
            if image_value is None or question is None:
                print(f"[warn] Missing image/question in record: {entry}", file=sys.stderr)
                skipped += 1
                continue

            image_rel = Path(str(image_value))
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image path: {image_path}", file=sys.stderr)
                skipped += 1
                continue

            mask_array = load_mask_from_dir(MASK_DIR, image_rel.stem, str(prompt or ""), rec_id_str)
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
            if not args.no_gaussian:
                mask_soft = (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)
            mask_dist = _normalize_dist(mask_soft.reshape(-1))

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
            question_inputs = prepare_question_inputs(tokenizer, str(question), device)
            tokens = tokenizer.convert_ids_to_tokens(question_inputs["input_ids"][0])
            focus_words = guess_focus_words(str(question))
            focus_indices = select_override_indices(tokens, focus_words, tokenizer)
            if not focus_indices:
                focus_indices = [0]

            with torch.no_grad():
                image_embeds = model.visual_encoder(image)
                image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
                outputs = model.text_encoder(
                    input_ids=question_inputs["input_ids"],
                    attention_mask=question_inputs["attention_mask"],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_att_mask,
                    output_attentions=True,
                    return_dict=True,
                )
            cross_attentions = outputs.cross_attentions
            if not cross_attentions:
                print(f"[warn] Empty cross-attentions for id={rec_id_str or image_rel.stem}", file=sys.stderr)
                skipped += 1
                continue

            q_len = int(cross_attentions[0].shape[2])
            focus_indices = [int(idx) for idx in focus_indices if 0 <= int(idx) < q_len]
            if not focus_indices:
                focus_indices = [0]
            focus_tokens = [str(tokens[idx]) if idx < len(tokens) else f"<idx:{idx}>" for idx in focus_indices]

            ranked_pairs: List[Dict[str, Any]] = []
            for layer_idx in target_layers:
                layer_tensor = cross_attentions[int(layer_idx)][0]
                for head_idx in target_heads:
                    per_token: List[Dict[str, Any]] = []
                    for token_idx in focus_indices:
                        attn_row = layer_tensor[int(head_idx), int(token_idx), 1:]
                        metrics = _pair_metrics(attn_row, mask_dist)
                        token_result: Dict[str, Any] = {"token_idx": int(token_idx)}
                        token_result.update(metrics)
                        per_token.append(token_result)
                    if not per_token:
                        continue

                    jsd_similarity = float(np.mean([x["jsd_similarity"] for x in per_token]))
                    jsd_norm = float(np.mean([x["jsd_norm"] for x in per_token]))
                    cosine = float(np.mean([x["cosine_similarity"] for x in per_token]))
                    intersection = float(np.mean([x["intersection"] for x in per_token]))
                    ranked_pairs.append(
                        {
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "jsd_similarity": jsd_similarity,
                            "jsd_norm": jsd_norm,
                            "cosine_similarity": cosine,
                            "intersection": intersection,
                            "token_metrics": per_token,
                        }
                    )

            ranked_pairs.sort(
                key=lambda x: (
                    float(x["jsd_similarity"]),
                    float(x["cosine_similarity"]),
                    float(x["intersection"]),
                ),
                reverse=True,
            )

            row_out: Dict[str, Any] = {
                "id": rec_id_str,
                "image": str(image_value),
                "question": str(question),
                "prompt": str(prompt or ""),
                "focus_indices": [int(i) for i in focus_indices],
                "focus_tokens": focus_tokens,
                "layers": [int(v) for v in target_layers],
                "heads": [int(v) for v in target_heads],
                "mask_config": {
                    "soften_ksize": 5,
                    "soften_iters": 2,
                    "gaussian_blend": bool(not args.no_gaussian),
                },
                "rank_formula": "sort by mean_jsd_similarity (desc), then cosine, then intersection",
                "ranked_pairs": ranked_pairs,
                "top_pair": ranked_pairs[0] if ranked_pairs else None,
            }
            out_file.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            processed += 1

            if args.print_top_n > 0 and ranked_pairs:
                limit = min(int(args.print_top_n), len(ranked_pairs))
                print(f"[top_pairs] id={rec_id_str or image_rel.stem} showing {limit}/{len(ranked_pairs)}")
                for rank, item in enumerate(ranked_pairs[:limit], 1):
                    print(
                        f"  {rank:02d}. (L{int(item['layer'])},H{int(item['head'])}) "
                        f"JSD_sim={float(item['jsd_similarity']):.6f} "
                        f"cos={float(item['cosine_similarity']):.6f} "
                        f"inter={float(item['intersection']):.6f}"
                    )

            if (rec_idx == 1) or (rec_idx % prog_step == 0):
                elapsed = max(1e-6, time.time() - start_time)
                rate = rec_idx / elapsed
                print(
                    f"[progress] {rec_idx}/{len(records)} records | processed={processed} skipped={skipped} | {rate:.2f} rec/s",
                    flush=True,
                )

    print("========================================")
    print(f"Wrote: {out_path}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
