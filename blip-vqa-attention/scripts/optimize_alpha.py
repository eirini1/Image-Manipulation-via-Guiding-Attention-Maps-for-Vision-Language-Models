import torch
import torch.nn.functional as F
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from torch import nn
from typing import Dict, Optional, Sequence, Tuple, List
from models.blip_vqa import blip_vqa
from utils import load_demo_image, soften_mask, make_forward
from utils2 import (
    apply_override,
    canonicalize_answer,
    data_path as DATA_PATH,
    ensure_mask,
    guess_focus_words,
    image_root,
    iter_jsonl,
    parse_heads,
    revert_override,
    select_override_indices,
    load_mask_from_dir,
    masks_root,
    gaussian_from_mask,
    normalize_answer,
)

IMAGE_SIZE = 480
PATCH_SIZE = 16
MASK_DIR = masks_root

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Do fine tuning on given example."
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer spec: 'all' or comma-separated indices.",
    )
    parser.add_argument(
        "--heads",
        default="all",
        help="Head spec: 'all' or comma-separated indices.",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help=(
            "Restrict processing to records whose id/image stem/path matches the given value. "
            "Can be supplied multiple times or as comma-separated values."
        ),
    )
    parser.add_argument(
        "--output",
        default="out/fine_tuning_results.jsonl",
        help="Write results JSONL to this path (pretty-printed, records may span lines).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over all records.",
    )
    parser.add_argument(
        "--steps-per-record",
        type=int,
        default=30,
        help="Optimization steps per record within each epoch.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PATH,
        help="Path to input JSONL dataset.",
    )
    parser.add_argument(
        "--opt-all",
        action="store_true",
        help="Optimize alpha for all records.",
    )
    parser.add_argument(
        "--opt-target",
        choices=("gt", "topk_gt_missing"),
        default="gt",
        help=(
            "Loss target: ground truth answer, or optimize separately for original top-K "
            "beams plus GT if GT is missing from top-K."
        ),
    )
    parser.add_argument(
        "--num-beams",
        dest="num_beams",
        type=int,
        default=3,
        help="Beam width K used by topk_gt_missing (top-K candidates from the original model).",
    )

    return parser.parse_args()


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

def _record_matches(entry: Dict[str, object], targets: Sequence[str]) -> bool:
    if not targets:
        return True
    rec_id = entry.get("id")
    image = entry.get("image")
    image_str = str(image) if image is not None else ""
    stem = Path(image_str).stem if image_str else ""
    for target in targets:
        if target == rec_id or target == stem or target == image_str:
            return True
    return False


def _as_text_answer(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


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
    num_beams: int = 3,
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
    for idx, seq in enumerate(sequences):
        answer = tokenizer.decode(seq, skip_special_tokens=True).strip()
        score = float(scores[idx].item()) if scores is not None else float("nan")
        beams.append((answer, score))
    return beams


def _resolve_optimization_targets(
    *,
    opt_target: str,
    model,
    tokenizer,
    image: torch.Tensor,
    question: str,
    gt_answer: str,
    num_beams: int,
) -> List[Dict[str, object]]:
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
        targets: List[Dict[str, object]] = []
        seen_norm: set = set()
        seen_topk_keys: set = set()
        baseline_topk: List[str] = []
        ranked_beams: List[Tuple[int, str, float]] = []

        for rank, (answer, score) in enumerate(beams[:num_beams], 1):
            cleaned = (answer or "").strip()
            if not cleaned:
                continue
            canonical = canonicalize_answer(cleaned)
            chosen = canonical if canonical else cleaned
            key = normalize_answer(chosen) or chosen.lower()
            if key in seen_topk_keys:
                continue
            seen_topk_keys.add(key)
            baseline_topk.append(chosen)
            ranked_beams.append((rank, chosen, score))
            seen_norm.add(key)

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
            gt_canonical = canonicalize_answer(gt_clean)
            gt_used = gt_canonical if gt_canonical else gt_clean
            gt_norm = normalize_answer(gt_used)
            if not gt_norm or gt_norm not in seen_norm:
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

        return [
            {
                "opt_answer": gt_answer,
                "opt_target": "gt_fallback",
                "target_beam_rank": None,
                "target_beam_score": None,
                "baseline_topk": baseline_topk,
            }
        ]

    return [
        {
            "opt_answer": gt_answer,
            "opt_target": "gt",
            "target_beam_rank": None,
            "target_beam_score": None,
            "baseline_topk": None,
        }
    ]


def _write_pretty_jsonl(path: Path, results: Sequence[Dict[str, object]]) -> None:
    preferred_keys = [
        "epoch",
        "index",
        "id",
        "image",
        "prompt",
        "question",
        "answer",
        "opt_answer",
        "opt_target",
        "target_beam_rank",
        "target_beam_score",
        "baseline_topk",
        "top3_alpha_answers_desc",
        "best_alpha",
        "head_gates",
        "layers",
        "heads",
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec_idx, result in enumerate(results):
            keys = [k for k in preferred_keys if k in result]
            keys += sorted(k for k in result.keys() if k not in preferred_keys)
            f.write("{\n")
            for key_idx, key in enumerate(keys):
                value = result[key]
                if key == "head_gates" and isinstance(value, list):
                    f.write(f'  "{key}": [\n')
                    for row_idx, row in enumerate(value):
                        row_json = json.dumps(row, sort_keys=True)
                        comma = "," if row_idx < len(value) - 1 else ""
                        f.write(f"    {row_json}{comma}\n")
                    f.write("  ]")
                else:
                    f.write(f'  "{key}": {json.dumps(value, sort_keys=True)}')
                comma = "," if key_idx < len(keys) - 1 else ""
                f.write(comma + "\n")
            f.write("}")
            if rec_idx < len(results) - 1:
                f.write("\n\n")


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.num_beams < 1:
        raise ValueError("--num-beams must be >= 1.")

    input_path = args.input.expanduser()
    if not input_path.exists():
        print(f"[warn] Input dataset not found: {input_path}", file=sys.stderr)
        return

    records = list(iter_jsonl(input_path))
    if not records:
        print("No records to process.")
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    model = blip_vqa(pretrained=model_url, image_size=IMAGE_SIZE, vit="base")
    model.eval()
    model = model.to(device)
    tokenizer = model.tokenizer

    heads_spec = parse_heads(args.heads)
    layers_spec = parse_heads(args.layers)

    encoder_layers = model.text_encoder.encoder.layer
    total_layers = len(encoder_layers)
    originals = []
    for layer in encoder_layers:
        sa = layer.crossattention.self
        originals.append((sa, sa.forward, getattr(sa, "save_attention", False)))

    # Resolve which layers to override
    if layers_spec == (-1,):
        target_layers: Tuple[int, ...] = tuple(range(total_layers))
    else:
        target_layers = tuple(int(x) for x in layers_spec)

    tokenizer = model.tokenizer
    mask_cache: Dict[Tuple[str, str], np.ndarray] = {}
    heads_arg = heads_spec[0] if len(heads_spec) == 1 else heads_spec
    results: List[Dict[str, object]] = []
    record_targets = _parse_record_ids(args.record_ids)
    force_per_target = args.opt_target == "topk_gt_missing"

    #------------------ Optimization setup ------------------
    if args.opt_all and not force_per_target:
        for p in model.parameters():
            p.requires_grad = False

        raw_alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
        opt = torch.optim.Adam([raw_alpha], lr=0.05)

    #-------------------------------------------------------

    for epoch in range(args.epochs):
        for idx, entry in enumerate(records, 1):
            if record_targets and not _record_matches(entry, record_targets):
                continue
            image_value = entry.get("image")
            prompt = entry.get("prompt")
            question = entry["question"]
            gt_answer = _as_text_answer(entry["answer"])
            id = entry.get("id")

            if image_value is None or entry.get("question") is None or entry.get("answer") is None:
                print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
                continue

            image_rel = Path(image_value)
            image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
                continue

            image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
            target_specs = _resolve_optimization_targets(
                opt_target=args.opt_target,
                model=model,
                tokenizer=tokenizer,
                image=image,
                question=question,
                gt_answer=gt_answer,
                num_beams=args.num_beams,
            )

            cache_key = (image_rel.as_posix(), prompt)
            if cache_key in mask_cache:
                mask_array = mask_cache[cache_key]
            else:
                rec_id = entry.get("id")
                mask_array = load_mask_from_dir(MASK_DIR, image_rel.stem, prompt or "", rec_id)
                if mask_array is not None:
                    mask_cache[cache_key] = mask_array

            gh = gw = IMAGE_SIZE // PATCH_SIZE
            try:
                mask_array = ensure_mask(mask_array, gh, gw, stem=image_rel.stem)
            except RuntimeError:
                continue
            
            mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)
            mask_small = F.interpolate(mask_tensor, size=(gh, gw), mode="bilinear", align_corners=False)
            mask_small = mask_small.squeeze(0).squeeze(0).clamp_(0.0, 1.0)
            mask_soft = soften_mask(mask_small, ksize=5, iters=2)
            mask_soft = (gaussian_from_mask(mask_soft) * mask_soft).clamp_(0.0, 1.0)

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

            #------------------ Optimization setup ------------------
            #------------------ Optimization setup ------------------
            per_target_results: List[Dict[str, object]] = []
            for target_spec in target_specs:
                opt_answer = str(target_spec["opt_answer"])
                opt_target_used = str(target_spec["opt_target"])

                if not args.opt_all or force_per_target:
                    for p in model.parameters():
                        p.requires_grad = False

                    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
                    opt = torch.optim.Adam([raw_alpha], lr=0.05)
                #------------------ Optimization setup ------------------

                print(
                    f"################ Epoch {epoch + 1}/{args.epochs} Record {idx} "
                    f"Target {opt_target_used} ################"
                )
                for step in range(args.steps_per_record):
                    alpha = torch.sigmoid(raw_alpha)
                    for layer_idx in target_layers:
                        apply_override(make_forward(heads_arg, override_rows, alpha), (layer_idx,), originals)

                    try:
                        weights = torch.tensor([1.0], device=device)
                        loss = model(image, [question], answer=[opt_answer], train=True, n=[1], weights=weights)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    finally:
                        revert_override(target_layers, originals)

                best_alpha = torch.sigmoid(raw_alpha).detach().item()
                if not args.quiet:
                    print("best alpha:", best_alpha)
                    print("###########################################")

                if epoch % 5 == 0 or epoch == args.epochs - 1:
                    print(
                        f"[info] Epoch {epoch + 1} Record {idx} "
                        f"Target {opt_target_used} done. Best alpha: {best_alpha}."
                    )

                rec_result = {
                    "epoch": epoch + 1,
                    "index": idx,
                    "id": id,
                    "image": image_rel.as_posix(),
                    "prompt": prompt,
                    "question": question,
                    "answer": gt_answer,
                    "opt_answer": opt_answer,
                    "opt_target": opt_target_used,
                    "target_beam_rank": target_spec["target_beam_rank"],
                    "target_beam_score": target_spec["target_beam_score"],
                    "baseline_topk": target_spec["baseline_topk"],
                    "best_alpha": best_alpha,
                    "layers": list(target_layers),
                    "heads": heads_arg,
                }
                results.append(rec_result)
                per_target_results.append(rec_result)

            if args.opt_target == "topk_gt_missing" and per_target_results:
                ranked = sorted(per_target_results, key=lambda r: float(r["best_alpha"]), reverse=True)
                top3_alpha_answers_desc = [
                    {
                        "answer": str(r["opt_answer"]),
                        "best_alpha": float(r["best_alpha"]),
                        "opt_target": str(r["opt_target"]),
                    }
                    for r in ranked[:3]
                ]
                for rec_result in per_target_results:
                    rec_result["top3_alpha_answers_desc"] = top3_alpha_answers_desc
                if not args.quiet:
                    print("[info] Top-3 answers by alpha:")
                    for rank, item in enumerate(top3_alpha_answers_desc, 1):
                        print(f"  {rank}. {item['answer']} (alpha={item['best_alpha']:.6f}, target={item['opt_target']})")

    if args.output:
        output_path = Path(args.output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_pretty_jsonl(output_path, results)
            print(f"Wrote results JSONL to {output_path}")
        except OSError as exc:
            print(f"[warn] Failed to write results to {output_path}: {exc}", file=sys.stderr)
        


if __name__ == "__main__":
    main()
