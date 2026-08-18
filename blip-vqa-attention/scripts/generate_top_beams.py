"""Print top-N beam answers for each record in a JSONL dataset."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from models.blip_vqa import blip_vqa
from utils import load_demo_image
from utils2 import data_path, image_root, iter_jsonl

IMAGE_SIZE = 480


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
    for idx, seq in enumerate(sequences):
        answer = tokenizer.decode(seq, skip_special_tokens=True).strip()
        score = float(scores[idx].item()) if scores is not None else float("nan")
        beams.append((answer, score))
    return beams


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print top-N beam answers for each dataset record.")
    parser.add_argument("--data-path", default=None, help="Path to dataset JSONL.")
    parser.add_argument("--target-id", default=None, help="Only process the record with this id.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many records.")
    parser.add_argument("--num-beams", type=int, default=3, help="Number of beam answers to return.")
    parser.add_argument("--max-length", type=int, default=10, help="Max generation length.")
    parser.add_argument("--min-length", type=int, default=1, help="Min generation length.")
    args = parser.parse_args(argv)

    dataset_path = Path(args.data_path) if args.data_path else data_path
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

    target_id_norm = str(args.target_id).strip().lower() if args.target_id else None
    processed = 0

    for entry in records:
        rec_id_val = entry.get("id")
        rec_id = "" if rec_id_val is None else str(rec_id_val)
        rec_id_norm = rec_id.strip().lower()
        if target_id_norm and rec_id_norm != target_id_norm:
            continue

        image_value = entry.get("image")
        question = entry.get("question")
        if image_value is None or question is None:
            print(f"[warn] Missing fields in record {entry}", file=sys.stderr)
            continue

        image_rel = Path(image_value)
        image_path = image_rel if image_rel.is_absolute() else image_root / image_rel
        if not image_path.exists():
            print(f"[warn] Missing image: {image_path}; skipping", file=sys.stderr)
            continue

        image = load_demo_image(image_path=str(image_path), image_size=IMAGE_SIZE, device=device)
        beams = _generate_top_beams(
            model,
            tokenizer,
            image,
            question,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
        )

        label = rec_id or "<unknown>"
        print(f"{label}\t{question}")
        for idx, (answer, score) in enumerate(beams, 1):
            print(f"  {idx}. {answer} (score={score:.4f})")

        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
