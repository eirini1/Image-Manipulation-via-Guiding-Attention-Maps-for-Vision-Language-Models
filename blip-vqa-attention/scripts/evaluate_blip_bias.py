from PIL import Image, ImageFile
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset, Image as HFImage
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from pathlib import Path

REPO_ID = "anvo25/vlms-are-biased"
DATA_FILES = {
    "main": [
        "data/main-00000-of-00002.parquet",
        "data/main-00001-of-00002.parquet",
    ],
    "original": [
        "data/original-00000-of-00001.parquet",
    ],
    "withtitle": [
        "data/withtitle-00000-of-00001.parquet",
    ],
}

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT = "main"

def load_split(name):
    paths = [
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname)
        for fname in DATA_FILES[name]
    ]
    tables = [pq.read_table(p) for p in paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    ds = Dataset.from_pandas(table.to_pandas(), preserve_index=False)
    return ds.cast_column("image", HFImage())

DS = load_split(SPLIT)

from models.blip_vqa import blip_vqa

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
image_size = 480
model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),  # drop alpha / force 3ch
    transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(  # BLIP / CLIP-style normalization
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    ),
])

def iter_pairs():
    for row in DS:
        img: Image.Image = row["image"]  
        prompt: str = row["prompt"]  
        gt: str = row["ground_truth"]
        eb: str = row["expected_bias"]
        meta = {k: row.get(k) for k in ("topic","sub_topic","type_of_question","pixel") if k in row}
        yield transform(img).unsqueeze(0), prompt, gt, eb, meta

_ARTICLES = {"a", "an", "the"}
_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_COMMA_STRIP = re.compile(r"(\d),(?=\d)")
_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")

def _normalize_numeric(text):
    if text is None:
        return None
    t = str(text).strip().replace(",", "")
    if not t:
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    if d == d.to_integral():
        return str(d.to_integral())
    s = format(d.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def _process_punctuation(text):
    out = text
    for p in _PUNCT:
        if (p + " " in out) or (" " + p in out) or _COMMA_STRIP.search(out):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    return out

def _normalize_text(text):
    if text is None:
        return ""
    out = str(text).lower().strip()
    out = out.replace("\n", " ").replace("\t", " ")
    out = _process_punctuation(out)
    words = []
    for w in out.split():
        if w in _ARTICLES:
            continue
        words.append(_NUM_WORDS.get(w, w))
    return " ".join(words).strip()

def _expand_answers(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for v in value:
            items.extend(_expand_answers(v))
        return items
    s = str(value).strip()
    if not s:
        return []
    if "{" in s and "}" in s:
        s = s.replace("{", "").replace("}", "")
    parts = [s]
    for sep in ["|", "/", ";"]:
        if any(sep in p for p in parts):
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(sep))
            parts = new_parts
    if any("," in p for p in parts):
        new_parts = []
        for p in parts:
            if re.search(r"\d,\d", p):
                new_parts.append(p)
            else:
                new_parts.extend(p.split(","))
        parts = new_parts
    return [p.strip() for p in parts if p.strip()]

def normalize_answer_set(value):
    answers = _expand_answers(value)
    normed = set()
    for a in answers:
        num = _normalize_numeric(a)
        if num is not None:
            normed.add(num)
        txt = _normalize_text(a)
        if txt:
            normed.add(txt)
    return normed

def normalize_model_answer(value):
    normed = set()
    num = _normalize_numeric(value)
    if num is not None:
        normed.add(num)
    txt = _normalize_text(value)
    if txt:
        normed.add(txt)
    return normed

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "out" / "blip_bias_answers.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

correct = 0
biased = 0
other = 0
total = 0
corr_on_topic = {}
biased_on_topic = {}
other_on_topic = {}

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for x, q, gt, eb, meta in iter_pairs():
        x = x.to(device)
        with torch.no_grad():
            answer = model(x, q, train=False, inference='generate')

            ans_set = normalize_model_answer(answer[0])
            gt_set = normalize_answer_set(gt)
            bias_set = normalize_answer_set(eb)

            if ans_set & gt_set:
                outcome = "CORRECT"
                correct += 1
                corr_on_topic[meta['topic']] = corr_on_topic.get(meta['topic'], 0) + 1
            elif ans_set & bias_set:
                outcome = "BIASED"
                biased += 1
                biased_on_topic[meta['topic']] = biased_on_topic.get(meta['topic'], 0) + 1
            else:
                outcome = "OTHER"
                other += 1
                other_on_topic[meta['topic']] = other_on_topic.get(meta['topic'], 0) + 1

            total += 1

            out_f.write(f"BLIP: {answer[0]}\n")
            out_f.write(f"GT: {gt}\n")
            out_f.write(f"RESULT: {outcome}\n\n")
            
counts = Counter(DS["topic"])

print(f'Out of {total} questions, {correct} correct, {biased} biased, and {other} other answers.')
print(f'({correct/total*100:.2f}% correct / {biased/total*100:.2f}% biased / {other/total*100:.2f}% other)')
print("=====Correct on topic======")
for k, v in corr_on_topic.items():
    print(f'Correct on topic {k}: {v} from {counts[k]} in total (~{v/counts[k]*100:.2f}%)')
print("=====Biased on topic======")
for k, v in biased_on_topic.items():
    print(f'Biased on topic {k}: {v} from {counts[k]} in total (~{v/counts[k]*100:.2f}%)')
print("=====Other on topic======")
for k, v in other_on_topic.items():
    print(f'Other on topic {k}: {v} from {counts[k]} in total (~{v/counts[k]*100:.2f}%)')

print("=========================================")
for k, v in counts.items():
    print(f'Topic {k} has {v} questions in total.')
        
