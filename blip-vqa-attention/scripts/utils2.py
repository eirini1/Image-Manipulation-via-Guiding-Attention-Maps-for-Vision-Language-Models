
import json
import re
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "colour", "color", "do", "does",
    "did", "for", "from", "how", "in", "is", "it", "many", "of", "on",
    "state", "states", "that", "the", "their", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with"
}

TOKEN_STOP = {"[SEP]", "?", ",", "."}
PATCH_SIZE = 16
IMAGE_SIZE = 480

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"


data_path = DATASET_ROOT / "dataset.jsonl"
image_root = DATASET_ROOT
masks_root = DATASET_ROOT / "masks"


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def canonicalize_answer(text: str) -> str:
    """Conservatively collapse common answer wrappers to a compact form."""
    base = normalize_answer(text)
    if not base:
        return ""

    tokens = base.split()
    prefixes = (
        ("it", "is"),
        ("it", "s"),
        ("this", "is"),
        ("that", "is"),
        ("there", "is"),
        ("there", "are"),
        ("color", "is"),
        ("colour", "is"),
        ("answer", "is"),
    )
    while tokens:
        removed = False
        for prefix in prefixes:
            n = len(prefix)
            if len(tokens) >= n and tuple(tokens[:n]) == prefix:
                tokens = tokens[n:]
                removed = True
                break
        if not removed:
            break

    canon = " ".join(tokens).strip()
    return canon if canon else base


def guess_focus_words(question: str) -> List[str]:
    clean = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = clean.split()
    content = [w for w in words if w not in STOP_WORDS]
    if not content:
        content = words
    if len(content) > 4:
        content = content[:4]
    return [w for w in content if w]


def find_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> List[int]:
    """Return all start indices where needle appears in haystack."""
    if not needle or len(needle) > len(haystack):
        return []
    starts: List[int] = []
    limit = len(haystack) - len(needle) + 1
    for i in range(limit):
        if haystack[i : i + len(needle)] == list(needle):
            starts.append(i)
    return starts


def select_override_indices(tokens: Sequence[str], focus_words: Sequence[str], tokenizer) -> List[int]:
    indices: List[int] = []
    if tokens and tokens[0] == "[ENC]":
        indices.append(0)
    taken = set(indices)
    for word in focus_words:
        subs = tokenizer.tokenize(word)
        if not subs:
            continue
        starts = find_subsequence(tokens, subs)
        for start in starts:
            for offset in range(len(subs)):
                idx = start + offset
                if 0 <= idx < len(tokens) and tokens[idx] not in TOKEN_STOP:
                    taken.add(idx)
    ordered = sorted(taken)
    if not ordered:
        tail = max(1, len(tokens) - 4)
        ordered = [i for i in range(tail, len(tokens)) if tokens[i] not in TOKEN_STOP]
    final = [i for i in ordered if 0 <= i < len(tokens) and tokens[i] not in TOKEN_STOP]
    if not final and tokens:
        final = [0]
    return final


def load_mask(stem: str, weights_root: Path) -> Optional[np.ndarray]:
    candidates = [
        weights_root / stem / "mask.npy",
        weights_root / f"{stem}_mask.npy",
        weights_root / f"{stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            data = np.load(path)
            if data.ndim > 2:
                data = data.squeeze()
            return data.astype(np.float32)
    return None


def parse_heads(spec: str) -> Sequence[int]:
    spec = spec.strip().lower()
    if spec in {"all", "*", "-1"}:
        return (-1,)
    parts = [p for p in re.split(r"[,\s]+", spec) if p]
    heads: List[int] = []
    for part in parts:
        if part == "-1":
            return (-1,)
        try:
            heads.append(int(part))
        except ValueError as err:
            raise ValueError(f"Invalid head spec: {part}") from err
    if not heads:
        return (-1,)
    return tuple(heads)


def iter_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if "//" in line:
                line = line.split("//", 1)[0].strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse line: {raw}") from err
            yield item


def build_layer_combos(
    total_layers: int,
    num_shards: int,
    shard_idx: int
) -> List[Tuple[int, ...]]:
    if not (0 <= shard_idx < num_shards):
        raise ValueError("Shard index must be within [0, num_shards)")
    combos: List[Tuple[int, ...]] = []
    counter = 0
    for mask in range(1, 1 << total_layers):
        layers = tuple(i for i in range(total_layers) if (mask >> i) & 1)
        if counter % num_shards == shard_idx:
            combos.append(layers)
        counter += 1
    return combos


def ensure_mask(mask: Optional[np.ndarray], gh: int, gw: int, *, stem: str) -> np.ndarray:
    if mask is None or mask.size == 0 or float(mask.max()) == float(mask.min()):
        mask = np.ones((gh, gw), dtype=np.float32)
        print(f"[warn] Using uniform mask for '{stem}'", file=sys.stderr)
    return mask


def apply_override(new_forward, layers: Tuple[int, ...], originals):
    for layer_idx in layers:
        module, orig_forward, orig_save = originals[layer_idx]
        module.forward = types.MethodType(new_forward, module)
        module.save_attention = True


def revert_override(layers: Tuple[int, ...], originals):
    for layer_idx in layers:
        module, orig_forward, orig_save = originals[layer_idx]
        module.forward = orig_forward
        module.save_attention = orig_save

def sanitize_name(s: str) -> str:
    s = "" if s is None else str(s)
    cleaned = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in s)
    return cleaned[:80]


def load_mask_from_dir(mask_dir: Path, image_stem: str, prompt: str, rec_id: Optional[str]) -> Optional[np.ndarray]:
    sanitized = sanitize_name(prompt)
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


def gaussian_from_mask(mask: torch.Tensor) -> torch.Tensor:
    h, w = mask.shape[-2], mask.shape[-1]
    if float(mask.max().item()) <= 0.0:
        return torch.zeros_like(mask)

    ys = torch.arange(h, device=mask.device, dtype=mask.dtype)
    xs = torch.arange(w, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij") if hasattr(torch.meshgrid, "__call__") else torch.meshgrid(ys, xs)

    total = mask.sum()
    if float(total.item()) <= 0.0:
        return torch.zeros_like(mask)

    weights = mask / total
    mu_x = (xx * weights).sum()
    mu_y = (yy * weights).sum()

    var_x = ((xx - mu_x) ** 2 * weights).sum()
    var_y = ((yy - mu_y) ** 2 * weights).sum()
    sigma_x = torch.sqrt(torch.clamp(var_x, min=1.0))
    sigma_y = torch.sqrt(torch.clamp(var_y, min=1.0))

    gx = (xx - mu_x) / sigma_x
    gy = (yy - mu_y) / sigma_y
    gauss = torch.exp(-0.5 * (gx * gx + gy * gy))
    maxv = gauss.max().clamp(min=1e-6)
    gauss = gauss / maxv
    return gauss
