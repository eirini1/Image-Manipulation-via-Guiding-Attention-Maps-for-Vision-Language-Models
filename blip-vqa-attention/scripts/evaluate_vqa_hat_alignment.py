import os, sys, json, re, math, copy, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr, rankdata, pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
VQA_HAT_ROOT = REPO_ROOT / "external_data" / "vqa_hat"
VQA_QUESTIONS_JSON = VQA_HAT_ROOT / "OpenEnded_mscoco_val2014_questions.json"
VQA_ANNO_JSON      = VQA_HAT_ROOT / "mscoco_val2014_annotations.json"
COCO_ROOT          = VQA_HAT_ROOT
SPLIT              = "val2014"
HAT_DIR = VQA_HAT_ROOT / "vqahat_val"
VQA_TOOLS_ROOT = REPO_ROOT / "external" / "VQA"
OUT_DIR          = REPO_ROOT / "out" / "hat_eval"

LAYER = 11

SUBSET_DIR = Path(COCO_ROOT) / "subset_vqa_files"
ques_subset_path = SUBSET_DIR / "OpenEnded_mscoco_val2014_questions_HATsubset.json"
anno_subset_path = SUBSET_DIR / "mscoco_val2014_annotations_HATsubset.json"

from models.blip_vqa import blip_vqa
BLIP_MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
IMAGE_SIZE = 480
GRID_14 = (14, 14)
MAX_QUESTION_LEN = 35

sys.path.extend([
    os.path.join(VQA_TOOLS_ROOT, "PythonHelperTools"),
    os.path.join(VQA_TOOLS_ROOT, "PythonEvaluationTools"),
])
try:
    from vqaTools.vqa import VQA
    from vqaEvaluation.vqaEval import VQAEval 
except Exception as e:
    raise RuntimeError(
        f"Could not import VQA tools. Check VQA_TOOLS_ROOT ('{VQA_TOOLS_ROOT}') "
        "points to the repo that contains PythonHelperTools/ and PythonEvaluationTools/."
    ) from e

OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])


# --- Utils ---
def _resize_np_heatmap(arr: np.ndarray, out_hw=(14, 14)) -> np.ndarray:
    """Bilinear resize for 2D numpy map -> numpy."""
    t = torch.from_numpy(arr).float()[None, None, ...]
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()

def _to_rank_vector(arr2d: np.ndarray, eps=1e-8, add_noise=True, rng=None) -> np.ndarray:
    """L1-normalize, optionally tiny noise to break ties, then rank-flatten."""
    x = arr2d.astype(np.float64)
    x = x / (x.sum() + eps)
    if add_noise:
        rng = np.random.default_rng() if rng is None else rng
        x = x + rng.normal(scale=1e-14, size=x.shape)
    return rankdata(x.ravel(), method="average")

def spearman_rho_trials(map_a_2d: np.ndarray, map_b_2d: np.ndarray, trials=3) -> float:
    """Mean Spearman rank correlation with tie-breaking noise on map_b_2d."""
    rhos = []
    for _ in range(trials):
        ra = _to_rank_vector(map_a_2d, add_noise=False)
        rb = _to_rank_vector(map_b_2d, add_noise=True)
        rhos.append(float(spearmanr(ra, rb).correlation))
    return float(np.mean(rhos))

def load_hat_maps_for_qid(qid: int, out_hw=(14, 14)) -> Optional[np.ndarray]:
    """Average all available HAT maps for a question, resized to out_hw."""
    paths = sorted(HAT_DIR.glob(f"{qid}_*.png"))
    if not paths:
        return None
    maps = []
    for p in paths:
        im = Image.open(p).convert("L")
        m = np.asarray(im, dtype=np.float32) / 255.0
        maps.append(_resize_np_heatmap(m, out_hw))
    return np.mean(np.stack(maps, axis=0), axis=0)

def gaussian_center_map(hw=(14, 14), sigma=4.0) -> np.ndarray:
    """Synthetic center prior (unnormalized Gaussian); rank-corr is scale-invariant."""
    H, W = hw
    cx, cy = (H - 1) / 2.0, (W - 1) / 2.0
    m = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            dist2 = (i - cx) ** 2 + (j - cy) ** 2
            m[i, j] = math.exp(-dist2 / (2 * sigma ** 2))
    return m

def sem(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0

def mean_sem_ignore_nan(x: np.ndarray) -> Tuple[float, float, int]:
    vals = x[~np.isnan(x)]
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), sem(vals), int(vals.size)


# --- LOAD DATA
print("Loading VQA question + annotation json...")
with open(VQA_QUESTIONS_JSON, "r") as f:
    questions_data = json.load(f)
with open(VQA_ANNO_JSON, "r") as f:
    anno_data = json.load(f)

questions_list = questions_data["questions"]
anno_list = anno_data["annotations"]
qid2ques = {q["question_id"]: q for q in questions_list}
qid2ann  = {a["question_id"]: a for a in anno_list}

hat_pngs = list(HAT_DIR.glob("*.png"))
qids_with_hat = sorted({int(p.name.split("_")[0]) for p in hat_pngs})
print(f"HAT val subset: {len(qids_with_hat)} question_ids with human attention PNGs.")

print("Loading BLIP1 VQA model...")
model = blip_vqa(pretrained=BLIP_MODEL_URL, image_size=IMAGE_SIZE, vit='base').to(device)
model.eval()
if not hasattr(model, "tokenizer") or model.tokenizer is None:
    raise RuntimeError(
        "BLIP model does not expose a tokenizer. "
        "Update models.blip_vqa to set model.tokenizer, then retry."
    )
tokenizer = model.tokenizer

pred_entries = [] 
perq_records = []  
center_map_14 = gaussian_center_map(GRID_14, sigma=4.0)

with torch.no_grad():
    for idx, qid in enumerate(qids_with_hat):
        ann = qid2ann.get(qid); ques = qid2ques.get(qid)
        if ann is None or ques is None:
            continue

        image_id = ques["image_id"]
        img_file = Path(COCO_ROOT) / SPLIT / f"COCO_{SPLIT}_{image_id:012d}.jpg"
        if not img_file.exists():
            continue

        raw_image = Image.open(img_file).convert("RGB")
        img_tensor = transform(raw_image).unsqueeze(0).to(device)
        question_str = ques["question"]

        result = model(img_tensor, question_str, train=False, inference='generate')
        pred_answer = result[0] if isinstance(result, (list, tuple)) else str(result)
        pred_answer = pred_answer.strip()

        pred_entries.append({"question_id": int(qid), "answer": pred_answer})

        image_embeds = model.visual_encoder(img_tensor)
        image_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        q_inputs = tokenizer(
            question_str,
            padding="longest",
            truncation=True,
            max_length=MAX_QUESTION_LEN,
            return_tensors="pt",
        )
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        enc_token_id = getattr(tokenizer, "enc_token_id", None)
        if enc_token_id is not None:
            q_inputs["input_ids"][:, 0] = enc_token_id
        outputs = model.text_encoder(
            input_ids=q_inputs["input_ids"],
            attention_mask=q_inputs["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_att_mask,
            output_attentions=True,
            return_dict=True
        )
        xattn = outputs.cross_attentions[LAYER][0] 
        xattn = xattn[:, :, 1:]  
        attn_per_token = xattn.mean(dim=0)
        grid = int(attn_per_token.size(1) ** 0.5)
        attn_maps = attn_per_token.view(-1, grid, grid).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(q_inputs["input_ids"][0])
        token_ids = q_inputs["input_ids"][0].tolist()
        att_mask = q_inputs["attention_mask"][0].tolist()

        cls_like_id = enc_token_id if enc_token_id is not None else tokenizer.cls_token_id
        exclude_ids = {i for i in [cls_like_id, tokenizer.sep_token_id, tokenizer.pad_token_id] if i is not None}
        mean_idx = [i for i, (tid, m) in enumerate(zip(token_ids, att_mask)) if m == 1 and tid not in exclude_ids]
        if not mean_idx:
            mean_idx = [i for i, m in enumerate(att_mask) if m == 1]

        content_idx = [
            i for i, tok in enumerate(tokens)
            if att_mask[i] == 1 and tok not in ["[CLS]", "[SEP]", "?", "!", ".", ",", "##", "##s"]
        ]
        if not content_idx:
            content_idx = list(range(attn_maps.shape[0]))

        blip_cls = attn_maps[0]
        blip_cls_14 = _resize_np_heatmap(blip_cls, GRID_14)

        blip_mean = attn_maps[mean_idx].mean(axis=0)
        blip_mean_14 = _resize_np_heatmap(blip_mean, GRID_14)

        blip_content = attn_maps[content_idx].mean(axis=0)
        blip_content_14 = _resize_np_heatmap(blip_content, GRID_14)

        hat_map_14 = load_hat_maps_for_qid(qid, GRID_14)
        if hat_map_14 is None:
            continue

        # --- Correlations ---
        rho_content = spearman_rho_trials(blip_content_14, hat_map_14, trials=3)
        rho_cls = spearman_rho_trials(blip_cls_14, hat_map_14, trials=3)
        rho_mean = spearman_rho_trials(blip_mean_14, hat_map_14, trials=3)
        center_corr = spearman_rho_trials(center_map_14, hat_map_14, trials=3)

        # Collect per-question record
        perq_records.append({
            "qid": int(qid),
            "image_id": int(image_id),
            "img_path": str(img_file),
            "question": question_str,
            "pred": pred_answer,
            "gt_majority": ann.get("multiple_choice_answer", ""),
            "answer_type": ann.get("answer_type", None),
            "question_type": ann.get("question_type", None),
            "rho_content": float(rho_content),
            "rho_cls": float(rho_cls),
            "rho_mean": float(rho_mean),
            "hat_center_corr": float(center_corr),
            "hat_map_14": hat_map_14
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(qids_with_hat)}...")

# Save predictions in official format
res_file = OUT_DIR / "v1_OpenEnded_mscoco_val2014_results_blip_hat_subset.json"
with open(res_file, "w") as f:
    json.dump(pred_entries, f)
print(f"Saved predictions to: {res_file}")

questions_subset = {
    k: v for k, v in questions_data.items() if k in ["info","task_type","data_type","data_subtype","license"]
}
questions_subset["questions"] = [
    q for q in questions_data["questions"] if int(q["question_id"]) in set(qids_with_hat)
]

anno_subset = {
    k: v for k, v in anno_data.items() if k in ["info","data_type","data_subtype","license"]
}
anno_subset["annotations"] = [
    a for a in anno_data["annotations"] if int(a["question_id"]) in set(qids_with_hat)
]

with open(ques_subset_path, "w") as f:
    json.dump(questions_subset, f)
with open(anno_subset_path, "w") as f:
    json.dump(anno_subset, f)

print(f"Subset question file:    {ques_subset_path}")
print(f"Subset annotation file:  {anno_subset_path}")

from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

print("Running VQAEval...")
vqa    = VQA(str(anno_subset_path), str(ques_subset_path))
vqaRes = vqa.loadRes(str(res_file),  str(ques_subset_path)) 
vqaEval = VQAEval(vqa, vqaRes)
vqaEval.evaluate()

qid2acc_pct = {int(k): float(v) for k, v in vqaEval.evalQA.items()} 
for r in perq_records:
    r["vqa_acc"] = qid2acc_pct.get(r["qid"], 0.0) / 100.0 

rho_content_vals = np.array([r["rho_content"] for r in perq_records], dtype=np.float64)
rho_cls_vals = np.array([r["rho_cls"] for r in perq_records], dtype=np.float64)
rho_mean_vals = np.array([r["rho_mean"] for r in perq_records], dtype=np.float64)
acc_vals = np.array([r["vqa_acc"]     for r in perq_records], dtype=np.float64)
ans_types = [r["answer_type"]   for r in perq_records]
ques_types = [r["question_type"] for r in perq_records]

rho_content_mean, rho_content_sem, rho_content_n = mean_sem_ignore_nan(rho_content_vals)
rho_cls_mean, rho_cls_sem, rho_cls_n = mean_sem_ignore_nan(rho_cls_vals)
rho_mean_mean, rho_mean_sem, rho_mean_n = mean_sem_ignore_nan(rho_mean_vals)
acc_mean, acc_sem = float(np.mean(acc_vals)), sem(acc_vals)
rho_vals = rho_content_vals
rho_mean = rho_content_mean
rho_sem = rho_content_sem

print("\n===== OVERALL (official accuracy + HAT ρ) =====")
print(f"Mean ρ (BLIP vs HAT): {rho_mean:.3f} ± {rho_sem:.3f} (SEM)")
print(f"Mean VQA soft accuracy: {100*acc_mean:.2f}% ± {100*acc_sem:.2f}% (SEM)")
print("Inter-human attention upper bound on val: ~0.623 (Spearman ρ).")  


print("\n===== OVERALL (extra attention maps) =====")
print(f"Mean rho (CLS token):        {rho_cls_mean:.3f} ± {rho_cls_sem:.3f} (SEM)  n={rho_cls_n}")
print(f"Mean rho (mean non-special): {rho_mean_mean:.3f} ± {rho_mean_sem:.3f} (SEM)  n={rho_mean_n}")
print("\n===== BY ANSWER TYPE =====")
by_ans = []
for t in sorted(set(x for x in ans_types if x is not None)):
    idxs = [i for i, a in enumerate(ans_types) if a == t]
    r_m = float(np.mean(rho_vals[idxs])) if idxs else float("nan")
    a_m = float(np.mean(acc_vals[idxs])) if idxs else float("nan")
    by_ans.append({"answer_type": t, "mean_rho": r_m, "mean_acc": a_m, "n": len(idxs)})
    print(f"{t:>7s}:  ρ={r_m:.3f}   acc={100*a_m:.2f}%   n={len(idxs)}")

# Pivot by question_type (mean ρ and mean accuracy)
print("\n===== BY QUESTION TYPE (pivot) =====")
by_qtype = []
for t in sorted(set(x for x in ques_types if x is not None)):
    idxs = [i for i, qt in enumerate(ques_types) if qt == t]
    r_m = float(np.mean(rho_vals[idxs])) if idxs else float("nan")
    a_m = float(np.mean(acc_vals[idxs])) if idxs else float("nan")
    by_qtype.append({"question_type": t, "mean_rho": r_m, "mean_acc": a_m, "n": len(idxs)})

df_qtype = pd.DataFrame(by_qtype).sort_values("n", ascending=False)
df_qtype[["question_type","mean_rho","mean_acc","n"]].to_csv(OUT_DIR / "pivot_question_type.csv", index=False)
print(df_qtype[["question_type","mean_rho","mean_acc","n"]].to_string(index=False))

# Correlation between skills: ρ vs accuracy
pear_r, pear_p = pearsonr(rho_vals, acc_vals)
spear_r, spear_p = spearmanr(rho_vals, acc_vals)
print("\n===== ρ vs. ACCURACY CORRELATION =====")
print(f"Pearson r = {pear_r:.3f}  (p={pear_p:.3g})")
print(f"Spearman  = {spear_r:.3f}  (p={spear_p:.3g})")

# CENTER-BIAS FILTERED METRIC
center_corrs = np.array([r["hat_center_corr"] for r in perq_records], dtype=np.float64)
keep_mask = center_corrs <= 0.0
rho_filtered = rho_vals[keep_mask]
rho_cls_filtered = rho_cls_vals[keep_mask]
rho_mean_filtered = rho_mean_vals[keep_mask]
print("\n===== HAT REDUCED SET (no center bias) =====")
print(f"Kept {len(rho_filtered)}/{len(rho_vals)} samples (filter: human-vs-center ρ <= 0).")
print(f"Mean ρ (filtered) = {float(np.mean(rho_filtered)):.3f} ± {sem(rho_filtered):.3f} (SEM)")

print(f"Mean rho (CLS, filtered) = {float(np.mean(rho_cls_filtered)):.3f} +/- {sem(rho_cls_filtered):.3f} (SEM)")
print(f"Mean rho (mean non-special, filtered) = {float(np.mean(rho_mean_filtered)):.3f} +/- {sem(rho_mean_filtered):.3f} (SEM)")
plt.figure(figsize=(6,4))
plt.hist(rho_vals, bins=20, edgecolor='black')
plt.xlabel("Spearman ρ (BLIP vs HAT)")
plt.ylabel("# Questions")
plt.title("Distribution of attention alignment (ρ)")
plt.axvline(x=float(np.mean(rho_vals)), linestyle='--', label=f"Mean={np.mean(rho_vals):.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_rho.png", dpi=200)
plt.close()

# Boxplot of ρ by answer_type
pairs = []
for t in sorted(set(ans_types)):
    if t is None: 
        continue
    grp = [rho_vals[i] for i,a in enumerate(ans_types) if a == t]
    if grp:
        pairs.append((t, grp))
labels = [p[0] for p in pairs]
groups = [p[1] for p in pairs]

plt.figure(figsize=(6,4))
plt.boxplot(groups, labels=labels, showfliers=True, whis=[0,100])  # full range whiskers
plt.ylabel("Spearman ρ (BLIP vs HAT)")
plt.title("Alignment (ρ) by answer_type")
plt.tight_layout()
plt.savefig(OUT_DIR / "box_rho_by_ans_type.png", dpi=200)
plt.close()

# Scatter ρ vs accuracy with regression
plt.figure(figsize=(6,5))
plt.scatter(rho_vals, acc_vals, alpha=0.6)
if len(rho_vals) > 1:
    m, b = np.polyfit(rho_vals, acc_vals, 1)
    xs = np.array([np.min(rho_vals), np.max(rho_vals)])
    plt.plot(xs, m*xs + b, linestyle='--', label=f"Pearson r={pear_r:.3f}")
    plt.legend()
plt.xlabel("Spearman ρ (attention alignment)")
plt.ylabel("VQA soft accuracy")
plt.title("ρ vs VQA accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_rho_vs_acc.png", dpi=200)
plt.close()

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write("==== OVERALL ====\n")
    f.write(f"Mean rho: {rho_mean:.4f} ± {rho_sem:.4f} (SEM)\n")
    f.write(f"Mean VQA soft acc: {100*acc_mean:.2f}% ± {100*acc_sem:.2f}% (SEM)\n")
    f.write("Inter-human upper bound (HAT val): ~0.623 rho\n\n")
    f.write("==== EXTRA MAPS ====\n")
    f.write(f"Mean rho (CLS token): {rho_cls_mean:.4f} ± {rho_cls_sem:.4f} (SEM)  n={rho_cls_n}\n")
    f.write(f"Mean rho (mean non-special): {rho_mean_mean:.4f} ± {rho_mean_sem:.4f} (SEM)  n={rho_mean_n}\n")
    f.write("\n")
    f.write("==== RHO vs ACC ====\n")
    f.write(f"Pearson r={pear_r:.4f} (p={pear_p:.3g}) | Spearman={spear_r:.4f} (p={spear_p:.3g})\n\n")
    f.write("==== REDUCED SET (no center bias) ====\n")
    f.write(f"Kept {len(rho_filtered)}/{len(rho_vals)} | Mean rho={float(np.mean(rho_filtered)):.4f} ± {sem(rho_filtered):.4f}\n\n")
    f.write(f"Mean rho (CLS, filtered) = {float(np.mean(rho_cls_filtered)):.4f} +/- {sem(rho_cls_filtered):.4f} (SEM)\n")
    f.write(f"Mean rho (mean non-special, filtered) = {float(np.mean(rho_mean_filtered)):.4f} +/- {sem(rho_mean_filtered):.4f} (SEM)\n\n")
    f.write("==== BY ANSWER TYPE ====\n")
    for row in by_ans:
        f.write(f"{row['answer_type']}: rho={row['mean_rho']:.4f} | acc={100*row['mean_acc']:.2f}% | n={row['n']}\n")
    f.write("\nTop files:\n")
    f.write(str(OUT_DIR) + "\n")
    f.write(str(OUT_DIR) + "\n")
    f.write(str(OUT_DIR) + "\n")
