from __future__ import annotations

import argparse
import glob as glob_mod
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score,
        classification_report,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _collect_paths(jsonl_args: Sequence[str], glob_pattern: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    seen: set = set()
    for raw in jsonl_args:
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            paths.append(p)
    if glob_pattern:
        for raw in sorted(glob_mod.glob(glob_pattern, recursive=True)):
            p = Path(raw)
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                seen.add(key)
                paths.append(p)
    return paths


def _load_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"[warn] Missing: {path}; skipping")
            continue
        rows.extend(_iter_jsonl(path))
    return rows



_EPS = 1e-30  


def _safe_skew(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return 0.0
    if HAS_SCIPY:
        return float(sp_stats.skew(arr, bias=False))
    m = arr.mean()
    s = arr.std(ddof=1)
    if s < _EPS:
        return 0.0
    return float(((arr - m) ** 3).mean() / (s ** 3) * (n * (n - 1)) ** 0.5 / (n - 2))


def _safe_kurtosis(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    if HAS_SCIPY:
        return float(sp_stats.kurtosis(arr, bias=False))
    m = arr.mean()
    s = arr.std(ddof=1)
    if s < _EPS:
        return 0.0
    m4 = ((arr - m) ** 4).mean()
    return float(m4 / (s ** 4) - 3.0)


def _extract_intrinsic_features(logprobs: np.ndarray) -> Dict[str, float]:
    n = len(logprobs)
    if n == 0:
        return {}
    mean_val = float(logprobs.mean())
    median_val = float(np.median(logprobs))
    std_val = float(logprobs.std(ddof=1)) if n > 1 else 0.0
    min_val = float(logprobs.min())
    max_val = float(logprobs.max())
    range_val = max_val - min_val
    q25 = float(np.percentile(logprobs, 25))
    q75 = float(np.percentile(logprobs, 75))
    iqr_val = q75 - q25
    cv_val = std_val / (abs(mean_val) + _EPS)  # coefficient of variation
    skew_val = _safe_skew(logprobs)
    kurt_val = _safe_kurtosis(logprobs)

    # Entropy of the (softmax-normalized) prob distribution across combos
    probs_across_combos = np.exp(logprobs - logprobs.max())  # numerically stable
    probs_across_combos = probs_across_combos / (probs_across_combos.sum() + _EPS)
    entropy_val = float(-np.sum(probs_across_combos * np.log(probs_across_combos + _EPS)))

    return {
        "lp_mean": mean_val,
        "lp_median": median_val,
        "lp_std": std_val,
        "lp_min": min_val,
        "lp_max": max_val,
        "lp_range": range_val,
        "lp_q25": q25,
        "lp_q75": q75,
        "lp_iqr": iqr_val,
        "lp_cv": cv_val,
        "lp_skew": skew_val,
        "lp_kurtosis": kurt_val,
        "lp_entropy_across_combos": entropy_val,
        "n_combos": float(n),
    }


def _extract_delta_features(
    logprobs: np.ndarray, before_logprob: float,
) -> Dict[str, float]:

    delta = logprobs - before_logprob
    n = len(delta)
    if n == 0:
        return {}

    mean_val = float(delta.mean())
    median_val = float(np.median(delta))
    std_val = float(delta.std(ddof=1)) if n > 1 else 0.0
    min_val = float(delta.min())
    max_val = float(delta.max())
    range_val = max_val - min_val
    q25 = float(np.percentile(delta, 25))
    q75 = float(np.percentile(delta, 75))
    iqr_val = q75 - q25
    cv_val = std_val / (abs(mean_val) + _EPS)
    skew_val = _safe_skew(delta)
    kurt_val = _safe_kurtosis(delta)


    frac_pos = float(np.mean(delta > 0)) 
    frac_neg = float(np.mean(delta < 0))  

    gain_snr = mean_val / (std_val + _EPS)

    pos_mask = delta > 0
    mean_pos_delta = float(delta[pos_mask].mean()) if pos_mask.any() else 0.0

    neg_mask = delta < 0
    mean_neg_delta = float(delta[neg_mask].mean()) if neg_mask.any() else 0.0

    return {
        "dlp_mean": mean_val,
        "dlp_median": median_val,
        "dlp_std": std_val,
        "dlp_min": min_val,
        "dlp_max": max_val,
        "dlp_range": range_val,
        "dlp_q25": q25,
        "dlp_q75": q75,
        "dlp_iqr": iqr_val,
        "dlp_cv": cv_val,
        "dlp_skew": skew_val,
        "dlp_kurtosis": kurt_val,
        "dlp_frac_positive": frac_pos,
        "dlp_frac_negative": frac_neg,
        "dlp_gain_snr": gain_snr,
        "dlp_mean_pos_delta": mean_pos_delta,
        "dlp_mean_neg_delta": mean_neg_delta,
    }


def _extract_precomputed_features(row: Dict[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for key in [
        "before_logprob",
        "delta_logprob_mean_across_combos",
        "delta_logprob_std_across_combos",
        "after_prob_mean_across_combos",
        "after_prob_std_across_combos",
        "p_pos_across_combos",
    ]:
        val = row.get(key)
        if val is not None:
            feats[key] = float(val)
    return feats


def _extract_relative_features(
    record_candidates: List[Dict[str, Any]],
) -> List[Dict[str, float]]:

    n_cand = len(record_candidates)
    if n_cand < 2:
        return [{} for _ in record_candidates]

    combo_to_vals: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for ci, cand in enumerate(record_candidates):
        indices = cand.get("combo_indices", [])
        vals = cand.get("after_logprob_values", [])
        for combo_i, lp in zip(indices, vals):
            combo_to_vals[int(combo_i)].append((ci, float(lp)))

    ranks: Dict[int, List[int]] = defaultdict(list)       # cand_idx -> [rank per combo]
    is_top1: Dict[int, List[bool]] = defaultdict(list)     # cand_idx -> [True/False per combo]
    gap_to_best: Dict[int, List[float]] = defaultdict(list)
    wins: Dict[int, int] = defaultdict(int)
    total_comparisons: Dict[int, int] = defaultdict(int)

    for combo_i, entries in combo_to_vals.items():
        if len(entries) < 2:
            continue
        sorted_entries = sorted(entries, key=lambda x: -x[1])
        best_lp = sorted_entries[0][1]
        for rank_pos, (ci, lp) in enumerate(sorted_entries):
            ranks[ci].append(rank_pos + 1) 
            is_top1[ci].append(rank_pos == 0)
            gap_to_best[ci].append(best_lp - lp)
            for other_pos, (oci, olp) in enumerate(sorted_entries):
                if oci != ci:
                    total_comparisons[ci] += 1
                    if lp > olp:
                        wins[ci] += 1

    result: List[Dict[str, float]] = []
    for ci in range(n_cand):
        feats: Dict[str, float] = {}
        if ranks[ci]:
            rank_arr = np.array(ranks[ci], dtype=np.float64)
            feats["mean_rank"] = float(rank_arr.mean())
            feats["median_rank"] = float(np.median(rank_arr))
            feats["std_rank"] = float(rank_arr.std(ddof=1)) if len(rank_arr) > 1 else 0.0
            feats["best_rank"] = float(rank_arr.min())
            feats["worst_rank"] = float(rank_arr.max())
        if is_top1[ci]:
            feats["top1_rate"] = float(np.mean(is_top1[ci]))
        if gap_to_best[ci]:
            gap_arr = np.array(gap_to_best[ci], dtype=np.float64)
            feats["mean_gap_to_best"] = float(gap_arr.mean())
            feats["std_gap_to_best"] = float(gap_arr.std(ddof=1)) if len(gap_arr) > 1 else 0.0
        if total_comparisons[ci] > 0:
            feats["dominance_score"] = wins[ci] / total_comparisons[ci]
        result.append(feats)
    return result


def _extract_relative_delta_features(
    record_candidates: List[Dict[str, Any]],
) -> List[Dict[str, float]]:

    n_cand = len(record_candidates)
    if n_cand < 2:
        return [{} for _ in record_candidates]

    combo_to_deltas: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for ci, cand in enumerate(record_candidates):
        before = cand.get("before_logprob")
        if before is None:
            continue
        before = float(before)
        indices = cand.get("combo_indices", [])
        vals = cand.get("after_logprob_values", [])
        for combo_i, lp in zip(indices, vals):
            combo_to_deltas[int(combo_i)].append((ci, float(lp) - before))

    ranks: Dict[int, List[int]] = defaultdict(list)
    is_top1: Dict[int, List[bool]] = defaultdict(list)
    gap_to_best: Dict[int, List[float]] = defaultdict(list)
    wins: Dict[int, int] = defaultdict(int)
    total_comparisons: Dict[int, int] = defaultdict(int)

    for combo_i, entries in combo_to_deltas.items():
        if len(entries) < 2:
            continue
        sorted_entries = sorted(entries, key=lambda x: -x[1]) 
        best_delta = sorted_entries[0][1]
        for rank_pos, (ci, d) in enumerate(sorted_entries):
            ranks[ci].append(rank_pos + 1)
            is_top1[ci].append(rank_pos == 0)
            gap_to_best[ci].append(best_delta - d)
            for _, (oci, od) in enumerate(sorted_entries):
                if oci != ci:
                    total_comparisons[ci] += 1
                    if d > od:
                        wins[ci] += 1

    result: List[Dict[str, float]] = []
    for ci in range(n_cand):
        feats: Dict[str, float] = {}
        if ranks[ci]:
            rank_arr = np.array(ranks[ci], dtype=np.float64)
            feats["delta_mean_rank"] = float(rank_arr.mean())
            feats["delta_median_rank"] = float(np.median(rank_arr))
            feats["delta_best_rank"] = float(rank_arr.min())
        if is_top1[ci]:
            feats["delta_top1_rate"] = float(np.mean(is_top1[ci]))
        if gap_to_best[ci]:
            gap_arr = np.array(gap_to_best[ci], dtype=np.float64)
            feats["delta_mean_gap_to_best"] = float(gap_arr.mean())
        if total_comparisons[ci] > 0:
            feats["delta_dominance_score"] = wins[ci] / total_comparisons[ci]
        result.append(feats)
    return result


# ---------------------------------------------------------------------------
# Build feature matrix from records
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # --- intrinsic (raw after_logprob) ---
    "lp_mean", "lp_median", "lp_std", "lp_min", "lp_max",
    "lp_range", "lp_q25", "lp_q75", "lp_iqr", "lp_cv",
    "lp_skew", "lp_kurtosis", "lp_entropy_across_combos",
    # --- intrinsic (delta = after - before) ---
    "dlp_mean", "dlp_median", "dlp_std", "dlp_min", "dlp_max",
    "dlp_range", "dlp_q25", "dlp_q75", "dlp_iqr", "dlp_cv",
    "dlp_skew", "dlp_kurtosis",
    "dlp_frac_positive", "dlp_frac_negative",
    "dlp_gain_snr", "dlp_mean_pos_delta", "dlp_mean_neg_delta",
    # --- precomputed ---
    "before_logprob",
    "delta_logprob_mean_across_combos", "delta_logprob_std_across_combos",
    "after_prob_mean_across_combos", "after_prob_std_across_combos",
    "p_pos_across_combos",
    # --- relative (raw after_logprob rankings) ---
    "mean_rank", "median_rank", "std_rank", "best_rank", "worst_rank",
    "top1_rate", "mean_gap_to_best", "std_gap_to_best", "dominance_score",
    # --- relative (delta rankings) ---
    "delta_mean_rank", "delta_median_rank", "delta_best_rank",
    "delta_top1_rate", "delta_mean_gap_to_best", "delta_dominance_score",
]


def _build_feature_rows(
    grouped: Dict[str, List[Dict[str, Any]]],
    *,
    min_combos: int = 2,
) -> Tuple[
    List[Dict[str, Any]],  
    List[str],            
]:
    """
    Build one feature row per candidate across all records.
    Returns (feature_rows, record_ids_processed).
    """
    all_feature_rows: List[Dict[str, Any]] = []
    processed_rids: List[str] = []

    for rid in sorted(grouped.keys()):
        candidates = grouped[rid]
        valid = [
            c for c in candidates
            if isinstance(c.get("after_logprob_values"), list)
            and len(c["after_logprob_values"]) >= min_combos
        ]
        if len(valid) < 2:
            continue

        rel_feats_list = _extract_relative_features(valid)
        rel_delta_feats_list = _extract_relative_delta_features(valid)

        for ci, cand in enumerate(valid):
            logprobs = np.array(cand["after_logprob_values"], dtype=np.float64)
            intrinsic = _extract_intrinsic_features(logprobs)
            precomputed = _extract_precomputed_features(cand)
            relative = rel_feats_list[ci]
            rel_delta = rel_delta_feats_list[ci]

            before_lp = cand.get("before_logprob")
            if before_lp is not None:
                delta_feats = _extract_delta_features(logprobs, float(before_lp))
            else:
                delta_feats = {}

            merged: Dict[str, Any] = {}
            merged.update(intrinsic)
            merged.update(delta_feats)
            merged.update(precomputed)
            merged.update(relative)
            merged.update(rel_delta)

            merged["record_id"] = rid
            merged["candidate_answer"] = str(cand.get("candidate_answer", ""))
            merged["candidate_answer_norm"] = str(cand.get("candidate_answer_norm", ""))
            merged["gold_answer_norm"] = str(cand.get("gold_answer_norm", ""))
            is_gt = cand.get("is_gt_candidate")
            merged["is_gt"] = int(is_gt) if is_gt is not None else None

            all_feature_rows.append(merged)

        processed_rids.append(rid)

    return all_feature_rows, processed_rids


def _rows_to_matrix(
    feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.full((len(feature_rows), len(feature_names)), np.nan, dtype=np.float64)
    y = np.full(len(feature_rows), np.nan, dtype=np.float64)
    for i, row in enumerate(feature_rows):
        for j, fname in enumerate(feature_names):
            val = row.get(fname)
            if val is not None:
                X[i, j] = float(val)
        gt = row.get("is_gt")
        if gt is not None:
            y[i] = float(gt)
    return X, y


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _correlation_analysis(
    feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    out: Any = sys.stdout,
) -> List[Tuple[str, float, float]]:

    X, y = _rows_to_matrix(feature_rows, feature_names)
    labelled = ~np.isnan(y)
    X_lab = X[labelled]
    y_lab = y[labelled]

    results: List[Tuple[str, float, float]] = []
    for j, fname in enumerate(feature_names):
        col = X_lab[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        if HAS_SCIPY:
            r, p = sp_stats.pointbiserialr(y_lab[valid], col[valid])
        else:
            a, b = y_lab[valid], col[valid]
            if a.std() < _EPS or b.std() < _EPS:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            p = float("nan")  
        results.append((fname, float(r), float(p)))

    results.sort(key=lambda x: -abs(x[1]))

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 1: POINT-BISERIAL CORRELATION  (feature vs is_gt_candidate)\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Rows (labelled candidates): {int(labelled.sum())}\n\n")
    out.write(f"  {'Feature':<42s} {'r':>8s} {'p-value':>12s} {'Direction':>12s}\n")
    out.write(f"  {'─' * 76}\n")
    for fname, r, p in results:
        direction = "GT higher" if r > 0 else "GT lower"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        out.write(f"  {fname:<42s} {r:+8.4f} {p:12.2e} {direction:>12s} {sig}\n")
    out.write("\n")
    out.write("  KEY: Positive r => GT candidates have HIGHER values of this feature.\n")
    out.write("       Large |r| + small p => strong, significant discriminator.\n")
    out.write("       *** p<0.001,  ** p<0.01,  * p<0.05\n\n")
    return results


# ---------------------------------------------------------------------------
# Simple heuristics
# ---------------------------------------------------------------------------

def _evaluate_heuristics(
    feature_rows: List[Dict[str, Any]],
    corr_results: Optional[List[Tuple[str, float, float]]] = None,
    auto_topk_correlated: int = 10,
    out: Any = sys.stdout,
) -> Dict[str, Dict[str, float]]:

    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        if row.get("is_gt") is not None:
            by_record[row["record_id"]].append(row)

    heuristics = [
        # --- absolute logprob ---
        ("Highest mean logprob",              "lp_mean",                          True),
        ("Highest median logprob",            "lp_median",                        True),
        ("Lowest std (most stable)",          "lp_std",                           False),
        ("Lowest CV (most stable)",           "lp_cv",                            False),
        ("Highest before_logprob",            "before_logprob",                   True),
        ("Highest after_prob_mean",           "after_prob_mean_across_combos",    True),
        ("Highest delta_logprob_mean",        "delta_logprob_mean_across_combos", True),
        ("Highest p_pos_across_combos",       "p_pos_across_combos",              True),
        ("Best mean_rank (lowest)",           "mean_rank",                        False),
        ("Highest top1_rate",                 "top1_rate",                        True),
        ("Highest dominance_score",           "dominance_score",                  True),
        ("Smallest gap_to_best",              "mean_gap_to_best",                 False),
        # --- delta logprob (after - before) ---
        ("Highest delta mean",                "dlp_mean",                         True),
        ("Highest delta median",              "dlp_median",                       True),
        ("Lowest delta std",                  "dlp_std",                          False),
        ("Highest frac positive delta",       "dlp_frac_positive",                True),
        ("Highest delta gain SNR",            "dlp_gain_snr",                     True),
        ("Highest delta mean_pos_delta",      "dlp_mean_pos_delta",               True),
        ("Best delta_mean_rank (lowest)",     "delta_mean_rank",                  False),
        ("Highest delta_top1_rate",           "delta_top1_rate",                  True),
        ("Highest delta_dominance",           "delta_dominance_score",            True),
        ("Smallest delta_gap_to_best",        "delta_mean_gap_to_best",           False),
    ]


    auto_added = 0
    if corr_results and int(auto_topk_correlated) > 0:
        seen_feature_keys = {fkey for _, fkey, _ in heuristics}
        for fname, r, _p in corr_results:
            if fname in seen_feature_keys:
                continue
            higher_is_better = bool(r > 0.0)
            label = "Highest" if higher_is_better else "Lowest"
            heuristics.append((f"[auto corr] {label} {fname}", fname, higher_is_better))
            seen_feature_keys.add(fname)
            auto_added += 1
            if auto_added >= int(auto_topk_correlated):
                break

    n_records = len(by_record)
    results: Dict[str, Dict[str, float]] = {}

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 2: SIMPLE HEURISTICS  (rank candidates, check if GT is #1)\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Records with labelled GT + wrong: {n_records}\n\n")
    if auto_added > 0:
        out.write(
            f"  [info] Added {auto_added} auto heuristics from top correlations "
            f"(direction from r sign).\n\n"
        )
    out.write(f"  {'Heuristic':<40s} {'GT=#1':>8s} {'GT<=2':>8s} {'MRR':>8s}\n")
    out.write(f"  {'─' * 66}\n")

    for name, fkey, higher_is_better in heuristics:
        correct_at_1 = 0
        correct_at_2 = 0
        reciprocal_ranks: List[float] = []

        for rid, cands in by_record.items():
            with_feat = [(c, c.get(fkey)) for c in cands if c.get(fkey) is not None]
            if not with_feat:
                continue
            with_feat.sort(key=lambda x: x[1], reverse=higher_is_better)
            for rank_pos, (c, _) in enumerate(with_feat):
                if c.get("is_gt") == 1:
                    reciprocal_ranks.append(1.0 / (rank_pos + 1))
                    if rank_pos == 0:
                        correct_at_1 += 1
                    if rank_pos <= 1:
                        correct_at_2 += 1
                    break

        n_eval = len(reciprocal_ranks)
        if n_eval == 0:
            continue
        acc1 = correct_at_1 / n_eval
        acc2 = correct_at_2 / n_eval
        mrr = float(np.mean(reciprocal_ranks))
        results[name] = {"accuracy@1": acc1, "accuracy@2": acc2, "mrr": mrr, "n_eval": float(n_eval)}
        out.write(f"  {name:<40s} {acc1:7.1%} {acc2:7.1%} {mrr:8.4f}\n")

    out.write("\n")
    out.write("  KEY: GT=#1 = fraction of records where GT is ranked 1st by this heuristic\n")
    out.write("       GT<=2 = fraction where GT is in top 2\n")
    out.write("       MRR   = Mean Reciprocal Rank (1.0 = always first)\n\n")
    return results


# ---------------------------------------------------------------------------
# Classifier with leave-one-record-out CV
# ---------------------------------------------------------------------------

def _classifier_evaluation(
    feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    out: Any = sys.stdout,
) -> Optional[Dict[str, Any]]:

    if not HAS_SKLEARN:
        out.write("\n" + "=" * 80 + "\n")
        out.write("  PART 3: CLASSIFIER  (SKIPPED — install scikit-learn)\n")
        out.write("=" * 80 + "\n")
        out.write("  pip install scikit-learn\n\n")
        return None

    by_record: Dict[str, List[int]] = defaultdict(list)
    labelled_indices: List[int] = []
    for i, row in enumerate(feature_rows):
        if row.get("is_gt") is not None:
            by_record[row["record_id"]].append(i)
            labelled_indices.append(i)

    if len(by_record) < 5:
        out.write("\n  [skip] Too few labelled records for cross-validation.\n")
        return None

    X, y = _rows_to_matrix(feature_rows, feature_names)

    def _impute_and_scale(X_train, X_test):
        medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = medians[j]
            X_test[np.isnan(X_test[:, j]), j] = medians[j]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    record_ids = sorted(by_record.keys())

    for clf_name, make_clf in [
        ("LogisticRegression", lambda: LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs",
        )),
        ("RandomForest", lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ]:
        correct_at_1 = 0
        correct_at_2 = 0
        reciprocal_ranks: List[float] = []
        all_y_true: List[int] = []
        all_y_prob: List[float] = []
        importances_acc = np.zeros(len(feature_names), dtype=np.float64)

        for test_rid in record_ids:
            test_idx = by_record[test_rid]
            train_idx = [i for rid2 in record_ids if rid2 != test_rid for i in by_record[rid2]]

            if len(train_idx) < 10:
                continue

            X_train = X[train_idx].copy()
            y_train = y[train_idx].copy()
            X_test = X[test_idx].copy()
            y_test = y[test_idx].copy()

            if y_train.sum() < 1 or (1 - y_train).sum() < 1:
                continue

            X_train_s, X_test_s = _impute_and_scale(X_train, X_test)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = make_clf()
                clf.fit(X_train_s, y_train)

            proba = clf.predict_proba(X_test_s)[:, 1]  

            for k in range(len(test_idx)):
                all_y_true.append(int(y_test[k]))
                all_y_prob.append(float(proba[k]))

            order = np.argsort(-proba)
            gt_rank = None
            for rank_pos, idx in enumerate(order):
                if int(y_test[idx]) == 1:
                    gt_rank = rank_pos
                    break
            if gt_rank is not None:
                reciprocal_ranks.append(1.0 / (gt_rank + 1))
                if gt_rank == 0:
                    correct_at_1 += 1
                if gt_rank <= 1:
                    correct_at_2 += 1

            if hasattr(clf, "coef_"):
                importances_acc += np.abs(clf.coef_[0])
            elif hasattr(clf, "feature_importances_"):
                importances_acc += clf.feature_importances_

        n_eval = len(reciprocal_ranks)
        if n_eval == 0:
            continue

        acc1 = correct_at_1 / n_eval
        acc2 = correct_at_2 / n_eval
        mrr = float(np.mean(reciprocal_ranks))

        out.write("\n" + "=" * 80 + "\n")
        out.write(f"  PART 3: CLASSIFIER — {clf_name}  (Leave-one-record-out CV)\n")
        out.write("=" * 80 + "\n")
        out.write(f"  Records evaluated: {n_eval}\n")
        out.write(f"  GT ranked #1 (accuracy@1): {acc1:.1%}  ({correct_at_1}/{n_eval})\n")
        out.write(f"  GT in top 2  (accuracy@2): {acc2:.1%}  ({correct_at_2}/{n_eval})\n")
        out.write(f"  Mean Reciprocal Rank:      {mrr:.4f}\n")

        try:
            auc = roc_auc_score(all_y_true, all_y_prob)
            out.write(f"  AUC-ROC (candidate-level): {auc:.4f}\n")
        except ValueError:
            auc = None

        importances_avg = importances_acc / n_eval
        imp_order = np.argsort(-importances_avg)
        out.write(f"\n  Top-10 feature importances ({clf_name}):\n")
        out.write(f"  {'Feature':<42s} {'Importance':>12s}\n")
        out.write(f"  {'─' * 56}\n")
        for rank_pos in range(min(10, len(imp_order))):
            j = imp_order[rank_pos]
            out.write(f"  {feature_names[j]:<42s} {importances_avg[j]:12.4f}\n")
        out.write("\n")

    return {"status": "done"}


# ---------------------------------------------------------------------------
# Train final model & predict on unsupervised data
# ---------------------------------------------------------------------------

def _predict_unsupervised(
    train_feature_rows: List[Dict[str, Any]],
    predict_feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    out_csv: Optional[Path],
    out: Any = sys.stdout,
) -> None:
    if not HAS_SKLEARN:
        out.write("\n  [skip] scikit-learn not installed; can't predict.\n")
        return

    train_rows = [r for r in train_feature_rows if r.get("is_gt") is not None]
    if len(train_rows) < 20:
        out.write("\n  [skip] Too few labelled rows to train a reliable model.\n")
        return

    X_train, y_train = _rows_to_matrix(train_rows, feature_names)
    medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = medians[j]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_pred, _ = _rows_to_matrix(predict_feature_rows, feature_names)
    for j in range(X_pred.shape[1]):
        X_pred[np.isnan(X_pred[:, j]), j] = medians[j]
    X_pred = scaler.transform(X_pred)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_pred)[:, 1]

    for i, row in enumerate(predict_feature_rows):
        row["predicted_gt_prob"] = float(proba[i])

    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predict_feature_rows:
        by_record[row["record_id"]].append(row)

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 4: UNSUPERVISED PREDICTIONS\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Records to predict: {len(by_record)}\n\n")

    predictions: List[Dict[str, Any]] = []
    for rid in sorted(by_record.keys()):
        cands = sorted(by_record[rid], key=lambda c: -c["predicted_gt_prob"])
        top = cands[0]
        out.write(
            f"  {rid:>12s}  predicted GT: \"{top['candidate_answer']}\""
            f"  P(GT)={top['predicted_gt_prob']:.4f}"
        )
        if len(cands) > 1:
            runner = cands[1]
            gap = top["predicted_gt_prob"] - runner["predicted_gt_prob"]
            out.write(f"  gap={gap:+.4f}")
        out.write("\n")
        for rank_pos, c in enumerate(cands):
            predictions.append({
                "record_id": rid,
                "predicted_rank": rank_pos + 1,
                "candidate_answer": c["candidate_answer"],
                "candidate_answer_norm": c.get("candidate_answer_norm", ""),
                "predicted_gt_prob": c["predicted_gt_prob"],
                "lp_mean": c.get("lp_mean", ""),
                "top1_rate": c.get("top1_rate", ""),
                "dominance_score": c.get("dominance_score", ""),
            })

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = list(predictions[0].keys()) if predictions else []
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for pred in predictions:
                f.write(",".join(str(pred.get(h, "")) for h in header) + "\n")
        out.write(f"\n  [info] Wrote predictions to {out_csv}\n")


# ---------------------------------------------------------------------------
# Summary & actionable interpretation
# ---------------------------------------------------------------------------

def _print_summary(out: Any = sys.stdout) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write("  INTERPRETATION & NEXT STEPS\n")
    out.write("=" * 80 + "\n")
    


# ---------------------------------------------------------------------------
# CSV output for features
# ---------------------------------------------------------------------------

def _write_features_csv(
    feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = ["record_id", "candidate_answer", "candidate_answer_norm", "is_gt"]
    header = meta_cols + feature_names
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in feature_rows:
            vals = []
            for h in header:
                v = row.get(h)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.8f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"[info] Wrote features CSV to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict ground-truth answer from logprob distributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jsonl", action="append", default=[],
        help="candidate_stability_features JSONL (labelled, repeat for multiple).",
    )
    parser.add_argument(
        "--glob", default=None,
        help="Glob for candidate_stability_features JSONL files.",
    )
    parser.add_argument(
        "--dynamic-jsonl", action="append", default=[],
        help="Dynamic-answer features JSONL (repeat for multiple).",
    )
    parser.add_argument(
        "--dynamic-glob", default=None,
        help="Glob for dynamic-answer feature JSONL files.",
    )
    parser.add_argument(
        "--predict-jsonl", action="append", default=[],
        help="Unsupervised features JSONL to predict on (repeat for multiple).",
    )
    parser.add_argument(
        "--predict-glob", default=None,
        help="Glob for unsupervised features JSONL files.",
    )
    parser.add_argument(
        "--min-combos", type=int, default=5,
        help="Min combos required per candidate (default 5).",
    )
    parser.add_argument(
        "--heuristics-topk-correlated",
        type=int,
        default=10,
        help=(
            "Add up to K extra Part-2 heuristics from top Part-1 correlated "
            "features (direction chosen by correlation sign). Default 10; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--out-features", default=None,
        help="Path to save extracted features CSV.",
    )
    parser.add_argument(
        "--out-predictions", default=None,
        help="Path to save unsupervised prediction CSV.",
    )
    parser.add_argument(
        "--out-report", default=None,
        help="Path to save text report.",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load labelled data
    # ------------------------------------------------------------------
    all_rows: List[Dict[str, Any]] = []
    cand_paths = _collect_paths(args.jsonl, args.glob)
    if cand_paths:
        rows = _load_rows(cand_paths)
        print(f"[info] Loaded {len(rows)} candidate rows from {len(cand_paths)} file(s).")
        all_rows.extend(rows)

    dyn_paths = _collect_paths(args.dynamic_jsonl, args.dynamic_glob)
    if dyn_paths:
        rows = _load_rows(dyn_paths)
        print(f"[info] Loaded {len(rows)} dynamic-answer rows from {len(dyn_paths)} file(s).")
        all_rows.extend(rows)

    if not all_rows:
        print("[error] No labelled data. Provide --jsonl/--glob.")
        return 1

    has_arrays = any(
        isinstance(row.get("after_logprob_values"), list) and len(row["after_logprob_values"]) > 0
        for row in all_rows
    )
    if not has_arrays:
        print("[error] No 'after_logprob_values' arrays found in JSONL.")
        return 1

    grouped = defaultdict(list)
    for row in all_rows:
        rid = str(row.get("record_id", "")).strip()
        if rid:
            grouped[rid].append(row)

    print(f"[info] {len(grouped)} records with candidates.")

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    feature_rows, processed_rids = _build_feature_rows(
        dict(grouped), min_combos=args.min_combos,
    )
    print(f"[info] Extracted features for {len(feature_rows)} candidates across {len(processed_rids)} records.")

    if not feature_rows:
        print("[error] No candidates with enough combos.")
        return 1

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out = sys.stdout
    report_file = None
    if args.out_report:
        report_path = Path(args.out_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_file = report_path.open("w", encoding="utf-8")

    class TeeWriter:
        """Write to both stdout and a file."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
        def flush(self):
            for st in self.streams:
                st.flush()

    if report_file:
        out = TeeWriter(sys.stdout, report_file)

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------
    corr_results = _correlation_analysis(feature_rows, FEATURE_NAMES, out=out)

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    heuristic_results = _evaluate_heuristics(
        feature_rows,
        corr_results=corr_results,
        auto_topk_correlated=max(0, int(args.heuristics_topk_correlated)),
        out=out,
    )

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    _classifier_evaluation(feature_rows, FEATURE_NAMES, out=out)

    # ------------------------------------------------------------------
    # Unsupervised predictions (if requested)
    # ------------------------------------------------------------------
    predict_paths = _collect_paths(args.predict_jsonl, args.predict_glob)
    if predict_paths:
        pred_rows = _load_rows(predict_paths)
        print(f"[info] Loaded {len(pred_rows)} unsupervised rows from {len(predict_paths)} file(s).")
        pred_grouped = defaultdict(list)
        for row in pred_rows:
            rid = str(row.get("record_id", "")).strip()
            if rid:
                pred_grouped[rid].append(row)
        pred_feature_rows, _ = _build_feature_rows(
            dict(pred_grouped), min_combos=args.min_combos,
        )
        if pred_feature_rows:
            _predict_unsupervised(
                feature_rows, pred_feature_rows, FEATURE_NAMES,
                out_csv=Path(args.out_predictions) if args.out_predictions else None,
                out=out,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(out=out)

    # ------------------------------------------------------------------
    # Save features CSV
    # ------------------------------------------------------------------
    if args.out_features:
        _write_features_csv(feature_rows, FEATURE_NAMES, Path(args.out_features))

    if report_file:
        report_file.close()
        print(f"[info] Report saved to {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
