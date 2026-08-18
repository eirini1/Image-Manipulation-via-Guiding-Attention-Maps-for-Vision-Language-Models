from __future__ import annotations

import argparse
import glob as glob_mod
import json
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
    from sklearn.metrics import roc_auc_score
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


def _flatten_record_jsonl(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    
    flat: List[Dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("record_id", "")).strip()
        if not rid:
            continue
        gold_norm = str(row.get("gold_answer_norm", "")).strip()
        num_combos = int(row.get("num_combos", 0))
        distributions = row.get("union_answer_alpha_distributions", [])
        if not isinstance(distributions, list):
            continue
        for ans_entry in distributions:
            if not isinstance(ans_entry, dict):
                continue
            ans_norm = str(ans_entry.get("answer_norm", "")).strip()
            alpha_vals = ans_entry.get("alpha_values", [])
            if not isinstance(alpha_vals, list) or not alpha_vals:
                continue
            is_gt = None
            if gold_norm:
                is_gt = 1 if ans_norm == gold_norm else 0
            flat.append({
                "record_id": rid,
                "candidate_answer": str(ans_entry.get("answer", "")),
                "candidate_answer_norm": ans_norm,
                "gold_answer_norm": gold_norm,
                "is_gt_candidate": is_gt,
                "alpha_values": [float(v) for v in alpha_vals],
                "combo_indices": list(range(len(alpha_vals))),
                # carry precomputed stats if present
                "alpha_mean": ans_entry.get("alpha_mean"),
                "alpha_std": ans_entry.get("alpha_std"),
                "alpha_min": ans_entry.get("alpha_min"),
                "alpha_max": ans_entry.get("alpha_max"),
                "alpha_p25": ans_entry.get("alpha_p25"),
                "alpha_p50": ans_entry.get("alpha_p50"),
                "alpha_p75": ans_entry.get("alpha_p75"),
                "source_rank_positions": ans_entry.get("source_rank_positions", []),
                "source_rank_count": ans_entry.get("source_rank_count", 0),
            })
    return flat


def _flatten_pair_jsonl(
    pair_rows: List[Dict[str, Any]],
    gold_map: Dict[str, str],
) -> List[Dict[str, Any]]:

    # Group: (record_id, answer_norm) -> {combo_index: best_alpha}
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in pair_rows:
        rid = str(row.get("record_id", "")).strip()
        ans_norm = str(row.get("answer_norm", "")).strip()
        if not rid or not ans_norm:
            continue
        key = (rid, ans_norm)
        if key not in grouped:
            grouped[key] = {
                "record_id": rid,
                "candidate_answer": str(row.get("answer", "")),
                "candidate_answer_norm": ans_norm,
                "combo_alpha_map": {},  # combo_index -> alpha
            }
        combo_idx = row.get("combo_index")
        alpha = row.get("best_alpha")
        if combo_idx is not None and alpha is not None:
            grouped[key]["combo_alpha_map"][int(combo_idx)] = float(alpha)

    flat: List[Dict[str, Any]] = []
    for (rid, ans_norm), info in grouped.items():
        cam = info["combo_alpha_map"]
        if not cam:
            continue
        sorted_combos = sorted(cam.keys())
        alpha_vals = [cam[c] for c in sorted_combos]
        gold_norm = gold_map.get(rid, "")
        is_gt = None
        if gold_norm:
            is_gt = 1 if ans_norm == gold_norm else 0
        flat.append({
            "record_id": rid,
            "candidate_answer": info["candidate_answer"],
            "candidate_answer_norm": ans_norm,
            "gold_answer_norm": gold_norm,
            "is_gt_candidate": is_gt,
            "alpha_values": alpha_vals,
            "combo_indices": sorted_combos,
            "alpha_mean": None,
            "alpha_std": None,
            "alpha_min": None,
            "alpha_max": None,
            "alpha_p25": None,
            "alpha_p50": None,
            "alpha_p75": None,
            "source_rank_positions": [],
            "source_rank_count": 0,
        })
    return flat


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


def _extract_alpha_intrinsic_features(alpha_arr: np.ndarray) -> Dict[str, float]:
    n = len(alpha_arr)
    if n == 0:
        return {}
    mean_val = float(alpha_arr.mean())
    median_val = float(np.median(alpha_arr))
    std_val = float(alpha_arr.std(ddof=1)) if n > 1 else 0.0
    min_val = float(alpha_arr.min())
    max_val = float(alpha_arr.max())
    range_val = max_val - min_val
    q25 = float(np.percentile(alpha_arr, 25))
    q75 = float(np.percentile(alpha_arr, 75))
    iqr_val = q75 - q25
    cv_val = std_val / (abs(mean_val) + _EPS)
    skew_val = _safe_skew(alpha_arr)
    kurt_val = _safe_kurtosis(alpha_arr)

    frac_high = float(np.mean(alpha_arr > 0.5))  
    frac_low = float(np.mean(alpha_arr < 0.5))  
    frac_extreme_high = float(np.mean(alpha_arr > 0.8))
    frac_extreme_low = float(np.mean(alpha_arr < 0.2))


    if range_val > _EPS and n > 1:
        counts, _ = np.histogram(alpha_arr, bins=10, range=(0, 1))
        probs = counts / (counts.sum() + _EPS)
        entropy_val = float(-np.sum(probs * np.log(probs + _EPS)))
    else:
        entropy_val = 0.0


    consistency = 1.0 / (std_val + _EPS)

    return {
        "a_mean": mean_val,
        "a_median": median_val,
        "a_std": std_val,
        "a_min": min_val,
        "a_max": max_val,
        "a_range": range_val,
        "a_q25": q25,
        "a_q75": q75,
        "a_iqr": iqr_val,
        "a_cv": cv_val,
        "a_skew": skew_val,
        "a_kurtosis": kurt_val,
        "a_frac_high": frac_high,
        "a_frac_low": frac_low,
        "a_frac_extreme_high": frac_extreme_high,
        "a_frac_extreme_low": frac_extreme_low,
        "a_entropy": entropy_val,
        "a_consistency": consistency,
        "n_combos": float(n),
    }


def _extract_source_rank_features(row: Dict[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    ranks = row.get("source_rank_positions", [])
    if isinstance(ranks, list) and ranks:
        rank_arr = np.array([float(r) for r in ranks], dtype=np.float64)
        feats["src_rank_min"] = float(rank_arr.min())
        feats["src_rank_mean"] = float(rank_arr.mean())
    count = row.get("source_rank_count")
    if count is not None:
        feats["src_rank_count"] = float(count)
    return feats


def _extract_relative_alpha_features(
    record_candidates: List[Dict[str, Any]],
) -> List[Dict[str, float]]:

    n_cand = len(record_candidates)
    if n_cand < 2:
        return [{} for _ in record_candidates]

    combo_to_alphas: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for ci, cand in enumerate(record_candidates):
        indices = cand.get("combo_indices", [])
        vals = cand.get("alpha_values", [])
        for combo_i, a in zip(indices, vals):
            combo_to_alphas[int(combo_i)].append((ci, float(a)))

    lo_ranks: Dict[int, List[int]] = defaultdict(list)
    lo_top1: Dict[int, List[bool]] = defaultdict(list)
    lo_gap: Dict[int, List[float]] = defaultdict(list)
    lo_wins: Dict[int, int] = defaultdict(int)
    lo_total: Dict[int, int] = defaultdict(int)

    hi_ranks: Dict[int, List[int]] = defaultdict(list)
    hi_top1: Dict[int, List[bool]] = defaultdict(list)
    hi_gap: Dict[int, List[float]] = defaultdict(list)
    hi_wins: Dict[int, int] = defaultdict(int)
    hi_total: Dict[int, int] = defaultdict(int)

    for combo_i, entries in combo_to_alphas.items():
        if len(entries) < 2:
            continue

        # --- lowest-alpha ranking ---
        sorted_lo = sorted(entries, key=lambda x: x[1])  # ascending
        best_lo = sorted_lo[0][1]
        for rp, (ci, a) in enumerate(sorted_lo):
            lo_ranks[ci].append(rp + 1)
            lo_top1[ci].append(rp == 0)
            lo_gap[ci].append(a - best_lo)  # >=0
            for _, (oci, oa) in enumerate(sorted_lo):
                if oci != ci:
                    lo_total[ci] += 1
                    if a < oa:
                        lo_wins[ci] += 1

        # --- highest-alpha ranking ---
        sorted_hi = sorted(entries, key=lambda x: -x[1])  # descending
        best_hi = sorted_hi[0][1]
        for rp, (ci, a) in enumerate(sorted_hi):
            hi_ranks[ci].append(rp + 1)
            hi_top1[ci].append(rp == 0)
            hi_gap[ci].append(best_hi - a)  # >=0
            for _, (oci, oa) in enumerate(sorted_hi):
                if oci != ci:
                    hi_total[ci] += 1
                    if a > oa:
                        hi_wins[ci] += 1

    result: List[Dict[str, float]] = []
    for ci in range(n_cand):
        feats: Dict[str, float] = {}
        # lowest-alpha relative features
        if lo_ranks[ci]:
            arr = np.array(lo_ranks[ci], dtype=np.float64)
            feats["lo_mean_rank"] = float(arr.mean())
            feats["lo_median_rank"] = float(np.median(arr))
            feats["lo_best_rank"] = float(arr.min())
        if lo_top1[ci]:
            feats["lo_top1_rate"] = float(np.mean(lo_top1[ci]))
        if lo_gap[ci]:
            feats["lo_mean_gap"] = float(np.mean(lo_gap[ci]))
        if lo_total[ci] > 0:
            feats["lo_dominance"] = lo_wins[ci] / lo_total[ci]

        # highest-alpha relative features
        if hi_ranks[ci]:
            arr = np.array(hi_ranks[ci], dtype=np.float64)
            feats["hi_mean_rank"] = float(arr.mean())
            feats["hi_median_rank"] = float(np.median(arr))
            feats["hi_best_rank"] = float(arr.min())
        if hi_top1[ci]:
            feats["hi_top1_rate"] = float(np.mean(hi_top1[ci]))
        if hi_gap[ci]:
            feats["hi_mean_gap"] = float(np.mean(hi_gap[ci]))
        if hi_total[ci] > 0:
            feats["hi_dominance"] = hi_wins[ci] / hi_total[ci]

        result.append(feats)
    return result



FEATURE_NAMES: List[str] = [
    # --- intrinsic alpha features ---
    "a_mean", "a_median", "a_std", "a_min", "a_max",
    "a_range", "a_q25", "a_q75", "a_iqr", "a_cv",
    "a_skew", "a_kurtosis",
    "a_frac_high", "a_frac_low",
    "a_frac_extreme_high", "a_frac_extreme_low",
    "a_entropy", "a_consistency",
    # --- source rank features ---
    "src_rank_min", "src_rank_mean", "src_rank_count",
    # --- relative: lowest-alpha ranking ---
    "lo_mean_rank", "lo_median_rank", "lo_best_rank",
    "lo_top1_rate", "lo_mean_gap", "lo_dominance",
    # --- relative: highest-alpha ranking ---
    "hi_mean_rank", "hi_median_rank", "hi_best_rank",
    "hi_top1_rate", "hi_mean_gap", "hi_dominance",
]



def _group_by_record(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("record_id", "")).strip()
        if rid:
            grouped[rid].append(row)
    return dict(grouped)


def _build_feature_rows(
    grouped: Dict[str, List[Dict[str, Any]]],
    *,
    min_combos: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    all_feature_rows: List[Dict[str, Any]] = []
    processed_rids: List[str] = []

    for rid in sorted(grouped.keys()):
        candidates = grouped[rid]
        valid = [
            c for c in candidates
            if isinstance(c.get("alpha_values"), list)
            and len(c["alpha_values"]) >= min_combos
        ]
        if len(valid) < 2:
            continue

        rel_feats_list = _extract_relative_alpha_features(valid)

        for ci, cand in enumerate(valid):
            alpha_arr = np.array(cand["alpha_values"], dtype=np.float64)
            intrinsic = _extract_alpha_intrinsic_features(alpha_arr)
            src_rank = _extract_source_rank_features(cand)
            relative = rel_feats_list[ci]

            merged: Dict[str, Any] = {}
            merged.update(intrinsic)
            merged.update(src_rank)
            merged.update(relative)

            # Metadata
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
# Correlation
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
            p = float("nan")  # unavailable without scipy; never report false significance
        results.append((fname, float(r), float(p)))

    results.sort(key=lambda x: -abs(x[1]))

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 1: POINT-BISERIAL CORRELATION  (alpha feature vs is_gt)\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Labelled candidates: {int(labelled.sum())}\n\n")
    out.write(f"  {'Feature':<42s} {'r':>8s} {'p-value':>12s} {'Direction':>12s}\n")
    out.write(f"  {'─' * 76}\n")
    for fname, r, p in results:
        direction = "GT higher" if r > 0 else "GT lower"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        out.write(f"  {fname:<42s} {r:+8.4f} {p:12.2e} {direction:>12s} {sig}\n")
    out.write("\n")
    out.write("  KEY: Positive r => GT has HIGHER values.\n")
    out.write("       Negative r => GT has LOWER values.\n")
    out.write("       *** p<0.001,  ** p<0.01,  * p<0.05\n\n")
    out.write("  INTERPRETATION:\n")
    out.write("    If GT has LOWER a_mean => GT needs LESS override (already correct).\n")
    out.write("    If GT has HIGHER a_mean => GT BENEFITS MORE from visual attention.\n")
    out.write("    Which pattern holds depends on your data — the correlation tells you.\n\n")
    return results


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _evaluate_heuristics(
    feature_rows: List[Dict[str, Any]],
    out: Any = sys.stdout,
) -> Dict[str, Dict[str, float]]:
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        if row.get("is_gt") is not None:
            by_record[row["record_id"]].append(row)

    # (name, feature_key, higher_is_better)
    heuristics = [
        # === alpha magnitude ===
        ("Lowest mean alpha",                   "a_mean",               False),
        ("Highest mean alpha",                  "a_mean",               True),
        ("Lowest median alpha",                 "a_median",             False),
        ("Highest median alpha",                "a_median",             True),
        ("Lowest std (most consistent)",        "a_std",                False),
        ("Lowest CV (most consistent)",         "a_cv",                 False),
        ("Highest consistency",                 "a_consistency",        True),
        ("Highest frac_low (<0.5)",             "a_frac_low",           True),
        ("Highest frac_high (>0.5)",            "a_frac_high",          True),
        ("Highest frac_extreme_high (>0.8)",    "a_frac_extreme_high",  True),
        ("Highest frac_extreme_low (<0.2)",     "a_frac_extreme_low",   True),
        # === source rank ===
        ("Best src_rank_min (lowest)",          "src_rank_min",         False),
        ("Best src_rank_mean (lowest)",         "src_rank_mean",        False),
        # === relative: lowest-alpha ranking ===
        ("Best lo_mean_rank (lowest)",          "lo_mean_rank",         False),
        ("Highest lo_top1_rate",                "lo_top1_rate",         True),
        ("Highest lo_dominance",                "lo_dominance",         True),
        ("Smallest lo_mean_gap",                "lo_mean_gap",          False),
        # === relative: highest-alpha ranking ===
        ("Best hi_mean_rank (lowest)",          "hi_mean_rank",         False),
        ("Highest hi_top1_rate",                "hi_top1_rate",         True),
        ("Highest hi_dominance",                "hi_dominance",         True),
        ("Smallest hi_mean_gap",                "hi_mean_gap",          False),
    ]

    n_records = len(by_record)
    results: Dict[str, Dict[str, float]] = {}

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 2: SIMPLE HEURISTICS  (rank candidates, check if GT is #1)\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Records with labelled GT + wrong: {n_records}\n\n")
    out.write(f"  {'Heuristic':<42s} {'GT=#1':>8s} {'GT<=2':>8s} {'MRR':>8s}\n")
    out.write(f"  {'─' * 68}\n")

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
        out.write(f"  {name:<42s} {acc1:7.1%} {acc2:7.1%} {mrr:8.4f}\n")

    out.write("\n")
    out.write("  KEY: GT=#1 = fraction where GT is ranked 1st by heuristic\n")
    out.write("       MRR   = Mean Reciprocal Rank (1.0 = always first)\n\n")
    out.write("  NOTE: We test BOTH directions for alpha mean/median because\n")
    out.write("  the signal depends on your data: GT might need MORE or LESS\n")
    out.write("  override. The correlation table (Part 1) tells you which.\n\n")
    return results


# ---------------------------------------------------------------------------
# Classifier
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
    for i, row in enumerate(feature_rows):
        if row.get("is_gt") is not None:
            by_record[row["record_id"]].append(i)

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
            pass

        importances_avg = importances_acc / n_eval
        imp_order = np.argsort(-importances_avg)
        out.write(f"\n  Top-10 feature importances ({clf_name}):\n")
        out.write(f"  {'Feature':<42s} {'Importance':>12s}\n")
        out.write(f"  {'─' * 56}\n")
        for rp in range(min(10, len(imp_order))):
            j = imp_order[rp]
            out.write(f"  {feature_names[j]:<42s} {importances_avg[j]:12.4f}\n")
        out.write("\n")

    return {"status": "done"}


# ---------------------------------------------------------------------------
# Unsupervised prediction
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
    out.write("  PART 4: UNSUPERVISED PREDICTIONS  (from alpha features)\n")
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
            gap = top["predicted_gt_prob"] - cands[1]["predicted_gt_prob"]
            out.write(f"  gap={gap:+.4f}")
        out.write("\n")
        for rp, c in enumerate(cands):
            predictions.append({
                "record_id": rid,
                "predicted_rank": rp + 1,
                "candidate_answer": c["candidate_answer"],
                "candidate_answer_norm": c.get("candidate_answer_norm", ""),
                "predicted_gt_prob": c["predicted_gt_prob"],
                "a_mean": c.get("a_mean", ""),
                "lo_top1_rate": c.get("lo_top1_rate", ""),
                "hi_top1_rate": c.get("hi_top1_rate", ""),
                "lo_dominance": c.get("lo_dominance", ""),
            })

    if out_csv and predictions:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = list(predictions[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for pred in predictions:
                f.write(",".join(str(pred.get(h, "")) for h in header) + "\n")
        out.write(f"\n  [info] Wrote predictions to {out_csv}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(out: Any = sys.stdout) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write("  INTERPRETATION & NEXT STEPS\n")
    out.write("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Features CSV
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
        description="Predict ground-truth answer from alpha distributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--record-jsonl", action="append", default=[],
        help=(
            "Per-record JSONL from ranked_combo_alpha_union.py "
            "(contains union_answer_alpha_distributions).  Repeat for multiple."
        ),
    )
    parser.add_argument(
        "--record-glob", default=None,
        help="Glob for per-record JSONL files.",
    )
    parser.add_argument(
        "--pair-jsonl", action="append", default=[],
        help="Per-pair JSONL from ranked_combo_alpha_union.py.  Repeat for multiple.",
    )
    parser.add_argument(
        "--pair-glob", default=None,
        help="Glob for per-pair JSONL files.",
    )
    parser.add_argument(
        "--gold-jsonl", action="append", default=[],
        help=(
            "JSONL with gold_answer_norm per record_id (needed if using --pair-jsonl "
            "without --record-jsonl).  Can be a per-record alpha JSONL or any JSONL "
            "with record_id + gold_answer_norm."
        ),
    )
    parser.add_argument(
        "--predict-jsonl", action="append", default=[],
        help="Unsupervised per-record JSONL to predict on.  Repeat for multiple.",
    )
    parser.add_argument(
        "--predict-glob", default=None,
        help="Glob for unsupervised per-record JSONL files.",
    )
    parser.add_argument(
        "--min-combos", type=int, default=5,
        help="Min alpha observations required per candidate (default 5).",
    )
    parser.add_argument("--out-features", default=None, help="Save extracted features CSV.")
    parser.add_argument("--out-predictions", default=None, help="Save prediction CSV.")
    parser.add_argument("--out-report", default=None, help="Save text report.")
    args = parser.parse_args(argv)


    candidate_rows: List[Dict[str, Any]] = []

    rec_paths = _collect_paths(args.record_jsonl, args.record_glob)
    if rec_paths:
        rec_rows = _load_rows(rec_paths)
        print(f"[info] Loaded {len(rec_rows)} per-record rows from {len(rec_paths)} file(s).")
        flat = _flatten_record_jsonl(rec_rows)
        print(f"[info] Flattened to {len(flat)} candidate rows.")
        candidate_rows.extend(flat)

    pair_paths = _collect_paths(args.pair_jsonl, args.pair_glob)
    if pair_paths:
        pair_rows = _load_rows(pair_paths)
        print(f"[info] Loaded {len(pair_rows)} per-pair rows from {len(pair_paths)} file(s).")
        gold_map: Dict[str, str] = {}
        for cr in candidate_rows:
            rid = cr.get("record_id", "")
            gn = cr.get("gold_answer_norm", "")
            if rid and gn:
                gold_map[rid] = gn
        gold_paths = _collect_paths(args.gold_jsonl, None)
        if gold_paths:
            gold_rows = _load_rows(gold_paths)
            for gr in gold_rows:
                rid = str(gr.get("record_id", "")).strip()
                gn = str(gr.get("gold_answer_norm", "")).strip()
                if rid and gn:
                    gold_map[rid] = gn
        flat = _flatten_pair_jsonl(pair_rows, gold_map)
        print(f"[info] Flattened pairs to {len(flat)} candidate rows.")
        candidate_rows.extend(flat)

    if not candidate_rows:
        print("[error] No data loaded.  Provide --record-jsonl or --pair-jsonl.")
        return 1

    has_arrays = any(
        isinstance(r.get("alpha_values"), list) and len(r["alpha_values"]) > 0
        for r in candidate_rows
    )
    if not has_arrays:
        print("[error] No alpha_values arrays found.")
        return 1

    grouped = _group_by_record(candidate_rows)
    print(f"[info] {len(grouped)} records with candidates.")

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    feature_rows, processed_rids = _build_feature_rows(
        grouped, min_combos=args.min_combos,
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
        rp = Path(args.out_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        report_file = rp.open("w", encoding="utf-8")

    class TeeWriter:
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
    # Analysis
    # ------------------------------------------------------------------
    _correlation_analysis(feature_rows, FEATURE_NAMES, out=out)
    _evaluate_heuristics(feature_rows, out=out)
    _classifier_evaluation(feature_rows, FEATURE_NAMES, out=out)

    # ------------------------------------------------------------------
    # Unsupervised prediction
    # ------------------------------------------------------------------
    predict_paths = _collect_paths(args.predict_jsonl, args.predict_glob)
    if predict_paths:
        pred_rows = _load_rows(predict_paths)
        print(f"[info] Loaded {len(pred_rows)} unsupervised rows from {len(predict_paths)} file(s).")
        pred_flat = _flatten_record_jsonl(pred_rows)
        pred_grouped = _group_by_record(pred_flat)
        pred_feature_rows, _ = _build_feature_rows(pred_grouped, min_combos=args.min_combos)
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

    if args.out_features:
        _write_features_csv(feature_rows, FEATURE_NAMES, Path(args.out_features))

    if report_file:
        report_file.close()
        print(f"[info] Report saved to {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
