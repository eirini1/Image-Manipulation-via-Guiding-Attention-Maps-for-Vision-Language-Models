from __future__ import annotations

import argparse
import glob as glob_mod
import json
import re
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


_EPS = 1e-30


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


def _normalize_answer(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _flatten_hide_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("union_answer_hide_distributions"), list):
            record_id = str(row.get("record_id", row.get("id", ""))).strip()
            if not record_id:
                continue
            gold_text = str(row.get("gold_answer", row.get("answer", ""))).strip()
            gold_norm = str(row.get("gold_answer_norm", _normalize_answer(gold_text))).strip()
            distributions = row.get("union_answer_hide_distributions", [])
            for ans_entry in distributions:
                if not isinstance(ans_entry, dict):
                    continue
                cand_text = str(ans_entry.get("answer", "")).strip()
                if not cand_text:
                    continue
                cand_norm = str(ans_entry.get("answer_norm", _normalize_answer(cand_text))).strip()
                if not cand_norm:
                    continue
                is_gt = None
                if gold_norm:
                    is_gt = 1 if cand_norm == gold_norm else 0
                hide_vals = ans_entry.get("hide_values", [])
                objh_vals = ans_entry.get("object_hidden_values", [])
                globh_vals = ans_entry.get("global_hidden_values", [])
                if not isinstance(hide_vals, list):
                    hide_vals = []
                if not isinstance(objh_vals, list):
                    objh_vals = []
                if not isinstance(globh_vals, list):
                    globh_vals = []
                for i, hv in enumerate(hide_vals):
                    objv = objh_vals[i] if i < len(objh_vals) else None
                    globv = globh_vals[i] if i < len(globh_vals) else None
                    out.append(
                        {
                            "record_id": record_id,
                            "candidate_answer": cand_text,
                            "candidate_answer_norm": cand_norm,
                            "gold_answer_norm": gold_norm,
                            "is_gt_candidate": is_gt,
                            "best_hide_strength": _to_float_or_none(hv),
                            "object_hidden_ratio_weighted": _to_float_or_none(objv),
                            "global_hidden_ratio_mean": _to_float_or_none(globv),
                            "delta_logprob": None,
                            "opt_target": "ranked_combo_hide_union",
                        }
                    )
            continue

        record_id = str(row.get("id", row.get("record_id", ""))).strip()
        if not record_id:
            continue

        cand_text = str(row.get("opt_answer", row.get("answer_candidate", ""))).strip()
        if not cand_text:
            continue

        gold_text = str(row.get("answer", row.get("gold_answer", ""))).strip()
        cand_norm = _normalize_answer(cand_text)
        gold_norm = _normalize_answer(gold_text)
        if not cand_norm:
            continue

        is_gt = None
        if gold_norm:
            is_gt = 1 if cand_norm == gold_norm else 0

        out.append(
            {
                "record_id": record_id,
                "candidate_answer": cand_text,
                "candidate_answer_norm": cand_norm,
                "gold_answer_norm": gold_norm,
                "is_gt_candidate": is_gt,
                "best_hide_strength": _to_float_or_none(row.get("best_hide_strength", row.get("best_hide"))),
                "object_hidden_ratio_weighted": _to_float_or_none(row.get("object_hidden_ratio_weighted")),
                "global_hidden_ratio_mean": _to_float_or_none(row.get("global_hidden_ratio_mean")),
                "delta_logprob": _to_float_or_none(row.get("delta_logprob")),
                "opt_target": row.get("opt_target"),
            }
        )
    return out


def _aggregate_candidate_vectors(
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for row in candidate_rows:
        rid = str(row.get("record_id", "")).strip()
        c_norm = str(row.get("candidate_answer_norm", "")).strip()
        if not rid or not c_norm:
            continue

        rec = grouped[rid]
        if c_norm not in rec:
            rec[c_norm] = {
                "record_id": rid,
                "candidate_answer": str(row.get("candidate_answer", "")),
                "candidate_answer_norm": c_norm,
                "gold_answer_norm": str(row.get("gold_answer_norm", "")),
                "is_gt_candidate": row.get("is_gt_candidate"),
                "hide_values": [],
                "object_hidden_values": [],
                "global_hidden_values": [],
                "delta_lp_values": [],
            }

        dst = rec[c_norm]
        h = row.get("best_hide_strength")
        if h is not None:
            dst["hide_values"].append(float(h))
        oh = row.get("object_hidden_ratio_weighted")
        if oh is not None:
            dst["object_hidden_values"].append(float(oh))
        gh = row.get("global_hidden_ratio_mean")
        if gh is not None:
            dst["global_hidden_values"].append(float(gh))
        dl = row.get("delta_logprob")
        if dl is not None:
            dst["delta_lp_values"].append(float(dl))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for rid, by_ans in grouped.items():
        out[rid] = list(by_ans.values())
    return out


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


def _extract_dist_features(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    n = len(arr)
    if n == 0:
        return {}
    mean_val = float(arr.mean())
    median_val = float(np.median(arr))
    std_val = float(arr.std(ddof=1)) if n > 1 else 0.0
    min_val = float(arr.min())
    max_val = float(arr.max())
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    out = {
        f"{prefix}_mean": mean_val,
        f"{prefix}_median": median_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_range": float(max_val - min_val),
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_cv": float(std_val / (abs(mean_val) + _EPS)),
        f"{prefix}_skew": _safe_skew(arr),
        f"{prefix}_kurtosis": _safe_kurtosis(arr),
        f"{prefix}_n": float(n),
    }

    # Useful hide-specific rates
    out[f"{prefix}_frac_gt_05"] = float(np.mean(arr > 0.5))
    out[f"{prefix}_frac_gt_08"] = float(np.mean(arr > 0.8))
    out[f"{prefix}_frac_lt_02"] = float(np.mean(arr < 0.2))
    return out


def _extract_relative_hide_features(record_candidates: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    vals: List[Tuple[int, float]] = []
    for idx, c in enumerate(record_candidates):
        hide_vals = c.get("hide_values", [])
        if not isinstance(hide_vals, list) or not hide_vals:
            continue
        vals.append((idx, float(np.mean(np.asarray(hide_vals, dtype=np.float64)))))

    if len(vals) < 2:
        return [{} for _ in record_candidates]

    out: List[Dict[str, float]] = [dict() for _ in record_candidates]

    sorted_lo = sorted(vals, key=lambda x: x[1])
    sorted_hi = sorted(vals, key=lambda x: -x[1])

    min_v = sorted_lo[0][1]
    max_v = sorted_hi[0][1]

    for rank, (ci, v) in enumerate(sorted_lo, 1):
        out[ci]["lo_rank"] = float(rank)
        out[ci]["lo_top1"] = 1.0 if rank == 1 else 0.0
        out[ci]["lo_gap"] = float(v - min_v)

    for rank, (ci, v) in enumerate(sorted_hi, 1):
        out[ci]["hi_rank"] = float(rank)
        out[ci]["hi_top1"] = 1.0 if rank == 1 else 0.0
        out[ci]["hi_gap"] = float(max_v - v)

    n = float(len(vals))
    for ci, _ in vals:
        lo_rank = out[ci].get("lo_rank", n)
        hi_rank = out[ci].get("hi_rank", n)
        out[ci]["lo_dominance"] = float((n - lo_rank) / max(1.0, n - 1.0))
        out[ci]["hi_dominance"] = float((n - hi_rank) / max(1.0, n - 1.0))

    return out


FEATURE_NAMES: List[str] = [
    # hide scalar distribution
    "h_mean",
    "h_median",
    "h_std",
    "h_min",
    "h_max",
    "h_range",
    "h_q25",
    "h_q75",
    "h_iqr",
    "h_cv",
    "h_skew",
    "h_kurtosis",
    "h_n",
    "h_frac_gt_05",
    "h_frac_gt_08",
    "h_frac_lt_02",
    # object/global hidden ratios
    "objh_mean",
    "objh_std",
    "globh_mean",
    "globh_std",
    # delta logprob stats
    "dlp_mean",
    "dlp_std",
    # relative features
    "lo_rank",
    "lo_top1",
    "lo_gap",
    "lo_dominance",
    "hi_rank",
    "hi_top1",
    "hi_gap",
    "hi_dominance",
]


def _build_feature_rows(
    grouped: Dict[str, List[Dict[str, Any]]],
    *,
    min_observations: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    feature_rows: List[Dict[str, Any]] = []
    processed_records: List[str] = []

    for rid in sorted(grouped.keys()):
        candidates = grouped[rid]
        valid: List[Dict[str, Any]] = []
        for c in candidates:
            hide_vals = c.get("hide_values", [])
            if isinstance(hide_vals, list) and len(hide_vals) >= int(min_observations):
                valid.append(c)

        if len(valid) < 2:
            continue

        rel_feats = _extract_relative_hide_features(valid)

        for ci, cand in enumerate(valid):
            h = np.asarray(cand.get("hide_values", []), dtype=np.float64)
            objh = np.asarray(cand.get("object_hidden_values", []), dtype=np.float64)
            globh = np.asarray(cand.get("global_hidden_values", []), dtype=np.float64)
            dlp = np.asarray(cand.get("delta_lp_values", []), dtype=np.float64)

            row: Dict[str, Any] = {}
            row.update(_extract_dist_features("h", h))
            if objh.size > 0:
                row["objh_mean"] = float(objh.mean())
                row["objh_std"] = float(objh.std(ddof=1)) if objh.size > 1 else 0.0
            if globh.size > 0:
                row["globh_mean"] = float(globh.mean())
                row["globh_std"] = float(globh.std(ddof=1)) if globh.size > 1 else 0.0
            if dlp.size > 0:
                row["dlp_mean"] = float(dlp.mean())
                row["dlp_std"] = float(dlp.std(ddof=1)) if dlp.size > 1 else 0.0

            row.update(rel_feats[ci])

            row["record_id"] = rid
            row["candidate_answer"] = str(cand.get("candidate_answer", ""))
            row["candidate_answer_norm"] = str(cand.get("candidate_answer_norm", ""))
            row["gold_answer_norm"] = str(cand.get("gold_answer_norm", ""))
            is_gt = cand.get("is_gt_candidate")
            row["is_gt"] = int(is_gt) if is_gt is not None else None

            feature_rows.append(row)

        processed_records.append(rid)

    return feature_rows, processed_records


def _rows_to_matrix(feature_rows: List[Dict[str, Any]], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
# Analysis blocks
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
    out.write("  PART 1: CORRELATION  (hide feature vs is_gt)\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Labelled candidates: {int(labelled.sum())}\n\n")
    out.write(f"  {'Feature':<30s} {'r':>8s} {'p-value':>12s} {'Direction':>12s}\n")
    out.write(f"  {'─' * 68}\n")
    for fname, r, p in results:
        direction = "GT higher" if r > 0 else "GT lower"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        out.write(f"  {fname:<30s} {r:+8.4f} {p:12.2e} {direction:>12s} {sig}\n")

    out.write("\n")
    out.write("  Interpretation for hide metrics:\n")
    out.write("    Negative r on h_mean/objh_mean/globh_mean => GT tends to hide less.\n")
    out.write("    Positive r on these metrics               => GT tends to hide more.\n")
    out.write("\n")
    return results


def _evaluate_heuristics(feature_rows: List[Dict[str, Any]], out: Any = sys.stdout) -> Dict[str, Dict[str, float]]:
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        if row.get("is_gt") is not None:
            by_record[row["record_id"]].append(row)

    heuristics = [
        ("Lowest hide mean", "h_mean", False),
        ("Highest hide mean", "h_mean", True),
        ("Lowest object-hidden mean", "objh_mean", False),
        ("Highest object-hidden mean", "objh_mean", True),
        ("Lowest global-hidden mean", "globh_mean", False),
        ("Highest global-hidden mean", "globh_mean", True),
        ("Best low-hide rank", "lo_rank", False),
        ("Highest low-hide dominance", "lo_dominance", True),
        ("Best high-hide rank", "hi_rank", False),
        ("Highest high-hide dominance", "hi_dominance", True),
    ]

    results: Dict[str, Dict[str, float]] = {}

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 2: SIMPLE HEURISTICS\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Records with labelled GT: {len(by_record)}\n\n")
    out.write(f"  {'Heuristic':<34s} {'GT=#1':>8s} {'GT<=2':>8s} {'MRR':>8s}\n")
    out.write(f"  {'─' * 64}\n")

    for name, fkey, higher_is_better in heuristics:
        correct_at_1 = 0
        correct_at_2 = 0
        reciprocal_ranks: List[float] = []

        for _, cands in by_record.items():
            with_feat = [(c, c.get(fkey)) for c in cands if c.get(fkey) is not None]
            if not with_feat:
                continue
            with_feat.sort(key=lambda x: x[1], reverse=higher_is_better)
            for rank_pos, (cand, _) in enumerate(with_feat):
                if cand.get("is_gt") == 1:
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
        results[name] = {
            "accuracy@1": acc1,
            "accuracy@2": acc2,
            "mrr": mrr,
            "n_eval": float(n_eval),
        }
        out.write(f"  {name:<34s} {acc1:7.1%} {acc2:7.1%} {mrr:8.4f}\n")

    out.write("\n")
    out.write("  Key hypothesis check: if 'Lowest hide mean' wins,\n")
    out.write("  then stronger hiding is associated with wrong answers.\n\n")
    return results


def _classifier_evaluation(
    feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    out: Any = sys.stdout,
) -> Optional[Dict[str, Any]]:
    if not HAS_SKLEARN:
        out.write("\n" + "=" * 80 + "\n")
        out.write("  PART 3: CLASSIFIER (SKIPPED — install scikit-learn)\n")
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
    record_ids = sorted(by_record.keys())

    def _impute_and_scale(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = medians[j]
            X_test[np.isnan(X_test[:, j]), j] = medians[j]
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test)

    for clf_name, make_clf in [
        (
            "LogisticRegression",
            lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs"),
        ),
        (
            "RandomForest",
            lambda: RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
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
        out.write("\n  Top-10 feature importances:\n")
        out.write(f"  {'Feature':<30s} {'Importance':>12s}\n")
        out.write(f"  {'─' * 46}\n")
        for rp in range(min(10, len(imp_order))):
            j = imp_order[rp]
            out.write(f"  {feature_names[j]:<30s} {importances_avg[j]:12.4f}\n")
        out.write("\n")

    return {"status": "done"}


def _predict_unsupervised(
    train_feature_rows: List[Dict[str, Any]],
    predict_feature_rows: List[Dict[str, Any]],
    feature_names: List[str],
    out_csv: Optional[Path],
    out: Any = sys.stdout,
) -> None:
    if not HAS_SKLEARN:
        out.write("\n  [skip] scikit-learn not installed; cannot predict.\n")
        return

    train_rows = [r for r in train_feature_rows if r.get("is_gt") is not None]
    if len(train_rows) < 20:
        out.write("\n  [skip] Too few labelled rows to train a reliable predictor.\n")
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
    probs = clf.predict_proba(X_pred)[:, 1]

    for i, row in enumerate(predict_feature_rows):
        row["predicted_gt_prob"] = float(probs[i])

    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predict_feature_rows:
        by_record[str(row.get("record_id", ""))].append(row)

    out.write("\n" + "=" * 80 + "\n")
    out.write("  PART 4: UNSUPERVISED PREDICTIONS\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Records to predict: {len(by_record)}\n\n")

    predictions: List[Dict[str, Any]] = []
    for rid in sorted(by_record.keys()):
        cands = sorted(by_record[rid], key=lambda r: -float(r.get("predicted_gt_prob", 0.0)))
        if not cands:
            continue
        top = cands[0]
        out.write(
            f"  {rid:>12s}  predicted GT: \"{top.get('candidate_answer', '')}\""
            f"  P(GT)={float(top.get('predicted_gt_prob', 0.0)):.4f}"
        )
        if len(cands) > 1:
            gap = float(top.get("predicted_gt_prob", 0.0)) - float(cands[1].get("predicted_gt_prob", 0.0))
            out.write(f"  gap={gap:+.4f}")
        out.write("\n")

        for rp, cand in enumerate(cands, 1):
            predictions.append(
                {
                    "record_id": rid,
                    "predicted_rank": rp,
                    "candidate_answer": cand.get("candidate_answer", ""),
                    "candidate_answer_norm": cand.get("candidate_answer_norm", ""),
                    "predicted_gt_prob": float(cand.get("predicted_gt_prob", 0.0)),
                    "h_mean": cand.get("h_mean", ""),
                    "objh_mean": cand.get("objh_mean", ""),
                    "globh_mean": cand.get("globh_mean", ""),
                    "lo_rank": cand.get("lo_rank", ""),
                    "hi_rank": cand.get("hi_rank", ""),
                }
            )

    if out_csv and predictions:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = list(predictions[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for row in predictions:
                f.write(",".join(str(row.get(h, "")) for h in header) + "\n")
        out.write(f"\n  [info] Wrote predictions to {out_csv}\n")


def _write_features_csv(feature_rows: List[Dict[str, Any]], feature_names: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = ["record_id", "candidate_answer", "candidate_answer_norm", "is_gt"]
    header = meta_cols + feature_names
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in feature_rows:
            vals: List[str] = []
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


def _print_summary(out: Any = sys.stdout) -> None:
    out.write("\n" + "=" * 80 + "\n")
    out.write("  INTERPRETATION\n")
    out.write("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict ground-truth answers from hide-object optimization outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hide-jsonl",
        action="append",
        default=[],
        help="Input fine_tuning_hide_object JSONL file(s). Repeat for multiple.",
    )
    parser.add_argument("--hide-glob", default=None, help="Glob for fine_tuning_hide_object JSONL files.")
    parser.add_argument(
        "--predict-jsonl",
        action="append",
        default=[],
        help="Unlabelled hide JSONL file(s) for prediction.",
    )
    parser.add_argument("--predict-glob", default=None, help="Glob for unlabelled hide JSONL files.")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=1,
        help="Minimum hide observations required per candidate (default 1).",
    )
    parser.add_argument("--out-features", default=None, help="Write extracted feature CSV.")
    parser.add_argument("--out-predictions", default=None, help="Write unsupervised predictions CSV.")
    parser.add_argument("--out-report", default=None, help="Write text report.")
    args = parser.parse_args(argv)

    if int(args.min_observations) <= 0:
        parser.error("--min-observations must be > 0.")

    hide_paths = _collect_paths(args.hide_jsonl, args.hide_glob)
    if not hide_paths:
        parser.error("Provide --hide-jsonl and/or --hide-glob.")

    hide_rows_raw = _load_rows(hide_paths)
    if not hide_rows_raw:
        print("[error] No hide rows loaded.")
        return 1

    flat_rows = _flatten_hide_rows(hide_rows_raw)
    if not flat_rows:
        print("[error] Could not parse any candidate hide rows.")
        return 1

    grouped = _aggregate_candidate_vectors(flat_rows)
    feature_rows, processed = _build_feature_rows(grouped, min_observations=int(args.min_observations))

    print(f"[info] Loaded raw rows: {len(hide_rows_raw)}")
    print(f"[info] Flattened candidate rows: {len(flat_rows)}")
    print(f"[info] Processed records: {len(processed)}")
    print(f"[info] Feature rows: {len(feature_rows)}")

    if not feature_rows:
        print("[error] No feature rows (check --min-observations and input schema).")
        return 1

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

    _correlation_analysis(feature_rows, FEATURE_NAMES, out=out)
    _evaluate_heuristics(feature_rows, out=out)
    _classifier_evaluation(feature_rows, FEATURE_NAMES, out=out)

    predict_paths = _collect_paths(args.predict_jsonl, args.predict_glob)
    if predict_paths:
        pred_raw = _load_rows(predict_paths)
        pred_flat = _flatten_hide_rows(pred_raw)
        pred_grouped = _aggregate_candidate_vectors(pred_flat)
        pred_features, _ = _build_feature_rows(pred_grouped, min_observations=int(args.min_observations))
        if pred_features:
            _predict_unsupervised(
                feature_rows,
                pred_features,
                FEATURE_NAMES,
                out_csv=Path(args.out_predictions) if args.out_predictions else None,
                out=out,
            )

    _print_summary(out=out)

    if args.out_features:
        _write_features_csv(feature_rows, FEATURE_NAMES, Path(args.out_features))

    if report_file:
        report_file.close()
        print(f"[info] Report saved to {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
