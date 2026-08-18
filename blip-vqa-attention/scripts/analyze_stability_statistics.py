import argparse
import glob as glob_mod
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False



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


# ---------------------------------------------------------------------------
# Group rows by record -> list of candidate rows
# ---------------------------------------------------------------------------

def _group_by_record(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("record_id", "")).strip()
        if not rid:
            continue
        grouped[rid].append(row)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Paired statistical tests
# ---------------------------------------------------------------------------

def _paired_wilcoxon(diff: np.ndarray) -> Tuple[float, float]:
    nonzero = diff[diff != 0.0]
    if len(nonzero) < 2:
        return 0.0, 1.0
    if HAS_SCIPY:
        res = sp_stats.wilcoxon(nonzero, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    abs_ranks = np.empty_like(nonzero)
    order = np.abs(nonzero).argsort()
    abs_ranks[order] = np.arange(1, len(nonzero) + 1, dtype=np.float64)
    abs_vals = np.abs(nonzero)
    unique_vals, counts = np.unique(abs_vals, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        if cnt > 1:
            mask = abs_vals == val
            abs_ranks[mask] = abs_ranks[mask].mean()
    W_plus = abs_ranks[nonzero > 0].sum()
    W_minus = abs_ranks[nonzero < 0].sum()
    W = min(W_plus, W_minus)
    n = len(nonzero)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return float(W), 1.0
    z = (W - mu) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(W), float(p)


def _paired_t_test(diff: np.ndarray) -> Tuple[float, float]:
    n = len(diff)
    if n < 2:
        return 0.0, 1.0
    if HAS_SCIPY:
        res = sp_stats.ttest_1samp(diff, 0.0)
        return float(res.statistic), float(res.pvalue)
    m = diff.mean()
    se = diff.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0, 1.0
    t = m / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return float(t), float(p)


def _paired_permutation_test(
    diff: np.ndarray,
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    observed = abs(diff.mean())
    n = len(diff)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = abs((diff * signs).mean())
        if perm_mean >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def _paired_bootstrap_ci(
    diff: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    n = len(diff)
    rng = np.random.RandomState(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = diff[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, float(means.mean()), hi


def _paired_cohens_d(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return 0.0
    s = diff.std(ddof=1)
    if s == 0:
        return 0.0
    return float(diff.mean() / s)


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = min(p * (n - rank), 1.0)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = cummax
    return adjusted


# ---------------------------------------------------------------------------
# Per-record analysis
# ---------------------------------------------------------------------------

def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _analyze_one_record(
    record_id: str,
    candidates: List[Dict[str, Any]],
    *,
    n_permutations: int,
    n_bootstrap: int,
    alpha: float,
    min_combos: int,
) -> Optional[Dict[str, Any]]:

    gt_candidates = [c for c in candidates if c.get("is_gt_candidate") == 1]
    wrong_candidates = [c for c in candidates if c.get("is_gt_candidate") == 0]

    if not gt_candidates or not wrong_candidates:
        return None

    gt = gt_candidates[0]
    gt_vals = gt.get("after_logprob_values", [])
    gt_combo_idx = gt.get("combo_indices", [])

    if not gt_vals:
        return None

    gt_by_combo: Dict[int, float] = {}
    for ci, val in zip(gt_combo_idx, gt_vals):
        gt_by_combo[int(ci)] = float(val)

    pair_results: List[Dict[str, Any]] = []
    raw_wilcoxon_p: List[float] = []
    raw_ttest_p: List[float] = []
    raw_perm_p: List[float] = []

    for wrong in wrong_candidates:
        wrong_vals = wrong.get("after_logprob_values", [])
        wrong_combo_idx = wrong.get("combo_indices", [])
        if not wrong_vals:
            continue

        wrong_by_combo: Dict[int, float] = {}
        for ci, val in zip(wrong_combo_idx, wrong_vals):
            wrong_by_combo[int(ci)] = float(val)

        common_combos = sorted(set(gt_by_combo.keys()) & set(wrong_by_combo.keys()))
        if len(common_combos) < min_combos:
            continue

        gt_arr = np.array([gt_by_combo[c] for c in common_combos], dtype=np.float64)
        wrong_arr = np.array([wrong_by_combo[c] for c in common_combos], dtype=np.float64)
        diff = gt_arr - wrong_arr

        W, p_wilcoxon = _paired_wilcoxon(diff)
        t_stat, p_ttest = _paired_t_test(diff)
        p_perm = _paired_permutation_test(diff, n_permutations=n_permutations)
        ci_lo, ci_mean, ci_hi = _paired_bootstrap_ci(diff, n_bootstrap=n_bootstrap, alpha=alpha)
        d = _paired_cohens_d(diff)

        gt_wins = int(np.sum(diff > 0))
        wrong_wins = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        raw_wilcoxon_p.append(p_wilcoxon)
        raw_ttest_p.append(p_ttest)
        raw_perm_p.append(p_perm)

        pair_results.append({
            "wrong_answer": str(wrong.get("candidate_answer", "")),
            "wrong_answer_norm": str(wrong.get("candidate_answer_norm", "")),
            "wrong_rank": int(wrong.get("candidate_rank", -1)),
            "n_paired_combos": len(common_combos),
            "gt_mean_logprob": float(gt_arr.mean()),
            "wrong_mean_logprob": float(wrong_arr.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "gt_wins": gt_wins,
            "wrong_wins": wrong_wins,
            "ties": ties,
            "wilcoxon_W": float(W),
            "wilcoxon_p": float(p_wilcoxon),
            "paired_t": float(t_stat),
            "paired_t_p": float(p_ttest),
            "permutation_p": float(p_perm),
            "cohens_d": float(d),
            f"bootstrap_{int(100*(1-alpha))}ci_lower": float(ci_lo),
            f"bootstrap_{int(100*(1-alpha))}ci_mean": float(ci_mean),
            f"bootstrap_{int(100*(1-alpha))}ci_upper": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        })

    if not pair_results:
        return None

    adj_wilcoxon = _holm_bonferroni(raw_wilcoxon_p)
    adj_ttest = _holm_bonferroni(raw_ttest_p)
    adj_perm = _holm_bonferroni(raw_perm_p)
    for i, pr in enumerate(pair_results):
        pr["wilcoxon_p_holm"] = float(adj_wilcoxon[i])
        pr["paired_t_p_holm"] = float(adj_ttest[i])
        pr["permutation_p_holm"] = float(adj_perm[i])

    n_sig_wilcoxon = sum(1 for pr in pair_results if pr["wilcoxon_p_holm"] < alpha)
    n_sig_ttest = sum(1 for pr in pair_results if pr["paired_t_p_holm"] < alpha)
    n_sig_perm = sum(1 for pr in pair_results if pr["permutation_p_holm"] < alpha)
    all_sig = all(
        pr["wilcoxon_p_holm"] < alpha and pr["permutation_p_holm"] < alpha
        for pr in pair_results
    )

    return {
        "record_id": record_id,
        "gt_answer": str(gt.get("candidate_answer", "")),
        "gt_answer_norm": str(gt.get("gold_answer_norm", "")),
        "n_wrong_answers": len(pair_results),
        "n_significant_wilcoxon": n_sig_wilcoxon,
        "n_significant_ttest": n_sig_ttest,
        "n_significant_perm": n_sig_perm,
        "all_wrong_significant": all_sig,
        "pairs": pair_results,
    }


# ---------------------------------------------------------------------------
# Print per-record results
# ---------------------------------------------------------------------------

def _print_record_result(res: Dict[str, Any], alpha: float) -> None:
    ci_pct = int(100 * (1 - alpha))
    rid = res["record_id"]
    gt_ans = res["gt_answer"]
    n_wrong = res["n_wrong_answers"]
    n_sig_w = res["n_significant_wilcoxon"]
    n_sig_p = res["n_significant_perm"]
    all_sig = res["all_wrong_significant"]

    verdict = "ALL SIGNIFICANT" if all_sig else f"{n_sig_w}/{n_wrong} significant"
    print(f"\n  Record: {rid}   GT answer: \"{gt_ans}\"   wrong answers: {n_wrong}   -> {verdict}")
    print(f"  {'─' * 90}")

    for pr in res["pairs"]:
        wrong = pr["wrong_answer"]
        n = pr["n_paired_combos"]
        md = pr["mean_diff"]
        gw = pr["gt_wins"]
        ww = pr["wrong_wins"]
        d = pr["cohens_d"]
        p_w_holm = pr["wilcoxon_p_holm"]
        p_t_holm = pr["paired_t_p_holm"]
        p_p_holm = pr["permutation_p_holm"]
        ci_lo = pr[f"bootstrap_{ci_pct}ci_lower"]
        ci_hi = pr[f"bootstrap_{ci_pct}ci_upper"]
        ci_zero = pr["ci_excludes_zero"]

        star = _significance_stars(min(p_w_holm, p_p_holm))
        print(
            f"    vs \"{wrong}\"  (n={n})  mean_diff={md:+.4f}  "
            f"GT_wins={gw} wrong_wins={ww}  d={d:+.3f}  "
            f"p_wilcox_holm={p_w_holm:.4f} p_perm_holm={p_p_holm:.4f}  "
            f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] excl0={'Y' if ci_zero else 'N'}  {star}"
        )


def _print_aggregate_summary(
    all_record_results: Sequence[Dict[str, Any]],
    alpha: float,
) -> None:
    n_records = len(all_record_results)
    n_all_sig = sum(1 for r in all_record_results if r["all_wrong_significant"])

    total_pairs = sum(r["n_wrong_answers"] for r in all_record_results)
    total_sig_wilcoxon = sum(r["n_significant_wilcoxon"] for r in all_record_results)
    total_sig_perm = sum(r["n_significant_perm"] for r in all_record_results)

    print()
    print("=" * 100)
    print("  AGGREGATE SUMMARY ACROSS ALL RECORDS")
    print("=" * 100)
    print(f"    Records tested:                    {n_records}")
    if n_records:
        print(f"    Records where ALL wrong answers")
        print(f"      are significantly separated:     {n_all_sig}/{n_records}  ({100*n_all_sig/n_records:.1f}%)")
    if total_pairs:
        print(f"    Total GT-vs-wrong pairs:           {total_pairs}")
        print(f"    Significant pairs (Wilcoxon+Holm): {total_sig_wilcoxon}/{total_pairs}  ({100*total_sig_wilcoxon/total_pairs:.1f}%)")
        print(f"    Significant pairs (Permut.+Holm):  {total_sig_perm}/{total_pairs}  ({100*total_sig_perm/total_pairs:.1f}%)")

    # Collect all Cohen's d values and gt-win rates
    all_d: List[float] = []
    all_gt_win_rate: List[float] = []
    for r in all_record_results:
        for pr in r["pairs"]:
            all_d.append(pr["cohens_d"])
            n = pr["n_paired_combos"]
            if n > 0:
                all_gt_win_rate.append(pr["gt_wins"] / n)
    if all_d:
        d_arr = np.array(all_d, dtype=np.float64)
        gw_arr = np.array(all_gt_win_rate, dtype=np.float64)
        print(f"    Cohen's d across all pairs:        mean={d_arr.mean():+.4f}  median={float(np.median(d_arr)):+.4f}  std={d_arr.std():.4f}")
        print(f"    GT-win rate across all pairs:       mean={gw_arr.mean():.4f}  median={float(np.median(gw_arr)):.4f}")



# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(all_record_results: Sequence[Dict[str, Any]], path: Path, alpha: float) -> None:
    ci_pct = int(100 * (1 - alpha))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "record_id", "gt_answer", "gt_answer_norm",
        "wrong_answer", "wrong_answer_norm", "wrong_rank",
        "n_paired_combos",
        "gt_mean_logprob", "wrong_mean_logprob", "mean_diff", "std_diff",
        "gt_wins", "wrong_wins", "ties",
        "wilcoxon_W", "wilcoxon_p", "wilcoxon_p_holm",
        "paired_t", "paired_t_p", "paired_t_p_holm",
        "permutation_p", "permutation_p_holm",
        "cohens_d",
        f"bootstrap_{ci_pct}ci_lower", f"bootstrap_{ci_pct}ci_mean", f"bootstrap_{ci_pct}ci_upper",
        "ci_excludes_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for rec in all_record_results:
            rid = rec["record_id"]
            gt_ans = rec["gt_answer"]
            gt_norm = rec["gt_answer_norm"]
            for pr in rec["pairs"]:
                vals = [
                    rid, gt_ans, gt_norm,
                    pr["wrong_answer"], pr["wrong_answer_norm"], str(pr["wrong_rank"]),
                    str(pr["n_paired_combos"]),
                    f"{pr['gt_mean_logprob']:.8f}", f"{pr['wrong_mean_logprob']:.8f}",
                    f"{pr['mean_diff']:.8f}", f"{pr['std_diff']:.8f}",
                    str(pr["gt_wins"]), str(pr["wrong_wins"]), str(pr["ties"]),
                    f"{pr['wilcoxon_W']:.4f}", f"{pr['wilcoxon_p']:.8f}", f"{pr['wilcoxon_p_holm']:.8f}",
                    f"{pr['paired_t']:.4f}", f"{pr['paired_t_p']:.8f}", f"{pr['paired_t_p_holm']:.8f}",
                    f"{pr['permutation_p']:.8f}", f"{pr['permutation_p_holm']:.8f}",
                    f"{pr['cohens_d']:.6f}",
                    f"{pr[f'bootstrap_{ci_pct}ci_lower']:.8f}",
                    f"{pr[f'bootstrap_{ci_pct}ci_mean']:.8f}",
                    f"{pr[f'bootstrap_{ci_pct}ci_upper']:.8f}",
                    str(pr["ci_excludes_zero"]),
                ]
                f.write(",".join(vals) + "\n")
    print(f"[info] Wrote per-pair CSV to {path}")


def _write_record_summary_csv(
    all_record_results: Sequence[Dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "record_id", "gt_answer", "n_wrong_answers",
        "n_significant_wilcoxon", "n_significant_ttest", "n_significant_perm",
        "all_wrong_significant",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for rec in all_record_results:
            vals = [
                rec["record_id"], rec["gt_answer"], str(rec["n_wrong_answers"]),
                str(rec["n_significant_wilcoxon"]), str(rec["n_significant_ttest"]),
                str(rec["n_significant_perm"]), str(rec["all_wrong_significant"]),
            ]
            f.write(",".join(vals) + "\n")
    print(f"[info] Wrote per-record summary CSV to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-record paired statistical tests: is the correct answer's logprob "
            "distribution across combos significantly different from each wrong answer?"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jsonl", action="append", default=[],
        help="candidate_stability_features JSONL (repeat for multiple).",
    )
    parser.add_argument(
        "--glob", default=None,
        help="Glob for candidate_stability_features JSONL files.",
    )
    parser.add_argument(
        "--dynamic-jsonl", action="append", default=[],
        help="candidate_stability_features_new_answers JSONL (repeat for multiple).",
    )
    parser.add_argument(
        "--dynamic-glob", default=None,
        help="Glob for dynamic-answer feature JSONL files.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=10000,
        help="Number of iterations for permutation test (default 10000).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples (default 10000).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default 0.05).",
    )
    parser.add_argument(
        "--min-combos", type=int, default=5,
        help="Minimum paired combos required to test a pair (default 5).",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Path to save per-pair CSV with all test results.",
    )
    parser.add_argument(
        "--out-summary-csv", default=None,
        help="Path to save per-record summary CSV.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print aggregate summary, not per-record details.",
    )
    args = parser.parse_args(argv)

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
        print("[error] No rows loaded. Provide --jsonl/--glob or --dynamic-jsonl/--dynamic-glob.")
        return 1

    has_arrays = any(
        isinstance(row.get("after_logprob_values"), list) and len(row["after_logprob_values"]) > 0
        for row in all_rows
    )
    if not has_arrays:
        print(
            "[error] The JSONL files don't contain 'after_logprob_values' arrays.\n"
            "        You need to re-run probability_on_top_beams.py for at least one record\n"
            "        (the updated version now saves the raw per-combo arrays)."
        )
        return 1

    grouped = _group_by_record(all_rows)
    print(f"[info] Found {len(grouped)} records with candidates.")

    all_record_results: List[Dict[str, Any]] = []
    for record_id in sorted(grouped.keys()):
        candidates = grouped[record_id]
        result = _analyze_one_record(
            record_id,
            candidates,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            min_combos=args.min_combos,
        )
        if result is None:
            continue
        all_record_results.append(result)
        if not args.quiet:
            _print_record_result(result, args.alpha)

    if not all_record_results:
        print("[warn] No records had both GT and wrong candidates with enough combos.")
        return 1

    _print_aggregate_summary(all_record_results, args.alpha)

    if args.out_csv:
        _write_csv(all_record_results, Path(args.out_csv), args.alpha)

    if args.out_summary_csv:
        _write_record_summary_csv(all_record_results, Path(args.out_summary_csv))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
