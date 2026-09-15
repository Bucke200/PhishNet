"""Diagnostic to evaluate the path/depth confound in PhishNet shape-only leakage audit.

This is a diagnostic task only. It does NOT modify production models,
frozen evaluation artifacts, collectors, or official split definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_splits  # noqa: E402

FEATURE_NAMES = [
    "url_len",
    "netloc_len",
    "path_len",
    "path_depth",
    "query_len",
    "query_equals_count",
    "is_https",
    "has_port",
]

# Column of netloc_len inside build_splits.shape_features output (the
# existing feature implementation; do not duplicate its logic).
NETLOC_LEN_INDEX = FEATURE_NAMES.index("netloc_len")


def netloc_len_values(urls: pd.Series) -> np.ndarray:
    """netloc_len for each URL via ``build_splits.shape_features``.

    No rows are discarded: ``shape_features`` applies ``urlparse`` to the
    stored (already normalised) URL exactly as the leakage audit does, so
    malformed URLs yield their parsed length rather than being silently
    dropped.
    """
    frame = pd.DataFrame({"url": list(urls)})
    return np.asarray(
        build_splits.shape_features(frame)[:, NETLOC_LEN_INDEX], dtype=float
    )


def describe_distribution(values: np.ndarray) -> dict[str, float]:
    """Count + location/spread/percentile summary of a 1-D value array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean": nan,
            "median": nan,
            "std": nan,
            "min": nan,
            "max": nan,
            "p10": nan,
            "p25": nan,
            "p75": nan,
            "p90": nan,
            "p95": nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def get_depth(u: str) -> int:
    return len([s for s in urlparse(u).path.split("/") if s])


def get_depth_bin(d: int, max_bin: int = 5) -> int:
    return min(d, max_bin)


def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def domain_bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    s: np.ndarray,
    domains: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(domains)
    index_of = {g: np.flatnonzero(domains == g) for g in uniq}
    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            v = fn(y[idx], s[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=float)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def run_diagnostic(
    eval_dir: Path,
    seed: int = 0,
    n_boot: int = 1000,
    output_json: Path | None = None,
) -> dict[str, Any]:
    train = pd.read_csv(eval_dir / "train.csv")
    test = pd.read_csv(eval_dir / "test.csv")

    for df in (train, test):
        df["path_depth"] = df["url"].map(get_depth)
        df["depth_bin"] = df["path_depth"].map(lambda d: get_depth_bin(d, max_bin=5))

    # 1. Original audit model and results
    Xtr_orig = build_splits.shape_features(train)
    Xte_orig = build_splits.shape_features(test)
    sc_orig = StandardScaler().fit(Xtr_orig)
    clf_orig = LogisticRegression(max_iter=2000).fit(
        sc_orig.transform(Xtr_orig), train.label
    )
    s_orig = clf_orig.predict_proba(sc_orig.transform(Xte_orig))[:, 1]

    orig_roc = float(roc_auc_score(test.label, s_orig))
    orig_pr = float(average_precision_score(test.label, s_orig))
    orig_ci = domain_bootstrap_ci(
        safe_roc_auc,
        test.label.to_numpy(),
        s_orig,
        test.registrable_domain.to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    # Pre-matching distributions. benign_te is the ORIGINAL benign URL
    # population: every benign row of the committed test set, measured here
    # before the stratified-subsampling matching below selects a subset.
    phish_te = test[test.label == 1]
    benign_te = test[test.label == 0]
    pre_match_netloc = describe_distribution(netloc_len_values(benign_te["url"]))

    pre_dist_phish = {
        int(k): int(v) for k, v in phish_te["depth_bin"].value_counts().items()
    }
    pre_dist_benign = {
        int(k): int(v) for k, v in benign_te["depth_bin"].value_counts().items()
    }
    pre_prop_phish = {
        int(k): float(v)
        for k, v in phish_te["depth_bin"].value_counts(normalize=True).items()
    }
    pre_prop_benign = {
        int(k): float(v)
        for k, v in benign_te["depth_bin"].value_counts(normalize=True).items()
    }

    # 2. Matching: Stratified Subsampling on path_depth
    # Match benign test distribution to phish test distribution
    bin_props_phish = phish_te["depth_bin"].value_counts(normalize=True).sort_index()

    # Bottleneck is bin 0:
    n_benign_bin0 = len(benign_te[benign_te.depth_bin == 0])
    total_target = int(np.floor(n_benign_bin0 / bin_props_phish[0]))

    sampled_benign = []
    for b in bin_props_phish.index:
        n_needed = int(round(total_target * bin_props_phish[b]))
        subset = benign_te[benign_te.depth_bin == b]
        sampled = subset.sample(n=min(n_needed, len(subset)), random_state=seed)
        sampled_benign.append(sampled)
    matched_benign_te = pd.concat(sampled_benign)
    matched_test = pd.concat([phish_te, matched_benign_te]).reset_index(drop=True)
    matched_netloc = describe_distribution(netloc_len_values(matched_benign_te["url"]))

    post_dist_benign = {
        int(k): int(v) for k, v in matched_benign_te["depth_bin"].value_counts().items()
    }
    post_prop_benign = {
        int(k): float(v)
        for k, v in matched_benign_te["depth_bin"].value_counts(normalize=True).items()
    }

    # 3. Matched results: original classifier evaluated on matched test
    Xte_matched = build_splits.shape_features(matched_test)
    s_matched_orig_clf = clf_orig.predict_proba(sc_orig.transform(Xte_matched))[:, 1]
    matched_roc_orig_clf = float(roc_auc_score(matched_test.label, s_matched_orig_clf))
    matched_pr_orig_clf = float(
        average_precision_score(matched_test.label, s_matched_orig_clf)
    )
    matched_ci_orig_clf = domain_bootstrap_ci(
        safe_roc_auc,
        matched_test.label.to_numpy(),
        s_matched_orig_clf,
        matched_test.registrable_domain.to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    # Importance weighting on full test set (non-parametric comparison)
    weights = np.ones(len(test))
    for b in bin_props_phish.index:
        w = bin_props_phish[b] / pre_prop_benign[b]
        mask = (test.label == 0) & (test.depth_bin == b)
        weights[mask] = w
    weighted_roc = float(roc_auc_score(test.label, s_orig, sample_weight=weights))

    # 4. Matched train and test: classifier retrained on matched train
    phish_tr = train[train.label == 1]
    benign_tr = train[train.label == 0]
    bin_props_phish_tr = phish_tr["depth_bin"].value_counts(normalize=True).sort_index()
    total_target_tr = int(
        len(benign_tr[benign_tr.depth_bin == 0]) / bin_props_phish_tr[0]
    )
    sampled_benign_tr = []
    for b in bin_props_phish_tr.index:
        n_needed = int(round(total_target_tr * bin_props_phish_tr[b]))
        subset = benign_tr[benign_tr.depth_bin == b]
        sampled = subset.sample(n=min(n_needed, len(subset)), random_state=seed)
        sampled_benign_tr.append(sampled)
    matched_train = pd.concat([phish_tr, pd.concat(sampled_benign_tr)]).reset_index(
        drop=True
    )

    Xtr_matched = build_splits.shape_features(matched_train)
    sc_matched = StandardScaler().fit(Xtr_matched)
    clf_matched = LogisticRegression(max_iter=2000).fit(
        sc_matched.transform(Xtr_matched), matched_train.label
    )
    s_matched_retrained = clf_matched.predict_proba(sc_matched.transform(Xte_matched))[
        :, 1
    ]
    matched_roc_retrained = float(
        roc_auc_score(matched_test.label, s_matched_retrained)
    )
    matched_pr_retrained = float(
        average_precision_score(matched_test.label, s_matched_retrained)
    )
    matched_ci_retrained = domain_bootstrap_ci(
        safe_roc_auc,
        matched_test.label.to_numpy(),
        s_matched_retrained,
        matched_test.registrable_domain.to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    # Single-feature ROC-AUC on original vs matched test sets
    single_features_orig: dict[str, float] = {}
    single_features_matched: dict[str, float] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        r_orig = roc_auc_score(test.label, Xte_orig[:, idx])
        r_orig_inv = roc_auc_score(test.label, -Xte_orig[:, idx])
        single_features_orig[name] = float(max(r_orig, r_orig_inv))

        r_m = roc_auc_score(matched_test.label, Xte_matched[:, idx])
        r_m_inv = roc_auc_score(matched_test.label, -Xte_matched[:, idx])
        single_features_matched[name] = float(max(r_m, r_m_inv))

    results = {
        "dataset_dir": str(eval_dir),
        "seed": seed,
        "n_boot": n_boot,
        "original": {
            "n_test": len(test),
            "n_phish": len(phish_te),
            "n_benign": len(benign_te),
            "base_rate": float(test.label.mean()),
            "roc_auc": orig_roc,
            "pr_auc": orig_pr,
            "roc_auc_95ci_domain": orig_ci,
            "mean_path_depth_phish": float(phish_te["path_depth"].mean()),
            "mean_path_depth_benign": float(benign_te["path_depth"].mean()),
            "median_path_depth_phish": float(phish_te["path_depth"].median()),
            "median_path_depth_benign": float(benign_te["path_depth"].median()),
            "depth_distribution_phish": pre_dist_phish,
            "depth_distribution_benign": pre_dist_benign,
            "depth_proportions_phish": pre_prop_phish,
            "depth_proportions_benign": pre_prop_benign,
        },
        "matching": {
            "method": "exact_stratified_subsampling",
            "bottleneck_bin": 0,
            "n_benign_retained": len(matched_benign_te),
            "n_benign_discarded": len(benign_te) - len(matched_benign_te),
            "n_phish_retained": len(phish_te),
            "n_test_matched": len(matched_test),
            "base_rate_matched": float(matched_test.label.mean()),
            "depth_distribution_phish": pre_dist_phish,
            "depth_distribution_benign": post_dist_benign,
            "depth_proportions_phish": pre_prop_phish,
            "depth_proportions_benign": post_prop_benign,
            "mean_path_depth_phish": float(phish_te["path_depth"].mean()),
            "mean_path_depth_benign_matched": float(
                matched_benign_te["path_depth"].mean()
            ),
            "median_path_depth_phish": float(phish_te["path_depth"].median()),
            "median_path_depth_benign_matched": float(
                matched_benign_te["path_depth"].median()
            ),
        },
        "netloc_len": {
            "definition": (
                "len(urlparse(url).netloc) via build_splits.shape_features "
                "column 1 (same implementation as the leakage audit); "
                "pre_match_benign covers every benign test URL before "
                "stratified-subsampling matching, matched_benign covers the "
                "retained subset only"
            ),
            "pre_match_benign": pre_match_netloc,
            "matched_benign": matched_netloc,
            "delta_matched_minus_pre": {
                "mean": matched_netloc["mean"] - pre_match_netloc["mean"],
                "median": matched_netloc["median"] - pre_match_netloc["median"],
            },
        },
        "matched_results": {
            "original_clf_on_matched_test": {
                "roc_auc": matched_roc_orig_clf,
                "pr_auc": matched_pr_orig_clf,
                "roc_auc_95ci_domain": matched_ci_orig_clf,
                "roc_auc_delta_abs": matched_roc_orig_clf - orig_roc,
                "roc_auc_delta_rel": (matched_roc_orig_clf - orig_roc) / orig_roc,
            },
            "importance_weighted_on_full_test": {
                "roc_auc": weighted_roc,
                "roc_auc_delta_abs": weighted_roc - orig_roc,
            },
            "retrained_clf_on_matched_data": {
                "roc_auc": matched_roc_retrained,
                "pr_auc": matched_pr_retrained,
                "roc_auc_95ci_domain": matched_ci_retrained,
                "roc_auc_delta_abs": matched_roc_retrained - orig_roc,
                "roc_auc_delta_rel": (matched_roc_retrained - orig_roc) / orig_roc,
            },
        },
        "single_feature_roc_auc": {
            "original": single_features_orig,
            "matched": single_features_matched,
        },
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, indent=2), encoding="utf-8", newline="\r\n"
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Path/depth confound diagnostic for PhishNet"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/splits-eval"),
        help="Path to evaluation splits dir (default: data/splits-eval)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scratch/path_depth_diagnostic_report.json"),
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    print(f"Running path/depth diagnostic on {args.eval_dir} (seed={args.seed})...")
    res = run_diagnostic(
        eval_dir=args.eval_dir,
        seed=args.seed,
        n_boot=args.n_boot,
        output_json=args.out,
    )

    orig = res["original"]
    matched = res["matching"]
    m_res = res["matched_results"]["original_clf_on_matched_test"]
    retrained = res["matched_results"]["retrained_clf_on_matched_data"]
    weighted = res["matched_results"]["importance_weighted_on_full_test"]

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(
        f"Original Test Set: N={orig['n_test']} "
        f"(Phish: {orig['n_phish']}, Benign: {orig['n_benign']})"
    )
    ci_o = orig["roc_auc_95ci_domain"]
    print(
        f"Original ROC-AUC:  {orig['roc_auc']:.4f}  "
        f"95% CI: [{ci_o[0]:.4f}, {ci_o[1]:.4f}]"
    )
    print(f"Original PR-AUC:   {orig['pr_auc']:.4f}")
    print(
        f"Mean Path Depth:   Phish={orig['mean_path_depth_phish']:.4f}, "
        f"Benign={orig['mean_path_depth_benign']:.4f}"
    )

    print("\nMatching (Stratified Subsampling):")
    print(
        f"Matched Test Set:  N={matched['n_test_matched']} "
        f"(Phish: {matched['n_phish_retained']}, "
        f"Benign: {matched['n_benign_retained']})"
    )
    print(
        f"Mean Path Depth:   Phish={matched['mean_path_depth_phish']:.4f}, "
        f"Benign={matched['mean_path_depth_benign_matched']:.4f}"
    )

    nl = res["netloc_len"]
    print("\nnetloc_len (pre-match benign vs matched benign):")
    print(
        f"Pre-match benign:  n={nl['pre_match_benign']['n']} "
        f"mean={nl['pre_match_benign']['mean']:.2f} "
        f"median={nl['pre_match_benign']['median']:.2f}"
    )
    print(
        f"Matched benign:    n={nl['matched_benign']['n']} "
        f"mean={nl['matched_benign']['mean']:.2f} "
        f"median={nl['matched_benign']['median']:.2f}"
    )
    print(
        f"Delta (matched-pre): mean={nl['delta_matched_minus_pre']['mean']:+.2f} "
        f"median={nl['delta_matched_minus_pre']['median']:+.2f}"
    )

    print("\nMatched Results (Original Classifier on Matched Test):")
    ci_m = m_res["roc_auc_95ci_domain"]
    print(
        f"Matched ROC-AUC:   {m_res['roc_auc']:.4f}  "
        f"95% CI: [{ci_m[0]:.4f}, {ci_m[1]:.4f}]"
    )
    delta_str = f"{m_res['roc_auc_delta_abs']:+.4f} ({m_res['roc_auc_delta_rel']:+.2%})"
    print(f"Delta ROC-AUC:     {delta_str}")

    print("\nImportance Weighted Result (Full Test Set):")
    w_delta = f"{weighted['roc_auc_delta_abs']:+.4f}"
    print(f"Weighted ROC-AUC:  {weighted['roc_auc']:.4f} (Delta: {w_delta})")

    print("\nRetrained Classifier on Matched Training Data:")
    ci_r = retrained["roc_auc_95ci_domain"]
    print(
        f"Retrained ROC-AUC: {retrained['roc_auc']:.4f}  "
        f"95% CI: [{ci_r[0]:.4f}, {ci_r[1]:.4f}]"
    )

    print("\nSingle Feature ROC-AUC (Original vs Matched Test):")
    for k in FEATURE_NAMES:
        r_orig = res["single_feature_roc_auc"]["original"][k]
        r_mat = res["single_feature_roc_auc"]["matched"][k]
        print(f"  {k:20s}: Original={r_orig:.4f} -> Matched={r_mat:.4f}")

    if args.out:
        print(f"\nWrote diagnostic report to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
