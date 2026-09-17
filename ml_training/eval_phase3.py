"""Phase 3 fixed-threshold evaluation driver (Amendment E).

Thresholds are fixed on the calib band (never test sweeps) at 0.5% and
1% FPR, then applied to test with wider-interval verdicts. Rows (a)
lexical and (b) +age only; row (e) is the Tranco diagnostic (never
trained). Produces one JSON report for the Phase 3 report's ablation
table. Reuses eval.py primitives; the harness itself is untouched.

Usage:
  python ml_training/eval_phase3.py --split-dir data/splits-p3 \\
      --snapshot data/enrichment-p3-2026-09-17.jsonl --run-id run-1 \\
      --assets-a backend/ablation_lexical_assets \\
      --assets-b backend/ablation_age_assets \\
      --out reports/phase3-ablation.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Importable when run from the repo root (matches the pytest pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_cc_benign  # noqa: E402
import eval as E  # noqa: E402
import predictors  # noqa: E402
from phishnet.enrichment.features import (  # noqa: E402
    ENRICHED_COLUMNS,
    HOSTED_COLUMN,
    apply_miss,
    build_feature_table,
)
from phishnet.enrichment.key import cache_key  # noqa: E402
from phishnet.enrichment.stub import UnknownStubProvider  # noqa: E402

TARGETS = (0.005, 0.01)
PHASE2_MISS_PP = 0.001  # 0.60% achieved at a 0.5% target (README)


def score_band(
    assets: str, snapshot: str, run_id: str, band_csv: Path
) -> tuple[np.ndarray, np.ndarray, predictors.EnrichedGbm]:
    """Scores for one band in eval mode (first_seen from the band CSV)."""
    pred = predictors.EnrichedGbm(
        assets_dir=assets,
        snapshot=snapshot,
        run_id=run_id,
        first_seen_csv=str(band_csv),
    )
    urls = pd.read_csv(band_csv, usecols=["url"])["url"].astype(str).tolist()
    y = pd.read_csv(band_csv, usecols=["label"])["label"].to_numpy().astype(int)
    scores, _ = E.score_all(pred, urls, batch_size=512)
    return y, scores, pred


def fixed_thresholds(y_calib: np.ndarray, s_calib: np.ndarray) -> dict[float, float]:
    """Lowest calib thresholds within budget (never test sweeps)."""
    return {t: E.threshold_at_fpr(y_calib, s_calib, t) for t in TARGETS}


def domain_groups(band_csv: Path) -> np.ndarray:
    """Domain-bootstrap clusters (registrable domains)."""
    return (
        pd.read_csv(band_csv, usecols=["registrable_domain"])["registrable_domain"]
        .astype(str)
        .to_numpy()
    )


def paired_recall_lift(
    y: np.ndarray,
    s_base: np.ndarray,
    s_cand: np.ndarray,
    thr_base: float,
    thr_cand: float,
    n_boot: int,
    seed: int,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Paired CIs on recall and PR-AUC lift (cand − base).

    Each row is judged at its OWN calib-fixed threshold (rows score on
    different scales, so one shared number is not one operating point):
    diff = recall_cand@thr_cand − recall_base@thr_base on the same
    resampled rows. PR-AUC lift is threshold-free as before.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            rb = E.rates_at(y[idx], s_cand[idx], thr_cand)["recall"]
            ra = E.rates_at(y[idx], s_base[idx], thr_base)["recall"]
            diffs.append(rb - ra)
        except Exception:
            continue
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    recall_ci = (
        [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        if arr.size
        else [float("nan"), float("nan")]
    )
    return {
        "recall_lift_ci": recall_ci,
        "pr_auc_lift_ci": list(
            E.paired_bootstrap_ci(E.pr_auc, y, s_cand, s_base, n_boot, seed, groups)
        ),
    }


def transfer_verdict(
    calib_fpr: float,
    test_fpr: float,
    diff_ci: tuple[float, float],
    bar_pp: float | None = PHASE2_MISS_PP,
) -> dict[str, Any]:
    """Transfer verdict by the wider-interval rule (unit-tested).

    ``bar_pp`` is the drift budget (Phase 2's 0.10pp miss at 0.5%);
    ``None`` means no comparator exists (1% target): the drift reports
    with its interval and the verdict reads indistinguishable. A point
    estimate inside the bar whose interval straddles it is likewise
    indistinguishable — the rule proposed with point estimates alone
    could not conclude, and is superseded here (review correction).
    """
    drift = test_fpr - calib_fpr
    lo, hi = diff_ci
    if bar_pp is None:
        verdict, reason = "indistinguishable", "no comparator at this target"
    elif not (np.isfinite(lo) and np.isfinite(hi)):
        verdict, reason = "indistinguishable", "interval unmeasurable"
    elif hi <= bar_pp and -lo <= bar_pp:
        verdict, reason = "fixed", "drift conclusively below Phase 2's miss"
    elif lo > bar_pp or -hi > bar_pp:
        verdict, reason = "not-fixed", "drift conclusively above Phase 2's miss"
    else:
        verdict, reason = "indistinguishable", "interval straddles the bar"
    return {
        "calib_fpr": calib_fpr,
        "test_fpr": test_fpr,
        "drift_pp": drift,
        "drift_ci": [lo, hi],
        "bar_pp": bar_pp,
        "verdict": verdict,
        "reason": reason,
    }


def drift_ci(
    y_calib: np.ndarray,
    s_calib: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    thr: float,
    n_boot: int,
    seed: int,
    groups_calib: np.ndarray,
    groups_test: np.ndarray,
) -> tuple[float, float]:
    """Interval on test−calib FPR drift (paired-by-replicate-index).

    Independent domain bootstraps per side, same replicate count, seeded
    streams: d_i = test_i − calib_i. Deterministic for fixed seeds.
    """
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    uc = np.unique(groups_calib[y_calib == 0])
    ut = np.unique(groups_test[y_test == 0])
    idx_c = {g: np.flatnonzero(groups_calib == g) for g in uc}
    idx_t = {g: np.flatnonzero(groups_test == g) for g in ut}
    for _ in range(n_boot):
        pick_c = rng.choice(uc, uc.size, replace=True)
        pick_t = rng.choice(ut, ut.size, replace=True)
        ic = np.concatenate([idx_c[g] for g in pick_c])
        it = np.concatenate([idx_t[g] for g in pick_t])
        fc = float((s_calib[ic][y_calib[ic] == 0] >= thr).mean())
        ft = float((s_test[it][y_test[it] == 0] >= thr).mean())
        if np.isfinite(fc) and np.isfinite(ft):
            diffs.append(ft - fc)
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def benign_stratum_table(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    thr_a: float,
    thr_b: float,
    snapshot: str,
    run_id: str,
    d1_corpus: Path,
) -> list[dict[str, Any]]:
    """Benign FPR and age-known rate per D1 popularity stratum.

    Test benign URLs map back to the pinned D1 corpus for
    ``popularity_stratum`` (s1 head → s6 tail). Unmapped rows group
    under "unmapped" with coverage reported — never silently dropped.
    """
    import json as _json

    stratum_of: dict[str, str] = {}
    with open(d1_corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = _json.loads(line)
            except ValueError:
                continue
            if r.get("label") == 0 and r.get("url"):
                stratum_of[str(r["url"])] = str(r.get("popularity_stratum", "unmapped"))
    from phishnet.enrichment.join import derive_row  # noqa: E402
    from phishnet.enrichment.key import cache_key as _ck  # noqa: E402
    from phishnet.enrichment.store import load_pinned_run  # noqa: E402

    table = load_pinned_run(Path(snapshot), run_id)
    rows = []
    urls = test_df["url"].astype(str).to_numpy()
    stamps = test_df["first_seen"].astype(str).to_numpy()
    strata = sorted({stratum_of.get(u, "unmapped") for u in urls})
    for s in strata:
        m = (
            test_df["url"].astype(str).map(lambda u: stratum_of.get(u, "unmapped")) == s
        ).to_numpy()
        m = m & (y_test == 0)
        n = int(m.sum())
        known = 0
        if n:
            # Each row's own first_seen: a late stamp would mask
            # re-registrations as known.
            for u, stamp in zip(urls[m], stamps[m], strict=True):
                key, hosted = _ck(str(u))
                rec = table.get(key) or {"cache_key": key}
                if derive_row(str(u), rec, str(stamp), hosted).get("age_known"):
                    known += 1
        rows.append(
            {
                "stratum": s,
                "n_benign": n,
                "age_known_rate": (known / n) if n else float("nan"),
                "fpr_row_a": float((s_a[m] >= thr_a).mean()) if n else float("nan"),
                "fpr_row_b": float((s_b[m] >= thr_b).mean()) if n else float("nan"),
            }
        )
    return rows


def stratified_shape_audit(split_dir: Path) -> dict[str, Any]:
    """Shape-only audit on the calib band, mixture + main stratum.

    Uses the committed ``build_splits.leakage_audit`` (same model the
    population audit ran): full train→calib first, then non-hosted rows
    only. If the hosted mixture explains the suspicious reading, the
    stratified number shows it; otherwise the thresholds were fixed on
    a confounded band.
    """
    from urllib.parse import urlparse  # noqa: E402

    import build_splits  # noqa: E402

    cols = ["url", "label", "is_hosted_tenant"]
    train = pd.read_csv(split_dir / "train.csv", usecols=cols)
    calib = pd.read_csv(split_dir / "calib.csv", usecols=cols)
    # Same path_depth rule the population audit used (build_splits._group):
    # split CSVs don't persist it.
    for frame in (train, calib):
        frame["path_depth"] = frame["url"].map(
            lambda u: len([s for s in urlparse(str(u)).path.split("/") if s])
        )

    def nonhosted(frame: pd.DataFrame) -> pd.DataFrame:
        flag = (
            frame["is_hosted_tenant"]
            .map({True: True, False: False, "True": True, "False": False})
            .fillna(False)
            .astype(bool)
        )
        return frame[~flag].reset_index(drop=True)

    mixture = build_splits.leakage_audit(train, calib)
    main = build_splits.leakage_audit(nonhosted(train), nonhosted(calib))
    return {
        "mixture_roc_auc": mixture["shape_only_roc_auc"],
        "mixture_verdict": mixture["verdict"],
        "main_stratum_roc_auc": main["shape_only_roc_auc"],
        "main_stratum_verdict": main["verdict"],
    }


def cold_start_curve(
    assets: str,
    snapshot: str,
    run_id: str,
    test_csv: Path,
    thr: float,
    seed: int,
) -> dict[str, Any]:
    """Recall/FPR at 0/50/100% missing age (key-grouped forced miss).

    Research number: how much the feature depends on lookups succeeding.
    Reported for all rows and the fresh slice.
    """
    frame = pd.read_csv(test_csv)
    train_cfg = json.loads(
        (Path(assets) / "train_config.json").read_text(encoding="utf-8")
    )
    canonicalize = bool(train_cfg.get("canonicalize_scheme", False))
    cols_path = Path(assets) / "feature_columns.pkl"
    columns: list[str] = pickle.loads(cols_path.read_bytes())
    frozen = [c for c in columns if c not in ENRICHED_COLUMNS and c != HOSTED_COLUMN]
    from ml_training.train_ablation import split_rows

    X_full, _, _ = build_feature_table(
        split_rows(frame, "test.csv"),
        Path(snapshot),
        {"rule": "pinned-run", "run_id": run_id},
        frozen,
        canonicalize=canonicalize,
    )
    keys = [cache_key(u)[0] for u in frame["url"].astype(str)]
    with open(Path(assets) / "gbm_model.pkl", "rb") as f:
        model: Any = pickle.load(f)
    y = frame["label"].to_numpy().astype(int)
    fresh = (frame["survival_stratum"].astype(str) == "fresh").to_numpy()
    out: dict[str, Any] = {}
    for miss in (0.0, 0.5, 1.0):
        # Miss first on the full joined table (apply_miss covers every
        # enriched column), then select the row's columns.
        Xm = apply_miss(X_full, keys, miss, seed)[columns].to_numpy(dtype=float)
        s = np.asarray(model.predict_proba(Xm)[:, 1], dtype=float)
        rep = E.rates_at(y, s, thr)
        fpr = float((s[y == 0] >= thr).mean())
        cell: dict[str, Any] = {"recall": rep["recall"], "fpr": fpr, "n": len(y)}
        if fresh.sum():
            sf, yf = s[fresh], y[fresh]
            cell["fresh"] = {
                "recall": E.rates_at(yf, sf, thr)["recall"],
                "n": int(fresh.sum()),
            }
        out[f"miss_{miss}"] = cell
    return out


def serving_shape_latency(assets: str, test_csv: Path, seed: int) -> dict[str, float]:
    """Tier-1 serving path: stub lookup + lexical/hosted features + model.

    The eval-mode join scorer (serving_stub_latency) routes a serving
    question through point-in-time join work production never does per
    request. This measures what tier-1 actually serves for the headline
    champion: URL-derived features plus the stub's cache-miss cost,
    timed per URL (a browser extension blocks on one URL, not a batch).
    """
    from phishnet.enrichment.features import hosted_flag  # noqa: E402
    from phishnet.features.extraction import featurise_frame  # noqa: E402

    cols: list[str] = pickle.loads((Path(assets) / "feature_columns.pkl").read_bytes())
    frozen = [c for c in cols if c not in ENRICHED_COLUMNS and c != HOSTED_COLUMN]
    cfg = json.loads((Path(assets) / "train_config.json").read_text(encoding="utf-8"))
    canonicalize = bool(cfg.get("canonicalize_scheme", False))
    with open(Path(assets) / "gbm_model.pkl", "rb") as f:
        model: Any = pickle.load(f)
    stub = UnknownStubProvider()
    urls = pd.read_csv(test_csv, usecols=["url"])["url"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(urls, min(300, len(urls)), replace=False))

    def serve_one(url: str) -> float:
        stub.lookup_many([url])
        lex = featurise_frame([url], frozen, canonicalize=canonicalize)
        lex[HOSTED_COLUMN] = hosted_flag([url])
        return float(model.predict_proba(lex[cols].to_numpy(dtype=float))[0, 1])

    for u in sample[:20]:
        serve_one(u)
    times = []
    for u in sample:
        t0 = time.perf_counter()
        serve_one(u)
        times.append(time.perf_counter() - t0)
    a = np.asarray(times) * 1000.0
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "n": len(a),
    }


def serving_stub_latency(
    assets: str, snapshot: str, run_id: str, test_csv: Path, seed: int
) -> dict[str, float]:
    """Eval-mode join scorer latency with the stub provider (reference).

    Routes through the point-in-time join; production serving never does
    that work per request (see serving_shape_latency for the tier-1
    number). Kept as the conservative upper bound. Nearly free: a few
    hundred single-URL calls.
    """
    pred = predictors.EnrichedGbm(
        assets_dir=assets,
        snapshot=snapshot,
        run_id=run_id,
        first_seen_csv=str(test_csv),
    )
    stub = UnknownStubProvider()
    urls = pd.read_csv(test_csv, usecols=["url"])["url"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(urls, min(300, len(urls)), replace=False))
    for u in sample[:20]:
        stub.lookup_many([u])
        pred.score([u])
    times = []
    for u in sample:
        t0 = time.perf_counter()
        stub.lookup_many([u])
        pred.score([u])
        times.append(time.perf_counter() - t0)
    a = np.asarray(times) * 1000.0
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "n": len(a),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--assets-a", required=True, help="row (a) lexical assets")
    p.add_argument("--assets-b", required=True, help="row (b) +age assets")
    p.add_argument(
        "--d1-corpus",
        type=Path,
        default=Path("data/raw/benign-cc-CC-MAIN-2026-34-d1.jsonl"),
        help="pinned D1 corpus (benign popularity_stratum source)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    calib_csv = a.split_dir / "calib.csv"
    test_csv = a.split_dir / "test.csv"
    train_csv = a.split_dir / "train.csv"
    groups_test = domain_groups(test_csv)
    groups_calib = domain_groups(calib_csv)

    scored: dict[str, dict[str, Any]] = {}
    for row, assets in (("a", a.assets_a), ("b", a.assets_b)):
        y_calib, s_calib, _ = score_band(assets, a.snapshot, a.run_id, calib_csv)
        y_test, s_test, _ = score_band(assets, a.snapshot, a.run_id, test_csv)
        scored[row] = {
            "y_calib": y_calib,
            "s_calib": s_calib,
            "y_test": y_test,
            "s_test": s_test,
        }

    # Each row gets thresholds fixed on its OWN calib scores: the two
    # models score on different scales, so one shared number is not one
    # operating point (review correction — shared thresholds overstated
    # row (a)'s FPR).
    thresholds = {
        row: fixed_thresholds(scored[row]["y_calib"], scored[row]["s_calib"])
        for row in ("a", "b")
    }
    for row in ("a", "b"):
        print(
            f"fixed thresholds (row {row}, calib): "
            + ", ".join(f"{t}: {thr:.6f}" for t, thr in thresholds[row].items()),
            flush=True,
        )

    report: dict[str, Any] = {
        "thresholds": {
            row: {str(t): thr for t, thr in thrs.items()}
            for row, thrs in thresholds.items()
        },
        "rows": {},
    }
    for row in ("a", "b"):
        y_test, s_test = scored[row]["y_test"], scored[row]["s_test"]
        cell: dict[str, Any] = {}
        for target, thr in thresholds[row].items():
            fp_n = int((s_test[y_test == 0] >= thr).sum())
            n_neg = int((y_test == 0).sum())
            cell[str(target)] = E.fpr_interval_report(
                fp_n,
                n_neg,
                y_test,
                s_test,
                thr,
                a.bootstrap,
                a.seed,
                groups_test,
                target,
            )
            cell[str(target)]["recall"] = E.rates_at(y_test, s_test, thr)["recall"]
            cell[str(target)]["pr_auc"] = E.pr_auc(y_test, s_test)
        report["rows"][row] = cell

    # Paired lift (b − a): each row at its own 0.5% threshold + PR-AUC.
    y_test = scored["b"]["y_test"]
    report["paired_lift_b_minus_a"] = paired_recall_lift(
        y_test,
        scored["a"]["s_test"],
        scored["b"]["s_test"],
        thresholds["a"][0.005],
        thresholds["b"][0.005],
        a.bootstrap,
        a.seed,
        groups_test,
    )

    # Slices at row (b)'s own 0.5% threshold, incl. per-URL-type.
    test_df = pd.read_csv(test_csv)
    test_df["url_type"] = test_df["url"].astype(str).map(build_cc_benign.url_type)
    report["slices_row_b_at_0_5pct"] = E.slice_report(
        test_df,
        y_test,
        scored["b"]["s_test"],
        thresholds["b"][0.005],
        E.EvalConfig(seed=a.seed),
    )

    # Per-URL-type breakdown (hosted benign has no FPR claim without
    # counts: report type mix with row counts beside every rate).
    thr05 = thresholds["b"][0.005]
    sb = scored["b"]["s_test"]
    hosted = (
        test_df["is_hosted_tenant"]
        .map({True: True, False: False, "True": True, "False": False})
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    url_types = test_df["url_type"].astype(str).to_numpy()
    ut_rows = []
    for t in sorted(set(url_types)):
        m = url_types == t
        hb = m & hosted & (y_test == 0)
        hp = m & hosted & (y_test == 1)
        ut_rows.append(
            {
                "url_type": t,
                "n": int(m.sum()),
                "n_hosted_benign": int(hb.sum()),
                "n_hosted_phish": int(hp.sum()),
                "hosted_benign_fpr": float((sb[hb] >= thr05).mean())
                if hb.sum()
                else float("nan"),
                "hosted_phish_recall": float((sb[hp] >= thr05).mean())
                if hp.sum()
                else float("nan"),
            }
        )
    report["url_type_hosted"] = ut_rows

    # Platform-prior baseline on the hosted slice, same metrics both
    # sides (review correction: PR-AUC beside recall is incomparable).
    prior = predictors.PlatformPriorBaseline(train_csv=str(train_csv))
    s_prior = np.asarray(prior.score(test_df["url"].astype(str).tolist()), dtype=float)
    s_b = scored["b"]["s_test"]
    thr_b05 = thresholds["b"][0.005]
    report["hosted_slice"] = {
        "n_hosted": int(hosted.sum()),
        "model_recall_at_0_5pct": E.rates_at(y_test[hosted], s_b[hosted], thr_b05)[
            "recall"
        ]
        if hosted.sum()
        else float("nan"),
        "model_pr_auc": E.pr_auc(y_test[hosted], s_b[hosted])
        if len(np.unique(y_test[hosted])) > 1
        else float("nan"),
        "prior_recall_at_row_b_threshold": float((s_prior[hosted] >= thr_b05).mean())
        if hosted.sum()
        else float("nan"),
        "prior_pr_auc": E.pr_auc(y_test[hosted], s_prior[hosted])
        if len(np.unique(y_test[hosted])) > 1
        else float("nan"),
    }

    # Benign-stratum age check (review correction): benign came from
    # Tranco, so benign domains are old by construction and row (e)
    # (hostname shape) does not cover age. Per D1 popularity stratum:
    # FPR per row plus the age-known rate — if the (b − a) advantage
    # holds in the lowest-ranked strata (s5, s6), that is real evidence.
    report["benign_stratum"] = benign_stratum_table(
        test_df,
        y_test,
        scored["a"]["s_test"],
        scored["b"]["s_test"],
        thresholds["a"][0.005],
        thresholds["b"][0.005],
        a.snapshot,
        a.run_id,
        a.d1_corpus,
    )

    # Stratified shape audit of the calib band (review correction): the
    # population audit read suspicious (0.815) on the mixture the
    # thresholds were fixed on. Main-stratum-only shows whether the
    # hosted mixture explains it.
    report["stratified_shape_audit"] = stratified_shape_audit(a.split_dir)

    # Cold-start curve on row (b) at its own 0.5% threshold.
    report["cold_start_row_b_at_0_5pct"] = cold_start_curve(
        a.assets_b, a.snapshot, a.run_id, test_csv, thresholds["b"][0.005], a.seed
    )

    # Threshold transfer (criterion 11): on row (a), the eligible
    # headline row — not row (b) (review correction). Bar 0.10pp at
    # 0.5%; no Phase 2 comparator at 1%.
    transfer: dict[str, Any] = {}
    for target, thr in thresholds["a"].items():
        y_calib, s_calib = scored["a"]["y_calib"], scored["a"]["s_calib"]
        s_test_a = scored["a"]["s_test"]
        calib_fpr = float((s_calib[y_calib == 0] >= thr).mean())
        test_fpr = float((s_test_a[y_test == 0] >= thr).mean())
        ci = drift_ci(
            y_calib,
            s_calib,
            y_test,
            s_test_a,
            thr,
            a.bootstrap,
            a.seed,
            groups_calib,
            groups_test,
        )
        bar = PHASE2_MISS_PP if target == 0.005 else None
        transfer[str(target)] = transfer_verdict(calib_fpr, test_fpr, ci, bar)
    report["transfer"] = transfer

    # Tier-1 stub latency (criterion 12): the serving shape on the
    # headline champion (row a) is the criterion instrument; the
    # eval-mode join scorer is the conservative reference.
    report["serving_shape_latency_row_a"] = serving_shape_latency(
        a.assets_a, test_csv, a.seed
    )
    report["stub_latency"] = serving_stub_latency(
        a.assets_b, a.snapshot, a.run_id, test_csv, a.seed
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
