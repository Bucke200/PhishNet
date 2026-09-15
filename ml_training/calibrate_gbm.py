"""Isotonic-calibrate the CC GBM on a domain-disjoint held-out slice (Step 5).

Carve: registrable-domain hash partition of <train> ONLY (default
data/splits-cc/train.csv), reusing build_splits.neg_domain_is_test with a
FRESH seed (default phishnet-calib-v1, which refuses the train/test seed)
and --calib-fraction (default 0.2). Domains below the fraction form the
calibration set; the rest refit the LGBMClassifier (hyperparameters
imported from train_gbm.build_model, not duplicated) in NATIVE units —
no scaler (see train_gbm.py). test.csv is never read, let alone touched.
The refit base is also persisted standalone as refit_base.pkl: it is the
deployable champion lineage (honest fixed threshold), unlike the
full-train weights, which have no held-out slice left to tune on.

Why refit instead of calibrating the shipped GBM: the shipped model saw all
of train, so no train-carved slice is held out from it — calibrating it
would tune on its own training distribution. A random slice would
additionally leak campaign structure (same registrable domains on both
sides, one phishing kit deciding both fits). Domain-hash carving keeps
every domain wholly on one side; overlap is asserted zero.

Threshold knob: --target-fpr (default 0.005, the repo's operating currency)
selects a threshold on the CALIBRATION slice via eval.threshold_at_fpr —
never on test. The sidecar records target -> threshold + achieved calib
metrics, so the product threshold is a re-derivable function of a declared
target, not a constant.

Outputs (deployment artifacts, git-ignored):
  <assets-out>/calibrated_gbm.pkl
  <assets-out>/refit_base.pkl  (uncalibrated refit LGBM; the champion artifact)
  <assets-out>/feature_columns.pkl  (copy of the frozen vocabulary)
  <assets-out>/calibration-report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

# Importable when run from the repo root (matches the pytest pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

import build_splits
import eval as eval_harness
from ml_training.train_gbm import FROZEN_COLUMNS, build_model, featurise

CALIB_SEED_DEFAULT = "phishnet-calib-v1"
CALIB_FRACTION_DEFAULT = 0.2


def carve_fit_calib(
    df: pd.DataFrame, seed: str, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into (fit, calibration) by registrable-domain hash.

    Same mechanism as the benign train/test partition
    (sha256('<seed>:<domain>') -> [0, 1)), fresh seed, applied to every
    train domain of either class. Whole domains stay together; fit and
    calibration share zero domains. Refuses the train/test seed (that would
    correlate the carve with the population split) and degenerate fractions,
    and aborts rather than fitting isotonic on a single class.
    """
    if seed == build_splits.NEG_HASH_SEED_DEFAULT:
        raise ValueError(
            f"calibration seed must be fresh, got the train/test seed {seed!r}"
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"calibration fraction must be in (0, 1), got {fraction}")
    calib_domains = {
        d
        for d in df["registrable_domain"].unique()
        if build_splits.neg_domain_is_test(d, seed, fraction)
    }
    calib = df[df["registrable_domain"].isin(calib_domains)].reset_index(drop=True)
    fit = df[~df["registrable_domain"].isin(calib_domains)].reset_index(drop=True)
    if fit.empty or calib.empty:
        raise ValueError("carve left an empty partition; adjust --calib-fraction")
    if fit["label"].nunique() < 2 or calib["label"].nunique() < 2:
        raise ValueError("carve left a single-class partition; isotonic needs both")
    overlap = set(fit["registrable_domain"]) & set(calib["registrable_domain"])
    if overlap:
        raise ValueError(f"carve leaked {len(overlap)} domains across partitions")
    return fit, calib


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/splits-cc/train.csv"))
    p.add_argument("--assets-out", type=Path, default=Path("backend/gbm_iso_assets"))
    p.add_argument("--frozen-columns", type=Path, default=FROZEN_COLUMNS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-seed", default=CALIB_SEED_DEFAULT)
    p.add_argument("--calib-fraction", type=float, default=CALIB_FRACTION_DEFAULT)
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.005,
        help="product FPR budget; threshold is selected on the calibration "
        "slice for this target, never on test",
    )
    p.add_argument(
        "--method",
        choices=("isotonic", "sigmoid"),
        default="isotonic",
        help="calibrator capacity: isotonic is non-parametric (fits the "
        "slice closely, transfers poorly under drift); sigmoid (Platt) is "
        "two parameters and usually survives drift better",
    )
    a = p.parse_args(argv)
    if a.calib_seed == build_splits.NEG_HASH_SEED_DEFAULT:
        p.error("refusing the train/test seed as --calib-seed (fresh seed required)")
    if not 0.0 < a.calib_fraction < 1.0:
        p.error("--calib-fraction must be strictly between 0 and 1")
    if not 0.0 < a.target_fpr < 1.0:
        p.error("--target-fpr must be strictly between 0 and 1")

    columns: list[str] = pickle.loads(a.frozen_columns.read_bytes())
    print(f"frozen vocabulary: {len(columns)} columns from {a.frozen_columns}")

    # test.csv is never read here: the calibration slice comes from train only.
    train = pd.read_csv(a.train)
    fit, calib = carve_fit_calib(train, a.calib_seed, a.calib_fraction)
    print(
        f"carve (seed={a.calib_seed!r}, fraction={a.calib_fraction}): "
        f"fit {len(fit):,} rows / {fit['registrable_domain'].nunique():,} domains, "
        f"calib {len(calib):,} rows / {calib['registrable_domain'].nunique():,} domains"
    )
    fit = shuffle(fit, random_state=a.seed).reset_index(drop=True)

    print("featurising fit...")
    t0 = time.perf_counter()
    X_fit = featurise(fit["url"].astype(str).tolist(), columns)
    print(f"fit features {X_fit.shape} in {time.perf_counter() - t0:.0f}s")
    print("featurising calibration...")
    t0 = time.perf_counter()
    X_cal = featurise(calib["url"].astype(str).tolist(), columns)
    print(f"calib features {X_cal.shape} in {time.perf_counter() - t0:.0f}s")
    y_fit = fit["label"].to_numpy().astype(int)
    y_cal = calib["label"].to_numpy().astype(int)

    # Native units end to end (see train_gbm.py): no scaler anywhere, so
    # there is nothing whose fit scope could leak across the partition.
    Xf = X_fit.to_numpy(dtype=float)
    Xc = X_cal.to_numpy(dtype=float)

    base = build_model()
    print("training base GBM on fit partition...")
    t0 = time.perf_counter()
    base.fit(Xf, y_fit)
    print(f"trained in {time.perf_counter() - t0:.0f}s")

    s_base = np.asarray(base.predict_proba(Xc)[:, 1], dtype=float)
    pre = eval_harness.calibration(y_cal, s_base, 10)

    iso = CalibratedClassifierCV(estimator=base, method=a.method, cv="prefit")
    iso.fit(Xc, y_cal)
    print(f"fitted {a.method} calibrator (prefit) on {len(y_cal):,} rows")
    s_iso = np.asarray(iso.predict_proba(Xc)[:, 1], dtype=float)
    post = eval_harness.calibration(y_cal, s_iso, 10)

    threshold = eval_harness.threshold_at_fpr(y_cal, s_iso, a.target_fpr)
    achieved = eval_harness.rates_at(y_cal, s_iso, threshold)
    conf = eval_harness.confusion_at(y_cal, s_iso, threshold)
    print(
        f"threshold for FPR<={a.target_fpr:.2%} (selected on calibration): "
        f"{threshold:.6f} -> calib recall {achieved['recall']:.4f} "
        f"at FPR {achieved['fpr']:.4f} (TP {conf['tp']} FP {conf['fp']})"
    )
    print(
        f"calibration slice: Brier {pre['brier']:.4f} -> {post['brier']:.4f}, "
        f"ECE {pre['ece']:.4f} -> {post['ece']:.4f} (in-sample: the honest "
        f"test-set numbers come from eval.py, not from rows {a.method} saw)"
    )

    a.assets_out.mkdir(parents=True, exist_ok=True)
    with open(a.assets_out / "calibrated_gbm.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(a.assets_out / "refit_base.pkl", "wb") as f:
        pickle.dump(base, f)
    shutil.copyfile(a.frozen_columns, a.assets_out / "feature_columns.pkl")
    stale = a.assets_out / "scaler.pkl"
    if stale.exists():
        stale.unlink()
        print(f"removed stale {stale} (scaler dropped; see train_gbm.py)")
    report = {
        "train_csv": str(a.train),
        "test_csv_touched": False,
        "calib_seed": a.calib_seed,
        "calib_fraction": a.calib_fraction,
        "n_fit_rows": len(fit),
        "n_fit_domains": int(fit["registrable_domain"].nunique()),
        "n_calib_rows": len(calib),
        "n_calib_domains": int(calib["registrable_domain"].nunique()),
        "fit_calib_domain_overlap": 0,
        "base_estimator": {"type": "LGBMClassifier", **base.get_params()},
        "calibration_method": a.method,
        "scaler_fit_scope": "none (scaler dropped; native units)",
        "product_target": {"target_fpr": a.target_fpr},
        "threshold": threshold,
        "threshold_selected_on": "calibration slice (never test)",
        "threshold_calib_achieved": {**conf, **achieved},
        "calib_slice_metrics": {
            "pre_brier": pre["brier"],
            "pre_ece": pre["ece"],
            "pre_mce": pre["mce"],
            "post_brier": post["brier"],
            "post_ece": post["ece"],
            "post_mce": post["mce"],
            "note": "in-sample (calibrator fit on these rows); "
            "honest numbers come from eval.py on untouched test data",
        },
        "ece_interpretation": (
            "A continuous model reports a HIGHER ECE than the five-score "
            "hard vote (e.g. 0.0184 on cc-retrained-oldpop) because five "
            "discrete scores land in five bins with nothing to miscalibrate. "
            "That low number is a resolution artifact, not good calibration. "
            "The continuous ECE is a truer measurement; read a rise as "
            "honesty, not regression."
        ),
    }
    (a.assets_out / "calibration-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\nwrote {a.assets_out}/calibrated_gbm.pkl, refit_base.pkl, "
        f"feature_columns.pkl, calibration-report.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
