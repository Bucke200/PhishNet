"""Train a single LightGBM on a CC-trial split (Step 4 ablation).

Near-copy of ml_training/train_cc_split.py: the canonical
comprehensive_phishing_features extractor, the frozen 78-column
vocabulary (tld dropped), and the same training population (default
data/splits-cc). Two deliberate changes vs the ensemble script: the
estimator (one LGBMClassifier with class_weight="balanced" instead of
the hard-voting VotingClassifier) and NO StandardScaler — a no-op for
trees whose only effect was unreadable standardized units. The ablation
row re-ran after the removal to confirm the no-op empirically.

Outputs (deployment artifacts, git-ignored):
  <assets-out>/gbm_model.pkl
  <assets-out>/feature_columns.pkl  (copy of the frozen vocabulary)

Usage:
  python ml_training/train_gbm.py
  python ml_training/train_gbm.py --split-dir data/splits-cc \
      --assets-out backend/gbm_assets
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

# Importable when run from the repo root (matches the pytest pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishnet.features.extraction import (  # type: ignore[import-untyped]
    comprehensive_phishing_features,
)

FROZEN_COLUMNS = (
    Path("src/phishnet/urlset_ml_assets/feature_columns.pkl")
    if Path("src/phishnet/urlset_ml_assets/feature_columns.pkl").exists()
    else Path("backend/urlset_ml_assets/feature_columns.pkl")
)


def featurise(urls: list[str], columns: list[str]) -> pd.DataFrame:
    """Same extract -> reindex -> coerce pipeline as serving/eval."""
    feats = []
    for i, u in enumerate(urls):
        feats.append(comprehensive_phishing_features(u))
        if (i + 1) % 5000 == 0:
            print(f"  extracted {i + 1:,}/{len(urls):,}", flush=True)
    frame = pd.DataFrame(feats)
    if "tld" in frame.columns:
        frame = frame.drop(columns=["tld"])
    for col in columns:
        if col not in frame.columns:
            frame[col] = 0
    cleaned: pd.DataFrame = (
        frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    return cleaned


def build_model() -> LGBMClassifier:
    """Single estimator under test: one balanced LightGBM, deterministic."""
    return LGBMClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=Path("data/splits-cc"))
    p.add_argument("--assets-out", type=Path, default=Path("backend/gbm_assets"))
    p.add_argument("--frozen-columns", type=Path, default=FROZEN_COLUMNS)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    columns: list[str] = pickle.loads(a.frozen_columns.read_bytes())
    print(f"frozen vocabulary: {len(columns)} columns from {a.frozen_columns}")

    train = pd.read_csv(a.split_dir / "train.csv")
    test = pd.read_csv(a.split_dir / "test.csv")
    print(f"train {len(train):,} / test {len(test):,}")
    # Shuffle train (the split builder orders by population, not for SGD —
    # trees don't care, but a shuffled artifact trains identically anywhere).
    train = shuffle(train, random_state=a.seed).reset_index(drop=True)

    print("featurising train...")
    t0 = time.perf_counter()
    X_train = featurise(train["url"].astype(str).tolist(), columns)
    print(f"train features {X_train.shape} in {time.perf_counter() - t0:.0f}s")
    print("featurising test...")
    t0 = time.perf_counter()
    X_test = featurise(test["url"].astype(str).tolist(), columns)
    print(f"test features {X_test.shape} in {time.perf_counter() - t0:.0f}s")
    y_train = train["label"].to_numpy().astype(int)
    y_test = test["label"].to_numpy().astype(int)

    # Native units end to end: trees are invariant to per-feature monotonic
    # rescaling, so the scaler was pure cost (and unreadable attributions).
    Xtr = X_train.to_numpy(dtype=float)
    Xte = X_test.to_numpy(dtype=float)

    clf = build_model()
    # Note: LightGBM records auto-generated feature names at fit, so the
    # first nameless-numpy predict per process emits sklearn's "fitted with
    # feature names" UserWarning. Benign here by construction: train and
    # serving share the frozen column order positionally (same featurise,
    # same vocabulary copy), exactly like the nameless ensemble path. Fitting
    # or predicting with named DataFrames instead would fragment the shared
    # scoring pipeline (and cost the single-URL fast path its win) for zero
    # safety gain, so the pipeline stays numpy end to end.
    print("training LightGBM...")
    t0 = time.perf_counter()
    clf.fit(Xtr, y_train)
    print(f"trained in {time.perf_counter() - t0:.0f}s")

    y_pred = clf.predict(Xte)
    print(f"\naccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("classification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Legit (0)", "Phishing (1)"]
        )
    )

    a.assets_out.mkdir(parents=True, exist_ok=True)
    with open(a.assets_out / "gbm_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    shutil.copyfile(a.frozen_columns, a.assets_out / "feature_columns.pkl")
    stale = a.assets_out / "scaler.pkl"
    if stale.exists():
        stale.unlink()
        print(f"removed stale {stale} (scaler dropped; see module docstring)")
    print(f"\nwrote {a.assets_out}/gbm_model.pkl, feature_columns.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
