"""Retrain the unchanged URLSet ensemble architecture on a CC-trial split.

Same architecture as ml_training/train_urlset.py: the canonical
comprehensive_phishing_features extractor, the frozen 78-column
vocabulary (tld dropped), a StandardScaler fit on the new train set,
and the hard-voting VotingClassifier over RandomForest /
LogisticRegression / DecisionTree / GradientBoosting with identical
hyperparameters. The only deliberate change is the training
population: a domain-disjoint, leakage-audited split (default
data/splits-cc) instead of urlset.csv.

Outputs (deployment artifacts, git-ignored like the models-v1 set):
  <assets-out>/cc_ensemble_model.pkl
  <assets-out>/scaler.pkl
  <assets-out>/feature_columns.pkl  (copy of the frozen vocabulary)

Usage:
  python ml_training/train_cc_split.py
  python ml_training/train_cc_split.py --split-dir data/splits-cc \
      --assets-out backend/cc_ml_assets
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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


def build_ensemble() -> VotingClassifier:
    """Identical estimators and hyperparameters to train_urlset.py."""
    rf_clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1, verbose=0
    )
    lr_clf = LogisticRegression(
        random_state=42, class_weight="balanced", solver="liblinear", max_iter=1000
    )
    dt_clf = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=0)
    return VotingClassifier(
        estimators=[("rf", rf_clf), ("lr", lr_clf), ("dt", dt_clf), ("gb", gb_clf)],
        voting="hard",
        n_jobs=-1,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=Path("data/splits-cc"))
    p.add_argument("--assets-out", type=Path, default=Path("backend/cc_ml_assets"))
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

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.to_numpy(dtype=float))
    Xte = scaler.transform(X_test.to_numpy(dtype=float))

    clf = build_ensemble()
    print("training ensemble...")
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
    with open(a.assets_out / "cc_ensemble_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(a.assets_out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    shutil.copyfile(a.frozen_columns, a.assets_out / "feature_columns.pkl")
    print(
        f"\nwrote {a.assets_out}/cc_ensemble_model.pkl, scaler.pkl, feature_columns.pkl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
