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
import json
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

FROZEN_COLUMNS = (
    Path("src/phishnet/urlset_ml_assets/feature_columns.pkl")
    if Path("src/phishnet/urlset_ml_assets/feature_columns.pkl").exists()
    else Path("backend/urlset_ml_assets/feature_columns.pkl")
)

TRAIN_CONFIG_FILENAME = "train_config.json"


def resolve_canonicalize(
    split_dir: Path, cli: bool | None
) -> tuple[bool, str]:
    """Follow the population manifest's scheme decision, not a default.

    Returns (canonicalize, source). An explicit CLI flag always wins
    (source "flag"); otherwise the split manifest's
    ``is_https_rule.decision`` governs ("manifest:drop" → True,
    "manifest:keep" → False). Splits predating the rule record nothing —
    they default to False ("absent-default"), preserving the Phase 2
    behavior of every existing asset directory.
    """
    if cli is not None:
        return cli, "flag"
    manifest = split_dir / "manifest.json"
    try:
        decision = (
            json.loads(manifest.read_text(encoding="utf-8"))
            .get("is_https_rule", {})
            .get("decision")
        )
    except (OSError, ValueError):
        decision = None
    if decision == "drop":
        return True, "manifest:drop"
    if decision == "keep":
        return False, "manifest:keep"
    return False, "absent-default"


def write_train_config(
    assets_out: Path,
    *,
    canonicalize: bool,
    scheme_source: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Persist the representation decision beside the weights.

    The predictor loads this at init (see CcRetrained) so the scoring
    path can never disagree with training about the scheme — and the
    setting lands in asset_fingerprint, so every report shows which way
    the population's rule went.
    """
    config: dict[str, object] = {
        "canonicalize_scheme": canonicalize,
        "scheme_source": scheme_source,
    }
    if extra:
        config.update(extra)
    (assets_out / TRAIN_CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def featurise(
    urls: list[str], columns: list[str], *, canonicalize: bool = False
) -> pd.DataFrame:
    """Same shared pipeline as serving/eval (see featurise_frame).

    ``canonicalize`` follows the split manifest's scheme rule: row (a)
    passes True when the rule says DROP, so the baseline trains on the
    same scheme-blind representation the enriched rows use.
    """
    from phishnet.features.extraction import (  # type: ignore[import-untyped]
        featurise_frame,
    )

    parts = []
    for i in range(0, len(urls), 5000):
        parts.append(
            featurise_frame(
                urls[i : i + 5000], columns, canonicalize=canonicalize
            )
        )
        if i + 5000 < len(urls):
            print(f"  extracted {i + 5000:,}/{len(urls):,}", flush=True)
    cleaned: pd.DataFrame = (
        parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
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
    p.add_argument(
        "--canonicalize-scheme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="strip the leading scheme before featurizing. Default (unset) "
        "follows the split manifest's is_https_rule decision; an explicit "
        "--canonicalize-scheme / --no-canonicalize-scheme overrides it.",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    columns: list[str] = pickle.loads(a.frozen_columns.read_bytes())
    print(f"frozen vocabulary: {len(columns)} columns from {a.frozen_columns}")

    canonicalize, scheme_source = resolve_canonicalize(
        a.split_dir, a.canonicalize_scheme
    )
    print(f"scheme representation: canonicalize={canonicalize} ({scheme_source})")

    train = pd.read_csv(a.split_dir / "train.csv")
    test = pd.read_csv(a.split_dir / "test.csv")
    print(f"train {len(train):,} / test {len(test):,}")
    # Shuffle train (the split builder orders by population, not for SGD —
    # trees don't care, but a shuffled artifact trains identically anywhere).
    train = shuffle(train, random_state=a.seed).reset_index(drop=True)

    print("featurising train..." + (" (scheme-canonicalized)" if canonicalize else ""))
    t0 = time.perf_counter()
    X_train = featurise(
        train["url"].astype(str).tolist(),
        columns,
        canonicalize=canonicalize,
    )
    print(f"train features {X_train.shape} in {time.perf_counter() - t0:.0f}s")
    print("featurising test...")
    t0 = time.perf_counter()
    X_test = featurise(
        test["url"].astype(str).tolist(),
        columns,
        canonicalize=canonicalize,
    )
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
    write_train_config(
        a.assets_out,
        canonicalize=canonicalize,
        scheme_source=scheme_source,
        extra={"split_dir": str(a.split_dir), "seed": a.seed},
    )
    print(
        f"\nwrote {a.assets_out}/gbm_model.pkl, feature_columns.pkl, "
        f"{TRAIN_CONFIG_FILENAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
