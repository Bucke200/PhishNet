"""Train one Phase 3 ablation row (Step 5).

One estimator (the shared LightGBM from train_gbm.build_model, native
units, no scaler), one feature group, one population. Groups differ ONLY
in columns — same rows, same seed, same hyperparameters — so the
ablation table measures signals, not training noise:

* lexical: the frozen 78 (row a — the only baseline deltas may reference)
* age:     78 + domain_age_days, age_known
* ct:      78 + ct_age_days, ct_cert_count_pre, ct_known
* all:     78 + all five enriched columns

Inputs are a --phase3 split (rows carry first_seen/survival_stratum for
the join) plus a sealed snapshot under one pinned run. Assets mirror the
GbmSingle layout (gbm_model.pkl + feature_columns.pkl + train_config.json)
plus ablation-report.json, so a future enriched predictor loads them
unchanged. Honest numbers come from eval.py, never from the accuracy
printed here.

Usage:
  python ml_training/train_ablation.py --split-dir data/splits-p3 \\
      --snapshot data/enrichment-<index>-<date>.jsonl --run-id run-1 \\
      --group all --assets-out backend/ablation_all_assets
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Importable when run from the repo root (matches the pytest pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_training.train_gbm import (
    FROZEN_COLUMNS,
    TRAIN_CONFIG_FILENAME,
    build_model,
    resolve_canonicalize,
    write_train_config,
)
from phishnet.enrichment.features import ENRICHED_COLUMNS, build_feature_table

GROUPS: dict[str, list[str]] = {
    "lexical": [],
    "age": ["domain_age_days", "age_known"],
    "ct": ["ct_age_days", "ct_cert_count_pre", "ct_known"],
    "all": list(ENRICHED_COLUMNS),
}

REQUIRED_SPLIT_COLUMNS = [
    "url",
    "label",
    "first_seen",
    "registrable_domain",
    "survival_stratum",
]


def group_columns(frozen: list[str], group: str) -> list[str]:
    """Lexical vocabulary plus the group's enriched columns, in order."""
    return [*frozen, *GROUPS[group]]


def split_rows(frame: pd.DataFrame, name: str) -> list[dict[str, object]]:
    """Split rows as join input; refuse legacy splits with a clear error."""
    missing = [c for c in REQUIRED_SPLIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} lacks phase-3 columns {missing}: rebuild the population "
            "with build_splits.py --phase3 (pinned populations stay frozen)"
        )
    return [
        {
            "url": str(r["url"]),
            "label": int(r["label"]),
            "first_seen": str(r["first_seen"]),
            "survival_stratum": str(r["survival_stratum"]),
        }
        for _, r in frame.iterrows()
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--group", choices=sorted(GROUPS), required=True)
    p.add_argument("--assets-out", type=Path, required=True)
    p.add_argument("--frozen-columns", type=Path, default=FROZEN_COLUMNS)
    p.add_argument(
        "--canonicalize-scheme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="strip the leading scheme before featurizing. Default (unset) "
        "follows the split manifest's is_https_rule decision.",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    frozen: list[str] = pickle.loads(a.frozen_columns.read_bytes())
    columns = group_columns(frozen, a.group)
    print(
        f"group {a.group}: {len(columns)} columns "
        f"({len(frozen)} lexical + {len(columns) - len(frozen)} enriched)"
    )

    canonicalize, scheme_source = resolve_canonicalize(
        a.split_dir, a.canonicalize_scheme
    )
    print(f"scheme representation: canonicalize={canonicalize} ({scheme_source})")

    selection = {"rule": "pinned-run", "run_id": a.run_id}
    train = pd.read_csv(a.split_dir / "train.csv")
    test = pd.read_csv(a.split_dir / "test.csv")
    print(f"train {len(train):,} / test {len(test):,}")
    train = shuffle(train, random_state=a.seed).reset_index(drop=True)

    print("building train feature table...")
    t0 = time.perf_counter()
    X_train, vocabulary, train_manifest = build_feature_table(
        split_rows(train, "train.csv"),
        a.snapshot,
        selection,
        frozen,
        canonicalize=canonicalize,
    )
    print(f"train table {X_train.shape} in {time.perf_counter() - t0:.0f}s")
    print("building test feature table...")
    t0 = time.perf_counter()
    X_test, test_vocab, test_manifest = build_feature_table(
        split_rows(test, "test.csv"),
        a.snapshot,
        selection,
        frozen,
        canonicalize=canonicalize,
    )
    print(f"test table {X_test.shape} in {time.perf_counter() - t0:.0f}s")
    assert test_vocab == vocabulary
    X_train = X_train[columns]
    X_test = X_test[columns]
    y_train = train["label"].to_numpy().astype(int)
    y_test = test["label"].to_numpy().astype(int)

    clf = build_model()
    print(f"training LightGBM ({a.group})...")
    t0 = time.perf_counter()
    clf.fit(X_train.to_numpy(dtype=float), y_train)
    Xtr = X_train.to_numpy(dtype=float)
    Xte = X_test.to_numpy(dtype=float)
    print(f"trained in {time.perf_counter() - t0:.0f}s")
    print(
        f"train accuracy: {accuracy_score(y_train, clf.predict(Xtr)):.4f} "
        "(in-sample; honest numbers come from eval.py)"
    )
    print(
        f"test accuracy: {accuracy_score(y_test, clf.predict(Xte)):.4f} "
        "(same-population reference only)"
    )

    a.assets_out.mkdir(parents=True, exist_ok=True)
    with open(a.assets_out / "gbm_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(a.assets_out / "feature_columns.pkl", "wb") as f:
        pickle.dump(columns, f)
    stale = a.assets_out / "scaler.pkl"
    if stale.exists():
        stale.unlink()
    write_train_config(
        a.assets_out,
        canonicalize=canonicalize,
        scheme_source=scheme_source,
        extra={
            "split_dir": str(a.split_dir),
            "snapshot": str(a.snapshot),
            "run_id": a.run_id,
            "group": a.group,
            "seed": a.seed,
        },
    )
    report = {
        "group": a.group,
        "columns": columns,
        "n_train_rows": len(train),
        "n_train_domains": int(train["registrable_domain"].nunique()),
        "n_test_rows": len(test),
        "n_test_domains": int(test["registrable_domain"].nunique()),
        "train_join": train_manifest["join"],
        "test_join": test_manifest["join"],
        "note": "same rows/seed/hyperparameters across groups; "
        "the table measures signals, not training noise",
    }
    (a.assets_out / "ablation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"\nwrote {a.assets_out}/gbm_model.pkl, feature_columns.pkl, "
        f"{TRAIN_CONFIG_FILENAME}, ablation-report.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
