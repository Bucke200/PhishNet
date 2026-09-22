"""Tier-1 serving: Phase 3 row (a) as a plain LightGBM, pandas-free.

The headline champion (Phase 3 row (a), `ablation_lexical_assets`, 79
columns) is a raw ``LGBMClassifier``. Its 79 columns are lexical plus
``is_hosted_tenant``; it takes **no enrichment input**, so serving never
joins the snapshot and never builds a per-call DataFrame.

``load_row_a`` (``phishnet.snapshot.tier1``) remains the eval-mode
reference; this module is the serving path and is asserted bit-equal to it
(max abs diff 0.0) by the C1 identity test. The fast path:

    canonicalize_scheme -> comprehensive_phishing_features -> preallocated
    row in pinned column order -> is_hosted_tenant -> booster_.predict

reproduces ``featurise_frame`` + ``hosted_flag`` + ``predict_proba`` exactly
while skipping pandas (~6 ms) and the sklearn wrapper (~0.7 ms) per call.

Startup verification (C1): the model and column SHA256 are checked against
``model_manifest.json`` and the three frozen thresholds against
``reports/phase4.json`` (the trust anchor is the registered constant table
below). Any mismatch raises — there is no degraded mode.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from phishnet.enrichment.features import HOSTED_COLUMN, hosted_flag
from phishnet.features.extraction import (
    canonicalize_scheme,
    comprehensive_phishing_features,
)
from phishnet.verified_download import load_manifest, sha256_of_file

# Manifest artifact names (release/installed names, see model_manifest.json).
ROW_A_MODEL_NAME = "ablation_lexical_gbm_model.pkl"
ROW_A_COLUMNS_NAME = "ablation_lexical_feature_columns.pkl"

# Trust anchor: the registered frozen thresholds (phase6-preregistration §0,
# recomputed from calib and asserted, never swept). `reports/phase4.json` is
# the runtime source; these constants catch a mutated/drifted source file.
REGISTERED_THRESHOLDS: dict[str, float] = {
    "t_alert": 0.9269363298832987,
    "lower_edge": 0.6493076453312958,
    "t_1pct": 0.8780843789420926,
}

# Row (a) trained on scheme-canonicalized URLs (`train_config.json`:
# `scheme_source: manifest:drop`). The row (a) release assets carry no
# train_config sidecar, so the registered decision is stated here; the C1
# identity test fails loudly if this ever diverges from the eval path.
CANONICALIZE_SCHEME = True


def _package_assets_dir() -> Path:
    """Default assets dir: packaged ``urlset_ml_assets`` (verified download)."""
    override = os.getenv("PHISHNET_ML_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "urlset_ml_assets"


def _repo_root() -> Path:
    """Repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[3]


def _resolve_threshold_file(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    override = os.getenv("PHISHNET_THRESHOLDS_FILE")
    if override:
        return Path(override)
    return _repo_root() / "reports" / "phase4.json"


class Tier1Servable:
    """Loaded, hash-verified row (a) scorer with a pandas-free fast path."""

    def __init__(
        self,
        assets_dir: str | Path | None = None,
        threshold_file: str | Path | None = None,
        *,
        canonicalize: bool = CANONICALIZE_SCHEME,
        verify: bool = True,
    ):
        self.assets_dir = Path(assets_dir) if assets_dir else _package_assets_dir()
        self.canonicalize = canonicalize
        model_path = self.assets_dir / ROW_A_MODEL_NAME
        columns_path = self.assets_dir / ROW_A_COLUMNS_NAME
        for path in (model_path, columns_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. Fetch row (a) artifacts first:\n"
                    f"  uv run python -m phishnet.verified_download"
                )

        if verify:
            _, specs = load_manifest()
            self._verify_against_manifest(specs, model_path, columns_path)

        self.model_hash = sha256_of_file(model_path)
        self.columns_hash = sha256_of_file(columns_path)
        model = pickle.loads(model_path.read_bytes())
        booster = getattr(model, "booster_", None)
        if booster is None:
            raise TypeError(
                "row (a) is expected to be a LightGBM estimator with a "
                f"booster_; got {type(model).__name__}"
            )
        self._booster: Any = booster
        self.columns: list[str] = list(pickle.loads(columns_path.read_bytes()))
        if HOSTED_COLUMN not in self.columns:
            raise ValueError(
                f"row (a) vocabulary is missing {HOSTED_COLUMN!r}; the hosted "
                "flag cannot be filled and scores would diverge"
            )
        self._hosted_idx = self.columns.index(HOSTED_COLUMN)

        self.thresholds_file = _resolve_threshold_file(threshold_file)
        self.thresholds = self._load_thresholds(self.thresholds_file)
        digest = sha256_of_file(self.thresholds_file)
        self.thresholds_source = f"{self.thresholds_file.name}:{digest}"

    # -- verification ---------------------------------------------------
    def _verify_against_manifest(
        self,
        specs: dict[str, Any],
        model_path: Path,
        columns_path: Path,
    ) -> None:
        for name, path in (
            (ROW_A_MODEL_NAME, model_path),
            (ROW_A_COLUMNS_NAME, columns_path),
        ):
            spec = specs.get(name)
            if spec is None:
                raise RuntimeError(f"{name} missing from model_manifest.json")
            actual = sha256_of_file(path)
            if actual != spec.sha256:
                raise RuntimeError(
                    f"{name} SHA256 mismatch: {actual} != {spec.sha256} "
                    "(refusing to start)"
                )

    def _load_thresholds(self, path: Path) -> dict[str, float]:
        if not path.is_file():
            raise FileNotFoundError(
                f"threshold source {path} not found; set PHISHNET_THRESHOLDS_FILE"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded = {
            "t_alert": float(raw["t_alert"]),
            "lower_edge": float(raw["lower_edge"]),
            "t_1pct": float(raw["t_1pct"]),
        }
        for key, expected in REGISTERED_THRESHOLDS.items():
            if loaded[key] != expected:
                raise RuntimeError(
                    f"threshold {key} = {loaded[key]!r} != registered "
                    f"{expected!r} (refusing to start)"
                )
        return loaded

    # -- scoring --------------------------------------------------------
    def row(self, url: str) -> np.ndarray:
        """One feature row (1, n_columns), ``featurise_frame``-equal."""
        target = canonicalize_scheme(url) if self.canonicalize else url
        feats = comprehensive_phishing_features(target)
        row = np.empty(len(self.columns), dtype=float)
        for i, col in enumerate(self.columns):
            if i == self._hosted_idx:
                continue
            value = feats.get(col, 0)
            try:
                f = float(value)
            except (TypeError, ValueError):
                f = 0.0
            row[i] = 0.0 if math.isnan(f) else f
        row[self._hosted_idx] = float(hosted_flag([url])[0])
        return row.reshape(1, -1)

    def score_one(self, url: str) -> float:
        """P(phish) for one URL, via ``booster_.predict`` (no sklearn)."""
        return float(self._booster.predict(self.row(url))[0])

    def explain_one(self, url: str, top_k: int = 10) -> dict[str, Any]:
        """Top-k native tree-SHAP contributions for the scoring row."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        contrib = np.asarray(
            self._booster.predict(self.row(url), pred_contrib=True), dtype=float
        ).ravel()
        if contrib.shape[0] != len(self.columns) + 1:
            raise RuntimeError(
                f"expected {len(self.columns)} + bias contributions, "
                f"got {contrib.shape[0]}"
            )
        per_feature, bias = contrib[:-1], float(contrib[-1])
        order = np.argsort(-np.abs(per_feature), kind="stable")[:top_k]
        return {
            "features": [
                {
                    "feature": self.columns[int(i)],
                    "contribution": float(per_feature[int(i)]),
                }
                for i in order
            ],
            "bias": bias,
        }
