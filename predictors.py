"""Predictors for the harness.

Every class here satisfies the `eval.Predictor` protocol: `.name` and
`.score(urls) -> list[float]`, higher = more likely phishing.

The trivial ones are not filler. If `UrlShapeHeuristic` gets a good PR-AUC on
your test set, the dataset is leaking collection artifacts and no model result
from it means anything.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


class ConstantScorer:
    """Floor. Everything must beat this."""

    def __init__(self, value: float = 0.5):
        self.value = float(value)
        self.name = f"constant({value})"

    def score(self, urls: Sequence[str]) -> list[float]:
        return [self.value] * len(urls)


class RandomScorer:
    """Deterministic per-URL noise. PR-AUC should land on the base rate."""

    name = "random"

    def score(self, urls: Sequence[str]) -> list[float]:
        out = []
        for u in urls:
            h = hashlib.sha256(u.encode()).digest()
            out.append(int.from_bytes(h[:4], "big") / 2**32)
        return out


class UrlShapeHeuristic:
    """Leak canary: length and path depth only, no phishing knowledge at all.

    A 2017-era dataset where benign rows are bare domains and phishing rows are
    full URLs will hand this thing a PR-AUC over 0.9.
    """

    name = "url_shape_canary"

    def score(self, urls: Sequence[str]) -> list[float]:
        out = []
        for u in urls:
            p = urlparse(u if "://" in u else "http://" + u)
            depth = len([s for s in p.path.split("/") if s])
            raw = (
                0.35 * min(len(u), 200) / 200
                + 0.45 * min(depth, 6) / 6
                + 0.2 * bool(p.query)
            )
            out.append(float(np.clip(raw, 0, 1)))
        return out


def _default_assets_dir() -> Path:
    override = os.getenv("PHISHNET_ML_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "src" / "phishnet" / "urlset_ml_assets"


class LegacyEnsemble:
    """The current PhishNet model, frozen as the baseline to beat."""

    def __init__(self, assets_dir: str | None = None):
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            comprehensive_phishing_features,
        )

        self._extract = comprehensive_phishing_features
        d = Path(assets_dir) if assets_dir else _default_assets_dir()
        missing = [
            f
            for f in ("urlset_ensemble_model.pkl", "scaler.pkl", "feature_columns.pkl")
            if not (d / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{missing} not in {d}. Fetch them first:\n"
                f"  uv run python -m phishnet.verified_download"
            )
        self.model: Any = pickle.loads((d / "urlset_ensemble_model.pkl").read_bytes())
        self.scaler: Any = pickle.loads((d / "scaler.pkl").read_bytes())
        self.columns: list[str] = list(
            pickle.loads((d / "feature_columns.pkl").read_bytes())
        )
        self.name = "legacy_ensemble(models-v1)"
        self.mode = (
            "predict_proba"
            if hasattr(self.model, "predict_proba")
            else "vote_fraction"
            if hasattr(self.model, "estimators_")
            else "hard_label"
        )

    def _features(self, urls: Sequence[str]) -> np.ndarray:
        # Same extract -> reindex -> coerce -> scale pipeline as
        # phishnet.api.preprocess_single_url_traditional. Any divergence here is
        # training/serving skew wearing an evaluation costume.
        frame = pd.DataFrame([self._extract(u) for u in urls])
        for col in self.columns:
            if col not in frame.columns:
                frame[col] = 0
        frame = frame[self.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        return np.asarray(
            self.scaler.transform(frame.to_numpy(dtype=float)), dtype=float
        )

    def score(self, urls: Sequence[str]) -> list[float]:
        X = self._features(urls)
        if self.mode == "predict_proba":
            return [float(v) for v in self.model.predict_proba(X)[:, 1]]
        if self.mode == "vote_fraction":
            votes = np.column_stack([e.predict(X) for e in self.model.estimators_])
            return [float(v) for v in votes.mean(axis=1)]
        return [float(v) for v in self.model.predict(X)]


def _require_member_proba(estimators: Sequence[Any]) -> None:
    """Guard: soft voting is only defined when every member predicts probas."""
    missing = [type(e).__name__ for e in estimators if not hasattr(e, "predict_proba")]
    if missing:
        raise TypeError(
            "soft voting needs predict_proba on all members, missing on: "
            + ", ".join(missing)
        )


class SoftVoteEnsemble(LegacyEnsemble):
    """Phase 2 candidate: same frozen model, same preprocessing, soft votes.

    The models-v1 members all expose fitted ``predict_proba`` (verified at
    init), so the mean member probability is a continuous ranking score with
    the same train/test data and feature pipeline as the hard-vote baseline.
    No refit, no artifact change, no threshold tuning — the only difference
    from ``LegacyEnsemble`` is the combination rule.
    """

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        _require_member_proba(self.model.estimators_)
        self.name = "soft_vote(models-v1)"
        self.mode = "soft_vote"

    def score(self, urls: Sequence[str]) -> list[float]:
        X = self._features(urls)
        probas = np.column_stack(
            [e.predict_proba(X)[:, 1] for e in self.model.estimators_]
        )
        return [float(v) for v in probas.mean(axis=1)]
