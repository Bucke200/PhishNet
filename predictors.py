"""Predictors for the harness.

Every class here satisfies the `eval.Predictor` protocol: `.name` and
`.score(urls) -> list[float]`, higher = more likely phishing.

The trivial ones are not filler. If `UrlShapeHeuristic` gets a good PR-AUC on
your test set, the dataset is leaking collection artifacts and no model result
from it means anything.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import numpy as np


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
            raw = 0.35 * min(len(u), 200) / 200 + 0.45 * min(depth, 6) / 6 + 0.2 * bool(p.query)
            out.append(float(np.clip(raw, 0, 1)))
        return out


class LegacyEnsemble:
    """The current PhishNet model, frozen as the baseline to beat.

    Feature extraction is imported from the canonical package — if this import
    fails, training/serving skew is still unresolved and the baseline number
    would be measuring the wrong code path anyway.
    """

    def __init__(
        self,
        model_path: str = "artifacts/urlset_model.pkl",
        scaler_path: str = "artifacts/scaler.pkl",
    ):
        from phishnet.features.extraction import extract_features  # noqa: PLC0415

        self._extract = extract_features
        self.model = pickle.loads(Path(model_path).read_bytes())
        self.scaler = pickle.loads(Path(scaler_path).read_bytes())
        self.name = f"legacy_ensemble({Path(model_path).name})"
        self.mode = (
            "predict_proba"
            if hasattr(self.model, "predict_proba")
            else "vote_fraction"
            if hasattr(self.model, "estimators_")
            else "hard_label"
        )

    def _features(self, urls: Sequence[str]) -> np.ndarray:
        X = np.asarray([list(self._extract(u).values()) for u in urls], dtype=float)
        return self.scaler.transform(X)

    def score(self, urls: Sequence[str]) -> list[float]:
        X = self._features(urls)
        if self.mode == "predict_proba":
            return self.model.predict_proba(X)[:, 1].tolist()
        if self.mode == "vote_fraction":
            # VotingClassifier(voting='hard') has no predict_proba. Averaging the
            # member votes gives a coarse score with n_estimators+1 levels —
            # enough to rank, nowhere near enough to threshold. The harness will
            # flag the low distinct-score count.
            votes = np.column_stack([e.predict(X) for e in self.model.estimators_])
            return votes.mean(axis=1).astype(float).tolist()
        return self.model.predict(X).astype(float).tolist()
