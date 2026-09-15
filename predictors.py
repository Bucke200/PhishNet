"""Predictors for the harness.

Every class here satisfies the `eval.Predictor` protocol: `.name` and
`.score(urls) -> list[float]`, higher = more likely phishing.

The trivial ones are not filler. If `UrlShapeHeuristic` gets a good PR-AUC on
your test set, the dataset is leaking collection artifacts and no model result
from it means anything.
"""

from __future__ import annotations

import hashlib
import math
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


def _default_cc_assets_dir() -> Path:
    override = os.getenv("PHISHNET_CC_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "backend" / "cc_ml_assets"


def _default_gbm_assets_dir() -> Path:
    override = os.getenv("PHISHNET_GBM_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "backend" / "gbm_assets"


def _default_gbm_iso_assets_dir() -> Path:
    override = os.getenv("PHISHNET_GBM_ISO_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "backend" / "gbm_iso_assets"


def _default_gbm_sig_assets_dir() -> Path:
    override = os.getenv("PHISHNET_GBM_SIG_ASSETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "backend" / "gbm_sig_assets"


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

    def _features_single(self, url: str) -> np.ndarray:
        """Single-URL path: straight into a preallocated row, no DataFrame.

        Exactly the extract -> reindex (missing = 0) -> to_numeric (coerce,
        NaN = 0) -> scale pipeline of :meth:`_features`, minus the per-call
        DataFrame construction that dominates n=1 latency (the probe calls
        ``score([u])`` per URL: ~130x the batched per-URL cost). Batches keep
        the pandas path; a dedicated test pins the two bit-for-bit equal.
        Scaler-less pipelines (``self.scaler is None``) return native units.
        """
        feats = self._extract(url)
        row = np.empty(len(self.columns), dtype=float)
        for i, col in enumerate(self.columns):
            v = feats.get(col, 0)
            try:
                f = float(v)
            except (TypeError, ValueError):
                f = 0.0
            row[i] = 0.0 if math.isnan(f) else f
        values = row.reshape(1, -1)
        if self.scaler is None:
            return np.asarray(values, dtype=float)
        return np.asarray(self.scaler.transform(values), dtype=float)

    def _features(self, urls: Sequence[str]) -> np.ndarray:
        # Same extract -> reindex -> coerce -> scale pipeline as
        # phishnet.api.preprocess_single_url_traditional. Any divergence here is
        # training/serving skew wearing an evaluation costume.
        if len(urls) == 1:
            return self._features_single(urls[0])
        frame = pd.DataFrame([self._extract(u) for u in urls])
        for col in self.columns:
            if col not in frame.columns:
                frame[col] = 0
        frame = frame[self.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        values = frame.to_numpy(dtype=float)
        if self.scaler is None:
            return np.asarray(values, dtype=float)
        return np.asarray(self.scaler.transform(values), dtype=float)

    def score(self, urls: Sequence[str]) -> list[float]:
        X = self._features(urls)
        # *_prefit are probability contracts like predict_proba
        # (CalibratedClassifierCV exposes predict_proba); vote_fraction and
        # hard_label stay on their branches below.
        if self.mode in ("predict_proba", "isotonic_prefit", "sigmoid_prefit"):
            return [float(v) for v in self.model.predict_proba(X)[:, 1]]
        if self.mode == "vote_fraction":
            votes = np.column_stack([e.predict(X) for e in self.model.estimators_])
            return [float(v) for v in votes.mean(axis=1)]
        return [float(v) for v in self.model.predict(X)]

    def explain(self, url: str, top_k: int = 10) -> dict[str, Any]:
        """Top-k exact tree-SHAP contributions for one URL (Phase 6 popup).

        Uses LightGBM's native ``pred_contrib`` — no ``shap`` runtime
        dependency. Contributions are in NATIVE feature units (GBM pipelines
        are scaler-less by class contract) and sum with the bias term to the
        raw margin; for calibrated wrappers that margin is the pre-map
        score, since the monotonic map preserves order but not levels.
        Raises :class:`ExplainUnsupportedError` when the model has no
        native contributions (voting ensembles) or would attribute in
        standardized units (any future scaled LGBM).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        native = _native_lgbm(self.model)
        if native is None:
            raise ExplainUnsupportedError(
                f"{self.name}: exact attribution needs a LightGBM estimator, "
                f"got {type(self.model).__name__}"
            )
        if self.scaler is not None:
            raise ExplainUnsupportedError(
                f"{self.name}: contributions would land in standardized units; "
                "native-unit pipelines only"
            )
        row = self._features_single(url)
        contrib = np.asarray(
            native.predict(row, pred_contrib=True), dtype=float
        ).ravel()
        if contrib.shape[0] != len(self.columns) + 1:
            raise ExplainUnsupportedError(
                f"{self.name}: expected {len(self.columns)} + bias contributions, "
                f"got {contrib.shape[0]}"
            )
        per_feature, bias = contrib[:-1], float(contrib[-1])
        order = np.argsort(-np.abs(per_feature), kind="stable")[:top_k]
        return {
            "features": [
                (self.columns[int(i)], float(per_feature[int(i)])) for i in order
            ],
            "bias": bias,
            "margin": float(per_feature.sum() + bias),
        }


def _require_member_proba(estimators: Sequence[Any]) -> None:
    """Guard: soft voting is only defined when every member predicts probas."""
    missing = [type(e).__name__ for e in estimators if not hasattr(e, "predict_proba")]
    if missing:
        raise TypeError(
            "soft voting needs predict_proba on all members, missing on: "
            + ", ".join(missing)
        )


def _sha256_bytes(data: bytes) -> str:
    """Hex SHA256 of raw file bytes (asset identity, never unpickled)."""
    return hashlib.sha256(data).hexdigest()


class ExplainUnsupportedError(RuntimeError):
    """A predictor model cannot produce native tree SHAP contributions."""


def _native_lgbm(model: Any) -> Any | None:
    """Underlying fitted LGBMClassifier, unwrapping prefit calibrators.

    Raw LightGBM estimators expose it directly; CalibratedClassifierCV
    with cv="prefit" holds exactly one fitted calibrator whose estimator
    is the base model. Anything else (voting ensembles, ad-hoc scorers)
    returns None: no exact per-feature attribution exists for it.
    """
    if "lightgbm" in type(model).__module__:
        return model
    calibrators = getattr(model, "calibrated_classifiers_", None)
    if calibrators:
        first = calibrators[0]
        inner = (
            getattr(first, "estimator", None)
            or getattr(first, "base_estimator", None)
            or getattr(first, "estimator_", None)
        )
        if inner is not None and "lightgbm" in type(inner).__module__:
            return inner
    return None


def _mean_member_proba(estimators: Sequence[Any], X: np.ndarray) -> list[float]:
    """Mean member P(phish): the single soft-vote combination rule.

    Shared by every soft-voted predictor so the bodies can never drift:
    callers differ only in which fitted members they pass in. Requires
    every member to expose ``predict_proba`` (enforced by
    ``_require_member_proba`` at init, not here, so this stays pure).
    """
    probas = np.column_stack([e.predict_proba(X)[:, 1] for e in estimators])
    return [float(v) for v in probas.mean(axis=1)]


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
        return _mean_member_proba(self.model.estimators_, self._features(urls))


class CcRetrained(LegacyEnsemble):
    """Same architecture as LegacyEnsemble, retrained on the CC population.

    Identical estimator types, hyperparameters, frozen 78-column
    vocabulary and extract -> reindex -> scale pipeline (see
    ml_training/train_cc_split.py); only the training rows (a
    domain-disjoint, leakage-audited split) and the fitted scaler differ.
    Scores via predict_proba when available, else member vote fractions.
    """

    # Asset contract, overridden by subclasses training other estimators on
    # the same population (same pickle layout, same frozen vocabulary).
    model_filename = "cc_ensemble_model.pkl"
    train_script = "ml_training/train_cc_split.py"
    # Pure-tree estimators skip the scaler (a no-op for trees whose only
    # effect is unreadable standardized units). The ensemble keeps it:
    # LogisticRegression genuinely needs scaling.
    uses_scaler = True

    def _resolve_dir(self, assets_dir: str | None) -> Path:
        if assets_dir:
            return Path(assets_dir)
        return _default_cc_assets_dir()

    def __init__(self, assets_dir: str | None = None):
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            comprehensive_phishing_features,
        )

        self._extract = comprehensive_phishing_features
        d = self._resolve_dir(assets_dir)
        required = [self.model_filename, "feature_columns.pkl"]
        if self.uses_scaler:
            required.append("scaler.pkl")
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"{missing} not in {d}. Train them first:\n"
                f"  python {self.train_script} --assets-out {d}"
            )
        model_raw = (d / self.model_filename).read_bytes()
        self.model: Any = pickle.loads(model_raw)
        columns_raw = (d / "feature_columns.pkl").read_bytes()
        self.columns: list[str] = list(pickle.loads(columns_raw))
        if self.uses_scaler:
            scaler_raw = (d / "scaler.pkl").read_bytes()
            self.scaler: Any = pickle.loads(scaler_raw)
            scaler_sha: str | None = _sha256_bytes(scaler_raw)
        else:
            self.scaler = None
            scaler_sha = None
        # Asset identity: model + columns + scaler PRESENCE. A columns-hash
        # alone misses representation changes with an unchanged vocabulary
        # (e.g. dropping the scaler), so absence is recorded as explicit
        # null, never omitted. evaluate() persists this in the report JSON.
        self.asset_fingerprint: dict[str, str | None] = {
            "model": _sha256_bytes(model_raw),
            "scaler": scaler_sha,
            "columns": _sha256_bytes(columns_raw),
        }
        self.name = "cc_retrained(hard-vote)"
        self.mode = (
            "predict_proba"
            if hasattr(self.model, "predict_proba")
            else "vote_fraction"
            if hasattr(self.model, "estimators_")
            else "hard_label"
        )


class GbmSingle(CcRetrained):
    """Single LightGBM trained on the CC population (Step 4 ablation).

    Same frozen 78-column vocabulary, same extract -> reindex pipeline in
    NATIVE units (no scaler: a no-op for trees whose only effect was
    unreadable standardized contributions), same training rows as
    :class:`CcRetrained` (see ml_training/train_gbm.py). The only variable
    under test is the estimator: one ``LGBMClassifier`` instead of the
    voted ensemble. ``LGBMClassifier`` exposes ``predict_proba``, so the
    inherited :meth:`LegacyEnsemble.score` takes the probability path — no
    new scoring code, and the single-URL fast path applies unchanged.

    Non-shipped ablation reference: the full-train weights below have no
    honest threshold (any held-out slice is data they trained on). The
    deployable lineage is :class:`GbmRefit`.
    """

    model_filename = "gbm_model.pkl"
    train_script = "ml_training/train_gbm.py"
    uses_scaler = False

    def _resolve_dir(self, assets_dir: str | None) -> Path:
        if assets_dir:
            return Path(assets_dir)
        return _default_gbm_assets_dir()

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        self.name = "gbm_single"


class CalibratedGbm(CcRetrained):
    """Isotonic-calibrated GBM (Step 5).

    Same frozen vocabulary, pipeline, and training population as
    :class:`GbmSingle`, except the base estimator is refit on the fit
    partition and wrapped in ``CalibratedClassifierCV(isotonic, prefit)``
    on a domain-disjoint calibration slice (see
    ml_training/calibrate_gbm.py). ``predict_proba`` takes the inherited
    scoring path; ``mode`` records the calibration method so reports
    distinguish this contract from the uncalibrated one.
    """

    model_filename = "calibrated_gbm.pkl"
    train_script = "ml_training/calibrate_gbm.py"
    uses_scaler = False

    def _resolve_dir(self, assets_dir: str | None) -> Path:
        if assets_dir:
            return Path(assets_dir)
        return _default_gbm_iso_assets_dir()

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        self.name = "gbm_isotonic"
        self.mode = "isotonic_prefit"


class SigmoidGbm(CalibratedGbm):
    """Platt-scaled GBM: same slice, same script, two-parameter calibrator.

    The capacity diagnostic against :class:`CalibratedGbm`: if sigmoid
    transfers where isotonic does not, the failure was calibrator capacity
    (isotonic overfitting the slice), not era drift — and no three-band
    population rebuild is needed. If both degrade, the cause is era drift.
    """

    train_script = "ml_training/calibrate_gbm.py --method sigmoid"

    def _resolve_dir(self, assets_dir: str | None) -> Path:
        if assets_dir:
            return Path(assets_dir)
        return _default_gbm_sig_assets_dir()

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        self.name = "gbm_sigmoid"
        self.mode = "sigmoid_prefit"


class GbmRefit(CalibratedGbm):
    """Uncalibrated refit base: the deployable champion lineage.

    The same LGBM weights the isotonic/sigmoid wrappers calibrate, served
    without a map: native units, inherited probability scoring. This — not
    :class:`GbmSingle` — is what ships, because it alone has an honest
    fixed threshold (fit partition never saw the calibration slice the
    threshold comes from). Loads ``refit_base.pkl`` from the iso asset dir,
    sharing its vocabulary; the sigmoid run's dir holds an identical copy.
    """

    model_filename = "refit_base.pkl"
    train_script = "ml_training/calibrate_gbm.py"

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        self.name = "gbm_refit"
        # Uncalibrated LGBM: the inherited chain would have resolved
        # predict_proba before CalibratedGbm.__init__ overwrote it — restore
        # that contract explicitly rather than reporting a map we don't serve.
        self.mode = "predict_proba"


class CcSoftVote(CcRetrained):
    """CC-retrained members, soft-voted. No refit, no artifact change.

    Same combination rule as :class:`SoftVoteEnsemble` (shared
    ``_mean_member_proba`` helper) applied to the CC-retrained members:
    continuous mean member probability instead of the 5-level hard-vote
    fraction. Fails loudly at init if any member lacks ``predict_proba``.
    """

    def __init__(self, assets_dir: str | None = None):
        super().__init__(assets_dir=assets_dir)
        _require_member_proba(self.model.estimators_)
        self.name = "cc_soft_vote"
        self.mode = "soft_vote"

    def score(self, urls: Sequence[str]) -> list[float]:
        return _mean_member_proba(self.model.estimators_, self._features(urls))
