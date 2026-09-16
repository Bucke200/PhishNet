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

    def __init__(self, assets_dir: str | None = None, *, canonicalize: bool = False):
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            canonicalize_scheme,
            comprehensive_phishing_features,
        )

        self._extract = comprehensive_phishing_features
        self._canonicalize_scheme = canonicalize_scheme
        # Scheme switch (shared with the training path): when the split
        # manifest's scheme rule says DROP, the champion scores
        # scheme-canonicalized URLs — the same representation row (a)
        # trained on. One decision, one switch, both paths.
        self.canonicalize = canonicalize
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
        if getattr(self, "canonicalize", False):
            # Idempotent (a stripped URL has no leading scheme to strip),
            # so the len==1 delegation in _features re-applying it is a
            # no-op, not a double transform.
            url = self._canonicalize_scheme(url)
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
        if getattr(self, "canonicalize", False):
            urls = [self._canonicalize_scheme(u) for u in urls]
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


def _read_train_config(assets_dir: Path) -> dict[str, Any]:
    """Training-time representation decisions, {} when absent (pre-sidecar)."""
    import json

    try:
        return dict(
            json.loads((assets_dir / "train_config.json").read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return {}


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

    def __init__(
        self, assets_dir: str | None = None, *, canonicalize: bool | None = None
    ):
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            canonicalize_scheme,
            comprehensive_phishing_features,
        )

        self._extract = comprehensive_phishing_features
        self._canonicalize_scheme = canonicalize_scheme
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
        # Scheme representation follows the population manifest's decision
        # (written to train_config.json by the training script), never a
        # class default: an explicit constructor flag wins, else the
        # sidecar, else False (pre-sidecar assets keep Phase 2 behavior).
        train_config = _read_train_config(d)
        if canonicalize is None:
            canonicalize = bool(train_config.get("canonicalize_scheme", False))
            self.scheme_source = str(
                train_config.get("scheme_source", "absent-default")
            )
        else:
            self.scheme_source = "flag"
        self.canonicalize = canonicalize
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
        # Asset identity: model + columns + scaler PRESENCE + scheme
        # representation. A columns-hash alone misses representation
        # changes with an unchanged vocabulary (e.g. dropping the scaler,
        # or canonicalizing the scheme), so both ride here explicitly —
        # every report shows which way the population's rule went.
        # evaluate() persists this in the report JSON.
        self.asset_fingerprint: dict[str, str | None] = {
            "model": _sha256_bytes(model_raw),
            "scaler": scaler_sha,
            "columns": _sha256_bytes(columns_raw),
            "canonicalize_scheme": "true" if self.canonicalize else "false",
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

    def __init__(
        self, assets_dir: str | None = None, *, canonicalize: bool | None = None
    ):
        super().__init__(assets_dir=assets_dir, canonicalize=canonicalize)
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

    def __init__(
        self, assets_dir: str | None = None, *, canonicalize: bool | None = None
    ):
        super().__init__(assets_dir=assets_dir, canonicalize=canonicalize)
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

    def __init__(
        self, assets_dir: str | None = None, *, canonicalize: bool | None = None
    ):
        super().__init__(assets_dir=assets_dir, canonicalize=canonicalize)
        self.name = "gbm_refit"
        # Uncalibrated LGBM: the inherited chain would have resolved
        # predict_proba before CalibratedGbm.__init__ overwrote it — restore
        # that contract explicitly rather than reporting a map we don't serve.
        self.mode = "predict_proba"


class GbmRefitWithEnrichment(GbmRefit):
    """Phase 3 champion path: same weights, enrichment stub on-path.

    The Phase 2 champion (`GbmRefit`) stays byte-for-byte identical so
    re-runs of Phase 2 reports never include stub cost ("latency measured
    in the serving benchmark, not eval reports"). This subclass resolves
    every scored URL through an enrichment provider — default None (plain
    lexical scoring); pass the shared unknown stub to measure the tier-1
    p50 cache-miss floor in the serving benchmark. Results are discarded
    by the lexical weights, so scores are unchanged; a provider error
    degrades to plain scoring, never into it. `api.py` wiring waits for
    Phase 6 with the servability fix.
    """

    def __init__(
        self,
        assets_dir: str | None = None,
        provider: Any = None,
        *,
        canonicalize: bool | None = None,
    ):
        # canonicalize=None follows the population manifest via the
        # assets' train_config.json (DROP → True, KEEP → False); an
        # explicit flag wins. Never a class-level default that could
        # disagree with the decision the weights trained under.
        super().__init__(assets_dir=assets_dir, canonicalize=canonicalize)
        self.name = "gbm_refit+enrichment"
        self.enrichment_provider = provider

    def score(self, urls: Sequence[str]) -> list[float]:
        provider = getattr(self, "enrichment_provider", None)
        if provider is not None:
            try:
                provider.lookup_many(list(urls))
            except Exception:
                pass
        return super().score(urls)


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


class EnrichedGbm(CcRetrained):
    """Ablation-row scorer: enriched vocab + snapshot join at score time.

    Loads ``train_ablation`` assets (gbm_model.pkl + feature_columns.pkl
    subset + train_config.json) and serves them with the same snapshot
    join the tables trained on: per URL, the cache key resolves the
    pinned run's raw payload and the row's ``first_seen`` derives the
    five enriched features (see ``enrichment.join.derive_row``).

    ``first_seen`` sourcing (explicit, never silent): ``first_seen_csv``
    maps url → observation timestamp (the evaluated split CSV qualifies
    — timestamps are observation metadata, never labels, exactly what
    production substitutes with request time). With a map supplied (eval
    mode), a miss RAISES: scoring at now would compute
    now − creation_date, the exact future-knowledge leak the
    point-in-time design exists to prevent. Without a map (production
    mode only), every URL scores at now and the count lands in
    ``n_now_fallback``.
    """

    model_filename = "gbm_model.pkl"
    train_script = "ml_training/train_ablation.py"
    uses_scaler = False

    def __init__(
        self,
        assets_dir: str | None = None,
        snapshot: str | None = None,
        run_id: str | None = None,
        first_seen_csv: str | None = None,
        *,
        canonicalize: bool | None = None,
    ):
        from phishnet.enrichment.join import derive_row  # type: ignore[import-untyped]
        from phishnet.enrichment.key import (  # type: ignore[import-untyped]
            cache_key as _cache_key,
        )
        from phishnet.enrichment.store import (  # type: ignore[import-untyped]
            load_pinned_run,
            sha256_file,
        )

        if not snapshot or not run_id:
            raise ValueError("EnrichedGbm needs snapshot= and run_id= (pinned run)")
        super().__init__(assets_dir=assets_dir, canonicalize=canonicalize)
        self._derive_row = derive_row
        self._cache_key = _cache_key
        snap_path = Path(snapshot)
        self._table = load_pinned_run(snap_path, run_id)
        self.snapshot = str(snap_path)
        self.run_id = run_id
        self.first_seen_map: dict[str, str] = {}
        if first_seen_csv is not None:
            from phishnet.features.extraction import (  # type: ignore[import-untyped]
                canonicalize_scheme as _canon,
            )

            frame = pd.read_csv(first_seen_csv, usecols=["url", "first_seen"])
            # Index by raw AND canonicalized URL: a scheme switch between
            # the map's spelling and score-time spelling must not turn
            # every row into a fallback.
            for u, stamp in zip(
                frame["url"].astype(str),
                frame["first_seen"].astype(str),
                strict=True,
            ):
                self.first_seen_map[u] = stamp
                self.first_seen_map.setdefault(_canon(u), stamp)
        self.eval_mode = first_seen_csv is not None
        self.n_now_fallback = 0
        group = _read_train_config(self._resolve_dir(assets_dir)).get("group", "?")
        self.name = f"enriched_gbm({group})"
        self.mode = "predict_proba"
        # Fingerprint extends the inherited one: same weights under a
        # different snapshot/run would be a different system.
        self.asset_fingerprint = {
            **self.asset_fingerprint,
            "snapshot": sha256_file(snap_path),
            "snapshot_run": run_id,
        }

    def _first_seen(self, url: str) -> tuple[str, bool]:
        """Observation timestamp for a URL.

        Exact match first, then the canonicalized spelling (a scheme
        switch between the map's spelling and score-time spelling must
        not turn rows into fallbacks). Eval mode (map supplied): a miss
        on both raises — falling back to now would leak future knowledge
        into the score. Production mode (no map): now, counted in
        ``n_now_fallback``.
        """
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            canonicalize_scheme as _canon,
        )

        hit = self.first_seen_map.get(url)
        if hit is None:
            hit = self.first_seen_map.get(_canon(url))
        if hit is not None:
            return hit, False
        if self.eval_mode:
            raise ValueError(
                f"first_seen miss for {url!r}: eval mode refuses the "
                "now-fallback (future-knowledge leak); the map must cover "
                "every scored URL"
            )
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat(), True

    def _features(self, urls: Sequence[str]) -> np.ndarray:
        from phishnet.features.extraction import (  # type: ignore[import-untyped]
            featurise_frame,
        )

        frozen = [c for c in self.columns if c not in _enriched_column_set()]
        lex = featurise_frame(
            list(urls), frozen, canonicalize=self.canonicalize
        ).reset_index(drop=True)
        rows = []
        for u in urls:
            key, hosted = self._cache_key(u)
            first_seen, _ = self._first_seen(u)
            rows.append(
                _enriched_feature_row(
                    self._derive_row(u, self._table.get(key), first_seen, hosted)
                )
            )
        enr = pd.DataFrame(rows).reset_index(drop=True)
        frame = pd.concat(
            [lex, enr[[c for c in self.columns if c in enr.columns]]], axis=1
        )
        return frame[self.columns].to_numpy(dtype=float)

    def score(self, urls: Sequence[str]) -> list[float]:
        # Production mode counts its now-fallbacks; eval mode raises on
        # the first miss inside _first_seen instead of counting.
        if not self.eval_mode:
            self.n_now_fallback = len(urls)
        return super().score(urls)


def _enriched_column_set() -> set[str]:
    from phishnet.enrichment.features import (  # type: ignore[import-untyped]
        ENRICHED_COLUMNS,
    )

    return set(ENRICHED_COLUMNS)


def _enriched_feature_row(derived: dict[str, Any]) -> dict[str, float]:
    from phishnet.enrichment.features import (  # type: ignore[import-untyped]
        enriched_row_features,
    )

    return enriched_row_features(derived)
