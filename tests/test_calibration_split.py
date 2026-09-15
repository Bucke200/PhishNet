"""Tests for domain-hash calibration carving and reliability rendering.

The carve is the load-bearing guarantee behind Step 5: registrable domains
stay wholly on one side (no campaign leakage into the calibration slice),
the mechanism is exactly the manifest's benign-split hash with a fresh
seed, and degenerate carves abort instead of fitting isotonic on noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import build_splits
import predictors
from eval import EvalConfig, _reliability_bar, evaluate, to_markdown
from ml_training.calibrate_gbm import carve_fit_calib
from ml_training.calibrate_gbm import main as calibrate_main

SEED = "phishnet-calib-test-v1"


class _CalibScorer:
    """Local probe scorer: importing test_eval would drag the untyped root
    module into mypy's dependency graph and break the gate."""

    name = "probe"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, urls: Sequence[str]) -> list[float]:
        return self._scores[: len(urls)]


def _train_frame(n_domains: int = 40) -> pd.DataFrame:
    """Deterministic fixture; labels assigned so every carve has both classes.

    Domains hash deterministically, so the calib set for SEED is fixed:
    label rows to guarantee both classes on each side regardless of it.
    """
    domains = [f"calib{i:03d}-probe.com" for i in range(n_domains)]
    calib = {d for d in domains if build_splits.neg_domain_is_test(d, SEED, 0.2)}
    rows = []
    for i, d in enumerate(domains):
        for j in range(3):
            # Both labels appear in both partitions by construction below.
            label = (i + j) % 2
            rows.append(
                {
                    "url": f"https://{d}/p{j}?i={i}",
                    "label": label,
                    "first_seen": "2026-06-05T00:00:00+00:00",
                    "registrable_domain": d,
                    "suffix": "com",
                    "source": "probe",
                }
            )
    # Force both classes into the calibration side however the hash falls.
    in_calib = [r for r in rows if r["registrable_domain"] in calib]
    in_calib[0]["label"] = 0
    in_calib[1]["label"] = 1
    return pd.DataFrame(rows)


def test_carve_partitions_without_overlap_and_covers_all() -> None:
    df = _train_frame()
    fit, calib = carve_fit_calib(df, SEED, 0.2)
    assert len(fit) + len(calib) == len(df)
    assert not (set(fit["registrable_domain"]) & set(calib["registrable_domain"]))
    assert set(fit["registrable_domain"]) | set(calib["registrable_domain"]) == set(
        df["registrable_domain"]
    )


def test_carve_is_deterministic() -> None:
    df = _train_frame()
    fit_a, calib_a = carve_fit_calib(df, SEED, 0.2)
    fit_b, calib_b = carve_fit_calib(df, SEED, 0.2)
    pd.testing.assert_frame_equal(fit_a, fit_b)
    pd.testing.assert_frame_equal(calib_a, calib_b)


def test_carve_reuses_manifest_hash_mechanism() -> None:
    """Same hash rule as the benign train/test partition, fresh seed."""
    df = _train_frame()
    _, calib = carve_fit_calib(df, SEED, 0.2)
    expected = {
        d
        for d in df["registrable_domain"].unique()
        if build_splits.neg_domain_is_test(d, SEED, 0.2)
    }
    assert set(calib["registrable_domain"]) == expected


def test_carve_rejects_train_test_seed_and_bad_fractions() -> None:
    df = _train_frame()
    with pytest.raises(ValueError):
        carve_fit_calib(df, build_splits.NEG_HASH_SEED_DEFAULT, 0.2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            carve_fit_calib(df, SEED, bad)


def test_reliability_bar_marks_both_points() -> None:
    bar = _reliability_bar(0.5, 0.5)
    assert len(bar) == 49
    assert bar.count("*") == 1  # coincident markers merge
    bar = _reliability_bar(0.0, 1.0)
    assert bar[0] == "x" and bar[-1] == "o"


def test_report_renders_one_diagram_row_per_bin(tmp_path: Path) -> None:
    urls = [f"https://rel{i:04d}.com/x" for i in range(60)]
    frame = pd.DataFrame(
        {
            "url": urls,
            "label": [0] * 40 + [1] * 20,
            "first_seen": ["2026-09-01T00:00:00+00:00"] * 60,
            "registrable_domain": [f"rel{i:04d}.com" for i in range(60)],
        }
    )
    dataset = tmp_path / "rel.csv"
    frame.to_csv(dataset, index=False)
    rep = evaluate(
        _CalibScorer([float(i) / 60 for i in range(60)]),
        dataset,
        EvalConfig(bootstrap=0, seed=0),
    )
    md = to_markdown(rep)
    assert "reliability (x = mean score, o = empirical rate):" in md
    rows = [ln for ln in md.splitlines() if "| bin " in ln and "| bin |" not in ln]
    assert len(rows) == len(rep["calibration"]["bins"])
    assert any("x" in r and "o" in r for r in rows)


class _StubProbaModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.tile(np.array([[0.3, 0.7]]), (X.shape[0], 1))


class _StubProba(predictors.LegacyEnsemble):
    """Dispatch probe: no assets, no extraction — locks the mode strings
    that take the predict_proba branch (a new calibrator method that misses
    the branch would silently emit hard labels)."""

    def __init__(self, mode: str) -> None:
        self.model = _StubProbaModel()
        self.mode = mode

    def _features(self, urls: Sequence[str]) -> np.ndarray:
        return np.zeros((len(urls), 2))


def test_proba_modes_dispatch_to_predict_proba() -> None:
    for mode in ("predict_proba", "isotonic_prefit", "sigmoid_prefit"):
        pred = _StubProba(mode)
        assert pred.score(["https://example.com/", "https://example.org/x"]) == [
            0.7,
            0.7,
        ]
        assert pred.model.calls == 1


def test_calibrate_main_writes_ranked_method_assets(tmp_path: Path) -> None:
    """End to end on a tiny fixture: both methods write the scaler-less
    layout (model + refit base + columns + sidecar) and record their
    method + threshold in the sidecar."""
    import json

    train = tmp_path / "train.csv"
    _train_frame(60).to_csv(train, index=False)
    frozen = (
        Path("src/phishnet/urlset_ml_assets/feature_columns.pkl")
        if Path("src/phishnet/urlset_ml_assets/feature_columns.pkl").exists()
        else Path("backend/urlset_ml_assets/feature_columns.pkl")
    )
    for method in ("isotonic", "sigmoid"):
        out = tmp_path / f"assets-{method}"
        rc = calibrate_main(
            [
                "--train",
                str(train),
                "--assets-out",
                str(out),
                "--frozen-columns",
                str(frozen),
                "--calib-seed",
                SEED,
                "--method",
                method,
            ]
        )
        assert rc == 0
        for name in (
            "calibrated_gbm.pkl",
            "refit_base.pkl",
            "feature_columns.pkl",
            "calibration-report.json",
        ):
            assert (out / name).is_file(), (method, name)
        assert not (out / "scaler.pkl").exists()
        sidecar = json.loads((out / "calibration-report.json").read_text())
        assert sidecar["calibration_method"] == method
        assert sidecar["fit_calib_domain_overlap"] == 0
        assert sidecar["test_csv_touched"] is False
        assert isinstance(sidecar["threshold"], float)


def test_fingerprint_covers_scaler_absence(tmp_path: Path) -> None:
    """The tmp calibrate run above is loadable without a scaler file, and
    its fingerprint records the absence as explicit null (a columns-hash
    alone would miss a representation change with unchanged vocabulary)."""
    import hashlib

    train = tmp_path / "train-fp.csv"
    _train_frame(60).to_csv(train, index=False)
    out = tmp_path / "assets-fp"
    assert (
        calibrate_main(
            [
                "--train",
                str(train),
                "--assets-out",
                str(out),
                "--calib-seed",
                SEED,
                "--method",
                "sigmoid",
            ]
        )
        == 0
    )
    pred = predictors.SigmoidGbm(assets_dir=str(out))
    assert pred.scaler is None
    assert pred.asset_fingerprint["scaler"] is None
    assert (
        pred.asset_fingerprint["model"]
        == hashlib.sha256((out / "calibrated_gbm.pkl").read_bytes()).hexdigest()
    )
    assert (
        pred.asset_fingerprint["columns"]
        == hashlib.sha256((out / "feature_columns.pkl").read_bytes()).hexdigest()
    )

    refit = predictors.GbmRefit(assets_dir=str(out))
    assert refit.name == "gbm_refit"
    assert refit.mode == "predict_proba"
    assert refit.scaler is None
    assert refit.asset_fingerprint["scaler"] is None
    # Scaler-less single and batch paths agree on real (tiny-fixture) assets.
    urls = ["https://example.com/login", "http://192.168.1.1/admin"]
    np.testing.assert_array_equal(
        refit._features(urls)[[0]], refit._features([urls[0]])
    )
    assert all(np.isfinite(refit.score(urls)))


def _fingerprinted_report(tmp_path: Path, stem: str) -> dict[str, Any]:
    """Real evaluate() report with a synthetic asset fingerprint attached
    (both sides need dicts: a missing fingerprint proves nothing)."""
    urls = [f"https://{stem}{i:04d}.com/x" for i in range(60)]
    frame = pd.DataFrame(
        {
            "url": urls,
            "label": [0] * 40 + [1] * 20,
            "first_seen": ["2026-09-01T00:00:00+00:00"] * 60,
            "registrable_domain": [f"{stem}{i:04d}.com" for i in range(60)],
        }
    )
    dataset = tmp_path / f"{stem}.csv"
    frame.to_csv(dataset, index=False)
    rep = evaluate(
        _CalibScorer([float(i) / 60 for i in range(60)]),
        dataset,
        EvalConfig(bootstrap=0, seed=0),
    )
    rep["predictor"] = "gbm_single"
    rep["asset_fingerprint"] = {"model": "m" * 64, "scaler": None, "columns": "c" * 64}
    return rep


def _sigmoid_fixture_assets(tmp_path: Path) -> Path:
    """One tiny sigmoid calibrate run; returns its asset dir (real fitted
    LGBM base + wrapper, loadable without CI-absent production assets)."""
    train = tmp_path / "train-shap.csv"
    _train_frame(60).to_csv(train, index=False)
    out = tmp_path / "assets-shap"
    assert (
        calibrate_main(
            [
                "--train",
                str(train),
                "--assets-out",
                str(out),
                "--calib-seed",
                SEED,
                "--method",
                "sigmoid",
            ]
        )
        == 0
    )
    return out


def test_explain_additivity_and_ordering(tmp_path: Path) -> None:
    """Contributions + bias reconstruct the served margin; top-k is ordered
    by |contribution|; the bias term is model-constant (expected value)."""
    out = _sigmoid_fixture_assets(tmp_path)
    refit = predictors.GbmRefit(assets_dir=str(out))
    urls = ["https://example.com/login", "https://google.com/"]
    full0 = refit.explain(urls[0], top_k=len(refit.columns))
    full1 = refit.explain(urls[1], top_k=len(refit.columns))
    for ex in (full0, full1):
        assert len(ex["features"]) == len(refit.columns)
        total = sum(v for _, v in ex["features"])
        assert ex["margin"] == pytest.approx(total + ex["bias"], rel=1e-9)
    ex0 = refit.explain(urls[0], top_k=5)
    ex1 = refit.explain(urls[1], top_k=5)
    for ex in (ex0, ex1):
        assert len(ex["features"]) == 5
        magnitudes = [abs(v) for _, v in ex["features"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
    # Bias is the expected value: constant across inputs.
    assert ex0["bias"] == pytest.approx(ex1["bias"], rel=1e-9)
    # Margin is the logit of the served probability.
    proba = refit.score([urls[0]])[0]
    assert proba == pytest.approx(1.0 / (1.0 + math.exp(-ex0["margin"])), rel=1e-9)


def test_explain_wrapper_reports_base_margin(tmp_path: Path) -> None:
    """Calibrated wrappers attribute in margin space: same underlying base,
    same margin — the monotonic map preserves order, not levels."""
    out = _sigmoid_fixture_assets(tmp_path)
    base_margin = predictors.GbmRefit(assets_dir=str(out)).explain(
        "https://example.com/login"
    )["margin"]
    wrapped_margin = predictors.SigmoidGbm(assets_dir=str(out)).explain(
        "https://example.com/login"
    )["margin"]
    assert wrapped_margin == pytest.approx(base_margin, rel=1e-12)


def test_explain_unsupported_and_bad_top_k(tmp_path: Path) -> None:
    """Voting ensembles have no exact attribution (loud, not silent); top_k
    is validated before any model is touched."""
    legacy = predictors.LegacyEnsemble()
    with pytest.raises(predictors.ExplainUnsupportedError):
        legacy.explain("https://example.com/")
    out = _sigmoid_fixture_assets(tmp_path)
    with pytest.raises(ValueError):
        predictors.GbmRefit(assets_dir=str(out)).explain(
            "https://example.com/", top_k=0
        )


def test_same_name_different_assets_flagged(tmp_path: Path) -> None:
    """Same predictor name + same dataset, different asset bytes: the delta
    passes but never unattributed — including the scaler added/removed
    cause, which a columns-hash alone would miss."""
    run = _fingerprinted_report(tmp_path, "fp-a")
    assert "different assets" not in to_markdown(
        run, {**run, "asset_fingerprint": dict(run["asset_fingerprint"])}
    )

    changed_fp = {**run["asset_fingerprint"], "model": "n" * 64}
    changed = {**run, "asset_fingerprint": changed_fp}
    md_changed = to_markdown(run, changed)
    assert "different assets" in md_changed
    assert "mix retraining/representation changes" in md_changed

    scaler_back = {
        **run,
        "asset_fingerprint": {**run["asset_fingerprint"], "scaler": "s" * 64},
    }
    assert "scaler added/removed" in to_markdown(run, scaler_back)

    missing = {**run}
    del missing["asset_fingerprint"]
    assert "different assets" not in to_markdown(run, missing)
