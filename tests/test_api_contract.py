"""Contract tests for POST /predict's explain flag.

No server, no model assets, no Mongo: the ensemble and scaler are stubbed
at module level and ``predict_url`` is awaited directly. What is pinned
here is the contract — default response shape, the 501 while the serving
model exposes no native tree SHAP, request validation, and the success
branch of the shared helper against a pred_contrib-capable stub.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import phishnet.api as api


class _StubEnsemble:
    """Hard-vote-shaped stand-in: labels only, no predict_proba, no SHAP."""

    def predict(self, X: Any) -> Any:
        return np.array([1])


class _IdentityScaler:
    def transform(self, X: Any) -> Any:
        return np.asarray(X, dtype=float)


class _StubLgbm:
    """pred_contrib-capable stand-in: fixed (1, n+1) contributions."""

    def __init__(self, n_features: int) -> None:
        row = np.zeros(n_features + 1)
        row[0] = 0.9
        row[1] = -0.4
        row[-1] = 0.2
        self._row = row

    def predict(self, X: Any, pred_contrib: bool = False) -> Any:
        assert pred_contrib
        return np.tile(self._row, (np.asarray(X).shape[0], 1))


COLUMNS = ["url_length", "entropy"]


def _install_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "urlset_model", _StubEnsemble())
    monkeypatch.setattr(api, "urlset_scaler", _IdentityScaler())
    monkeypatch.setattr(api, "urlset_feature_columns", list(COLUMNS))


def _http_request() -> Any:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(mongodb=None)))


def _make_request(**kwargs: Any) -> api.URLRequest:
    """model_validate keeps mypy quiet: HttpUrl accepts strings at runtime
    but its constructor type does not."""
    return api.URLRequest.model_validate({"url": "https://example.com/login", **kwargs})


def test_default_response_names_model_and_has_no_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ensemble(monkeypatch)
    rep = asyncio.run(api.predict_url(_make_request(), _http_request()))
    assert rep["prediction"] == 1
    assert rep["model"] == "urlset_ensemble"
    assert "attribution" not in rep


def test_explain_on_ensemble_is_501_not_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ensemble(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            api.predict_url(
                _make_request(explain=True),
                _http_request(),
            )
        )
    assert exc.value.status_code == 501


def test_top_k_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        _make_request(top_k=0)


def test_explain_helper_orders_and_separates_bias() -> None:
    out = api.explain_prediction(_StubLgbm(len(COLUMNS)), np.zeros((1, 2)), COLUMNS, 10)
    assert [f["feature"] for f in out["features"]] == ["url_length", "entropy"]
    assert out["features"][0]["contribution"] == pytest.approx(0.9)
    assert out["bias"] == pytest.approx(0.2)
    limited = api.explain_prediction(
        _StubLgbm(len(COLUMNS)), np.zeros((1, 2)), COLUMNS, 1
    )
    assert [f["feature"] for f in limited["features"]] == ["url_length"]
