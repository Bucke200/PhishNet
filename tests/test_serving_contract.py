"""Contract tests for the Phase 6 serving app (`/health`, `/predict`, `/explain`).

No network and no legacy assets: a real `Tier1Servable` is injected, and the
shortener resolver is a no-op, so `/predict` scores the URL as given. What is
pinned here is the response contract (disposition vocabulary, raw Tier-1
score, model/threshold provenance), the retired explain path, and the
removal of `/report`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from phishnet.serving import Tier1Servable
from phishnet.serving.app import create_app
from phishnet.serving.shortener import Resolution

URL = "https://example.com/login"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    servable = Tier1Servable()
    app = create_app(servable=servable, resolver=lambda u: Resolution(u, 0, ""))
    with TestClient(app) as c:
        yield c


def test_health_reports_assets_and_thresholds(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["columns_hash"]) == 64
    assert body["n_columns"] == 79
    assert body["thresholds"]["t_alert"] == pytest.approx(0.9269363298832987)
    assert body["tier2_mode"] == "disabled"


def test_predict_contract(client: TestClient) -> None:
    body = client.post("/predict", json={"url": URL}).json()
    assert body["url"] == URL
    assert body["disposition"] in {"allow", "alert", "can't assess"}
    assert len(body["model_hash"]) == 64
    assert body["thresholds_source"].startswith("phase4.json:")
    if body["disposition"] == "alert":
        assert body["score"] == pytest.approx(body["tier1_score"])
    if body["disposition"] == "allow":
        assert body["score"] == pytest.approx(body["tier1_score"])


def test_predict_tier1_score_matches_serving_path(client: TestClient) -> None:
    servable = Tier1Servable()
    expected = servable.score_one(URL)
    body = client.post("/predict", json={"url": URL}).json()
    assert body["tier1_score"] == pytest.approx(expected, abs=0.0)


def test_explain_returns_attribution(client: TestClient) -> None:
    body = client.post("/explain", json={"url": URL, "top_k": 3}).json()
    features = body["attribution"]["features"]
    assert len(features) == 3
    assert all(set(f) == {"feature", "contribution"} for f in features)


def test_explain_rejects_nonpositive_top_k(client: TestClient) -> None:
    assert client.post("/explain", json={"url": URL, "top_k": 0}).status_code == 422


def test_predict_rejects_invalid_url(client: TestClient) -> None:
    assert client.post("/predict", json={"url": "not a url"}).status_code == 422


def test_report_endpoint_is_removed(client: TestClient) -> None:
    response = client.post("/report", json={"url": URL, "reported_label": 1})
    assert response.status_code == 404
