"""Mechanism-aware failure policy (2026-09-22).

Active rejection (WAF/bot-wall) and dead links are different signals; the
cascade keys the disposition on the fetch mechanism rather than one blunt
threshold. Non-fetch (LLM) failures stay fail-closed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishnet.serving.app import create_app
from phishnet.serving.cascade import (
    ALERT,
    CANT_ASSESS,
    Decision,
    Tier2Outcome,
    decide,
)
from phishnet.serving.tier2 import LiveTier2Provider

T_ALERT = 0.9269363298832987
LOWER = 0.6493076453312958


def _d(score: float, reason: str) -> Decision:
    return decide(
        score,
        Tier2Outcome("failure", reason),
        t_alert=T_ALERT,
        lower_edge=LOWER,
        failure_policy="mechanism",
    )


def test_active_rejection_alerts_across_the_band() -> None:
    for score in (0.65, 0.75, 0.90):
        assert _d(score, "http_403").disposition == ALERT
        assert _d(score, "blocked").disposition == ALERT


def test_dns_refused_tls_threshold_is_070() -> None:
    for reason in ("dns", "refused", "tls"):
        assert _d(0.70, reason).disposition == ALERT
        assert _d(0.699, reason).disposition == CANT_ASSESS


def test_origin_timeout_threshold_is_080() -> None:
    assert _d(0.80, "origin_timeout").disposition == ALERT
    assert _d(0.799, "origin_timeout").disposition == CANT_ASSESS


def test_auth_gateway_threshold_is_080() -> None:
    # Legit enterprise portals sit at 0.65-0.75 and are not alerted on; the
    # signal is spoofable, so a high-scoring fake gate still alerts.
    assert _d(0.80, "auth_gateway").disposition == ALERT
    assert _d(0.75, "auth_gateway").disposition == CANT_ASSESS


def test_404_never_alerts() -> None:
    assert _d(0.90, "http_404").disposition == CANT_ASSESS


def test_fetcher_internal_failures_never_alert() -> None:
    for reason in ("fetcher_timeout", "fetcher_error", "fetcher_http"):
        assert _d(0.90, reason).disposition == CANT_ASSESS


def test_llm_failures_stay_fail_closed() -> None:
    for reason in ("schema", "refusal", "api_error", "http=400"):
        assert _d(0.66, reason).disposition == ALERT


def test_unknown_mechanism_is_cant_assess() -> None:
    assert _d(0.90, "wat").disposition == CANT_ASSESS


def test_closed_policy_is_unchanged() -> None:
    d = decide(0.66, Tier2Outcome("failure", "dns"), t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALERT


# --- provider parsing of the structured fetcher contract -----------------


class _Resp:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


def _provider(monkeypatch: pytest.MonkeyPatch, payload: object) -> LiveTier2Provider:
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(payload))
    return LiveTier2Provider("http://fetcher:8100/fetch", "test-key")


def test_structured_target_failure_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        monkeypatch,
        {
            "ok": False,
            "error": "http_403",
            "status_code": 403,
            "trigger_type": "status",
            "trigger_match": "403",
            "detail": "x",
        },
    )
    outcome = provider.judge("https://inband.example/")
    assert outcome is not None
    assert outcome.kind == "failure"
    assert outcome.reason == "http_403"
    assert outcome.trigger_type == "status"
    assert outcome.trigger_match == "403"


def test_log_decision_emits_trigger_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import json
    import logging

    from phishnet.serving.app import _log_decision

    payload = {
        "disposition": "alert",
        "reason": "tier2_failure:blocked",
        "tier1_score": 0.742,
        "in_band": True,
        "url": "https://example-app.pages.dev/login",
        "scored_url": "https://example-app.pages.dev/login",
        "tier2": {
            "kind": "failure",
            "reason": "blocked",
            "trigger_type": "html_token",
            "trigger_match": "__cf_chl",
        },
    }
    with caplog.at_level(logging.INFO, logger="phishnet.serving"):
        _log_decision(payload)
    record = json.loads(caplog.records[-1].message)
    assert record["outcome"] == "alert"
    assert record["tier1_score"] == 0.742
    assert record["trigger_type"] == "html_token"
    assert record["trigger_match"] == "__cf_chl"
    assert record["host"] == "example-app.pages.dev"


def test_rpc_timeout_maps_to_fetcher_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    def _raise(*a: object, **k: object) -> object:
        raise requests.exceptions.ReadTimeout()

    monkeypatch.setattr(requests, "post", _raise)
    provider = LiveTier2Provider("http://fetcher:8100/fetch", "test-key")
    outcome = provider.judge("https://inband.example/")
    assert outcome is not None
    assert outcome.reason == "fetcher_timeout"


def test_non_json_body_maps_to_fetcher_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, None)
    outcome = provider.judge("https://inband.example/")
    assert outcome is not None
    assert outcome.reason == "fetcher_http"


def test_env_failure_policy_surfaces_in_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_FAILURE_POLICY", "mechanism")

    class _T1:
        model_hash = "a" * 64
        columns_hash = "b" * 64
        columns = ["x"]
        thresholds = {"t_alert": T_ALERT, "lower_edge": LOWER, "t_1pct": 0.87}
        thresholds_source = "test:deadbeef"

    app = create_app(servable=_T1(), tier2=None)  # type: ignore[arg-type]
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["tier2_failure_policy"] == "mechanism"
