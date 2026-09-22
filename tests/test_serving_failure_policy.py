"""Risk-graded fail-closed policy (T2-9, live-performance plan).

Registered behavior: any Tier-2 failure alerts. Graded behavior: a failure
alerts only when the Tier-1 score is at/above the floor, else `can't assess`.
"""

import pytest
from fastapi.testclient import TestClient

from phishnet.serving.app import create_app, predict_one
from phishnet.serving.cascade import (
    ALERT,
    ALLOW,
    CANT_ASSESS,
    Decision,
    Tier2Outcome,
    decide,
)

T_ALERT = 0.9269363298832987
LOWER = 0.6493076453312958


def _decide(
    score: float,
    outcome: Tier2Outcome | None,
    *,
    failure_floor: float | None = None,
    failure_policy: str = "graded",
) -> Decision:
    return decide(
        score,
        outcome,
        t_alert=T_ALERT,
        lower_edge=LOWER,
        failure_floor=failure_floor,
        failure_policy=failure_policy,
    )


def test_registered_failure_alerts_by_default() -> None:
    d = _decide(0.70, Tier2Outcome("failure", "unfetchable:HTTPError"))
    assert d.disposition == ALERT
    assert d.reason == "tier2_failure:unfetchable:HTTPError"


def test_graded_failure_below_floor_is_cant_assess() -> None:
    d = _decide(
        0.70, Tier2Outcome("failure", "unfetchable:HTTPError"), failure_floor=0.85
    )
    assert d.disposition == CANT_ASSESS
    assert d.reason == "tier2_failure_below_floor:unfetchable:HTTPError"


def test_graded_failure_at_or_above_floor_still_alerts() -> None:
    d = _decide(0.87, Tier2Outcome("failure", "timeout"), failure_floor=0.85)
    assert d.disposition == ALERT
    assert d.reason == "tier2_failure:timeout"


def test_graded_floor_does_not_touch_other_outcomes() -> None:
    assert (
        _decide(0.70, Tier2Outcome("benign"), failure_floor=0.85).disposition == ALLOW
    )
    assert (
        _decide(0.70, Tier2Outcome("phishing"), failure_floor=0.85).disposition == ALERT
    )
    # Out-of-band rows are decided before Tier-2 and ignore the floor.
    assert _decide(0.60, None, failure_floor=0.85).disposition == ALLOW


# --- wiring: predict_one and the app-level env ---------------------------


class FixedTier1:
    """Tier-1 stub with a settable score."""

    model_hash = "b" * 64
    columns_hash = "c" * 64
    columns = ["x"]
    thresholds = {"t_alert": T_ALERT, "lower_edge": LOWER, "t_1pct": 0.8780843789}
    thresholds_source = "test:deadbeef"

    def __init__(self, score: float) -> None:
        self._score = score

    def score_one(self, url: str) -> float:
        return self._score


class StubTier2:
    mode = "stub"

    def __init__(self, outcome: Tier2Outcome) -> None:
        self._outcome = outcome

    def judge(self, url: str) -> Tier2Outcome:
        return self._outcome


URL = "https://inband.example/login"


def test_predict_one_passes_the_failure_floor() -> None:
    body = predict_one(
        URL,
        tier1=FixedTier1(0.70),  # type: ignore[arg-type]
        resolver=None,
        tier2=StubTier2(Tier2Outcome("failure", "unfetchable:HTTPError")),
        t_alert=T_ALERT,
        lower_edge=LOWER,
        tier2_failure_floor=0.85,
        tier2_failure_policy="graded",
    )
    assert body["disposition"] == CANT_ASSESS
    assert body["reason"] == "tier2_failure_below_floor:unfetchable:HTTPError"
    assert body["tier2_failure_floor"] == 0.85


def test_predict_one_registered_floor_is_none() -> None:
    body = predict_one(
        URL,
        tier1=FixedTier1(0.70),  # type: ignore[arg-type]
        resolver=None,
        tier2=StubTier2(Tier2Outcome("failure", "unfetchable:HTTPError")),
        t_alert=T_ALERT,
        lower_edge=LOWER,
    )
    assert body["disposition"] == ALERT
    assert body["tier2_failure_floor"] is None


def test_env_failure_floor_surfaces_in_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_FAILURE_FLOOR", "0.85")
    app = create_app(servable=FixedTier1(0.70), tier2=None)  # type: ignore[arg-type]
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["tier2_failure_floor"] == 0.85


def test_env_failure_floor_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHISHNET_TIER2_FAILURE_FLOOR", raising=False)
    app = create_app(servable=FixedTier1(0.70), tier2=None)  # type: ignore[arg-type]
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["tier2_failure_floor"] is None


def test_predict_endpoint_applies_the_env_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_FAILURE_POLICY", "graded")
    monkeypatch.setenv("PHISHNET_TIER2_FAILURE_FLOOR", "0.85")
    app = create_app(
        servable=FixedTier1(0.70),  # type: ignore[arg-type]
        tier2=StubTier2(Tier2Outcome("failure", "unfetchable:ReadTimeout")),
    )
    with TestClient(app) as client:
        body = client.post("/predict", json={"url": URL}).json()
        assert body["disposition"] == CANT_ASSESS
        assert body["reason"] == "tier2_failure_below_floor:unfetchable:ReadTimeout"
        assert body["tier2_failure_floor"] == 0.85
