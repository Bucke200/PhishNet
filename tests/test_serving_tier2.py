"""Tier-2 providers and end-to-end fail-closed wiring (C3/C6).

The sealed provider replays Phase 5 verdicts offline; the live provider is
opt-in and not exercised here. The wiring test uses a stub Tier-1 whose
score lands in band, so a failing Tier-2 must alert and a phishing verdict
must pin to the anchor.
"""

from __future__ import annotations

import pytest

from phishnet.serving.app import predict_one
from phishnet.serving.cascade import ALERT, Tier2Outcome
from phishnet.serving.tier2 import SealedTier2Provider, provider_from_env

T_ALERT = 0.9269363298832987
LOWER = 0.6493076453312958
IN_BAND = (LOWER + T_ALERT) / 2
URL = "https://inband.example/login"


class StubTier1:
    model_hash = "a" * 64
    thresholds_source = "test:deadbeef"

    def score_one(self, url: str) -> float:
        return IN_BAND


class StubTier2:
    def __init__(self, outcome: Tier2Outcome | None) -> None:
        self.mode = "stub"
        self._outcome = outcome

    def judge(self, url: str) -> Tier2Outcome | None:
        return self._outcome


def _run(tier2: StubTier2 | None):  # type: ignore[no-untyped-def]
    return predict_one(
        URL,
        tier1=StubTier1(),  # type: ignore[arg-type]
        resolver=None,
        tier2=tier2,
        t_alert=T_ALERT,
        lower_edge=LOWER,
    )


def test_in_band_failure_alerts_through_the_pipeline() -> None:
    body = _run(StubTier2(Tier2Outcome("failure", "schema")))
    assert body["disposition"] == ALERT
    assert body["score"] is not None


def test_in_band_phishing_pins_anchor_through_the_pipeline() -> None:
    body = _run(StubTier2(Tier2Outcome("phishing")))
    assert body["disposition"] == ALERT
    assert body["score"] > T_ALERT


def test_in_band_benign_allows_through_the_pipeline() -> None:
    body = _run(StubTier2(Tier2Outcome("benign")))
    assert body["disposition"] == "allow"


def test_in_band_without_provider_cannot_assess() -> None:
    body = _run(None)
    assert body["disposition"] == "can't assess"
    assert body["score"] is None


def test_sealed_provider_replays_known_pages() -> None:
    import json
    from pathlib import Path

    provider = SealedTier2Provider()
    manifest = json.loads(
        Path("reports/adversarial-manifest-p5.json").read_text(encoding="utf-8")
    )
    replayed = []
    for row in manifest:
        outcome = provider.judge(row["url"])
        if outcome is not None:
            replayed.append(outcome)
    assert replayed, "no manifest URL matched a sealed p5-h1 verdict"
    assert all(
        o.kind in {"phishing", "benign", "suspicious", "failure"} for o in replayed
    )


def test_sealed_provider_escalates_detector_hits() -> None:
    import json
    from pathlib import Path

    provider = SealedTier2Provider()
    manifest = json.loads(
        Path("reports/adversarial-manifest-p5.json").read_text(encoding="utf-8")
    )
    hit = next(
        r
        for r in manifest
        if r.get("detector_hit") and provider.judge(r["url"]) is not None
    )
    outcome = provider.judge(hit["url"])
    assert outcome is not None
    assert outcome.kind == "phishing"
    assert outcome.reason == "detector"


def test_sealed_provider_unknown_url_is_none() -> None:
    assert SealedTier2Provider().judge("https://not-in-the-demo-set.example/") is None


def test_provider_from_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_MODE", "disabled")
    assert provider_from_env() is None
