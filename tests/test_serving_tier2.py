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
        self.calls = 0

    def judge(self, url: str) -> Tier2Outcome | None:
        self.calls += 1
        return self._outcome


class FixedTier1:
    """Tier-1 stub with a settable score (for out-of-band checks)."""

    model_hash = "b" * 64
    thresholds_source = "test:deadbeef"

    def __init__(self, score: float) -> None:
        self._score = score

    def score_one(self, url: str) -> float:
        return self._score


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
    assert body["reason"] == "tier2_not_configured"


def test_in_band_provider_without_a_verdict_is_labeled() -> None:
    """A configured provider missing this URL is not a missing config."""
    body = _run(StubTier2(None))
    assert body["disposition"] == "can't assess"
    assert body["reason"] == "tier2_no_verdict"


def test_tier2_floor_extends_the_llm_band() -> None:
    """A floor below the registered edge lets Tier 2 review lower scores."""
    provider = StubTier2(Tier2Outcome("phishing"))
    body = predict_one(
        URL,
        tier1=FixedTier1(0.4),  # type: ignore[arg-type]  # below lower_edge
        resolver=None,
        tier2=provider,
        t_alert=T_ALERT,
        lower_edge=LOWER,
        tier2_floor=0.3,
    )
    assert provider.calls == 1
    assert body["disposition"] == "alert"
    assert body["tier2_floor"] == 0.3


def test_default_floor_is_the_registered_edge() -> None:
    provider = StubTier2(Tier2Outcome("phishing"))
    body = predict_one(
        URL,
        tier1=FixedTier1(0.4),  # type: ignore[arg-type]
        resolver=None,
        tier2=provider,
        t_alert=T_ALERT,
        lower_edge=LOWER,
    )
    assert provider.calls == 0
    assert body["disposition"] == "allow"
    assert body["tier2_floor"] == LOWER


@pytest.mark.parametrize("score", [0.1, T_ALERT])
def test_out_of_band_rows_never_call_tier2(score: float) -> None:
    provider = StubTier2(Tier2Outcome("phishing"))
    body = predict_one(
        URL,
        tier1=FixedTier1(score),  # type: ignore[arg-type]
        resolver=None,
        tier2=provider,
        t_alert=T_ALERT,
        lower_edge=LOWER,
    )
    assert provider.calls == 0
    assert body["tier2"] is None
    assert body["disposition"] == ("alert" if score >= T_ALERT else "allow")


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


def test_sealed_provider_is_scheme_insensitive() -> None:
    """A browser upgrading http→https must still find the sealed verdict."""
    import json
    from pathlib import Path

    provider = SealedTier2Provider()
    manifest = json.loads(
        Path("reports/adversarial-manifest-p5.json").read_text(encoding="utf-8")
    )
    http_url = next(
        r["url"]
        for r in manifest
        if r["url"].startswith("http://") and provider.judge(r["url"]) is not None
    )
    https_url = "https://" + http_url[len("http://") :]
    assert provider.judge(https_url) == provider.judge(http_url)


def test_sealed_provider_unknown_url_is_none() -> None:
    assert SealedTier2Provider().judge("https://not-in-the-demo-set.example/") is None


def test_provider_from_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_MODE", "disabled")
    assert provider_from_env() is None


def test_provider_from_env_live_requires_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live mode must fail loud, not silently disable the LLM layer."""
    monkeypatch.setenv("PHISHNET_TIER2_MODE", "live")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PHISHNET_FETCHER_URL", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        provider_from_env()


def test_provider_from_env_live_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHISHNET_TIER2_MODE", "live")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("PHISHNET_FETCHER_URL", "http://fetcher:8100/fetch")
    provider = provider_from_env()
    assert provider is not None
    assert provider.mode == "live"
