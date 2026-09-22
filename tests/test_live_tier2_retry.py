"""Bounded transient retry for the live Tier-2 provider (T2-7).

One transport failure or provider 5xx must not turn an in-band page into a
failure-alert; a 429 is returned immediately (the registered backoff is 60 s,
longer than a serving request should wait).
"""

from __future__ import annotations

from typing import Any

import pytest

import phishnet.serving.tier2 as tier2
from phishnet.llm.budget import BudgetExceeded
from phishnet.llm.client import Judgment
from phishnet.serving.cascade import Tier2Outcome
from phishnet.serving.tier2 import LiveTier2Provider, _judge_with_retry


def _judgment(status: int) -> Judgment:
    return Judgment(
        ok=status == 200,
        parsed=None,
        fingerprint=None,
        usage={},
        latency_ms=1.0,
        status=status,
        error="" if status == 200 else f"http={status}",
    )


def _fake_judge(statuses: list[int]) -> tuple[Any, list[int]]:
    calls: list[int] = []

    def judge_fake(
        api_key: str,
        page_host: str,
        extract_text: str,
        prompt_version: str = "p4-v1",
        timeout: int = 120,
    ) -> tuple[dict[str, Any], Judgment]:
        status = statuses[len(calls)]
        calls.append(status)
        return {}, _judgment(status)

    return judge_fake, calls


def test_transient_5xx_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    judge_fake, calls = _fake_judge([503, 200])
    monkeypatch.setattr(tier2, "judge", judge_fake)
    monkeypatch.setattr(tier2.time, "sleep", lambda _s: None)
    judgment = _judge_with_retry("k", "host", "text", "p6-v1")
    assert judgment.status == 200
    assert calls == [503, 200]


def test_transport_error_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    judge_fake, calls = _fake_judge([-1, 200])
    monkeypatch.setattr(tier2, "judge", judge_fake)
    monkeypatch.setattr(tier2.time, "sleep", lambda _s: None)
    judgment = _judge_with_retry("k", "host", "text", "p6-v1")
    assert judgment.status == 200
    assert calls == [-1, 200]


def test_429_is_returned_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    judge_fake, calls = _fake_judge([429, 200])
    monkeypatch.setattr(tier2, "judge", judge_fake)
    monkeypatch.setattr(tier2.time, "sleep", lambda _s: None)
    judgment = _judge_with_retry("k", "host", "text", "p6-v1")
    assert judgment.status == 429
    assert calls == [429]


def test_success_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    judge_fake, calls = _fake_judge([200, 200])
    monkeypatch.setattr(tier2, "judge", judge_fake)
    monkeypatch.setattr(tier2.time, "sleep", lambda _s: None)
    judgment = _judge_with_retry("k", "host", "text", "p6-v1")
    assert judgment.status == 200
    assert calls == [200]


def test_persistent_5xx_gives_up_after_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge_fake, calls = _fake_judge([500, 500, 500])
    monkeypatch.setattr(tier2, "judge", judge_fake)
    monkeypatch.setattr(tier2.time, "sleep", lambda _s: None)
    judgment = _judge_with_retry("k", "host", "text", "p6-v1")
    assert judgment.status == 500
    assert calls == [500, 500]


class _FakeFetchResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True, "extract": {"visible_text": "hello world"}}


def test_budget_refusal_maps_to_failure_not_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tripped spend guard must fail closed, never 500 the endpoint."""
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeFetchResponse())

    def raise_budget(*a: object, **k: object) -> Judgment:
        raise BudgetExceeded("next call would cross the cap")

    monkeypatch.setattr(tier2, "_judge_with_retry", raise_budget)
    provider = LiveTier2Provider("http://fetcher:8100/fetch", "test-key")
    outcome = provider.judge("https://inband.example/login")
    assert isinstance(outcome, Tier2Outcome)
    assert outcome.kind == "failure"
    assert outcome.reason == "budget:BudgetExceeded"
