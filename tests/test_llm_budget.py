"""Spend-guard tests: ledger, cap, STOP, lock, and `judge` wiring.

All tests point `PHISHNET_LLM_BUDGET_DIR` at a tmp path, so nothing here
touches the repo's real `.budget/` ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import phishnet.llm.client as client
from phishnet.llm import budget


def _configure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: str | None
) -> None:
    monkeypatch.setenv("PHISHNET_LLM_BUDGET_DIR", str(tmp_path))
    monkeypatch.setenv("PHISHNET_LLM_BUDGET_ID", "test")
    if cap is None:
        monkeypatch.delenv("PHISHNET_LLM_BUDGET_USD", raising=False)
    else:
        monkeypatch.setenv("PHISHNET_LLM_BUDGET_USD", cap)


def test_ledger_tracks_usage_without_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path, None)
    reservation = budget.reserve("x" * 400)
    assert reservation.prompt_tokens == 100
    budget.settle(reservation, {"prompt_tokens": 120, "completion_tokens": 8})
    status = budget.status()
    assert status["calls"] == 1
    assert status["prompt_tokens"] == 120
    assert status["completion_tokens"] == 8
    est = status["est_usd"]
    assert isinstance(est, (int, float))
    assert float(est) == pytest.approx(budget.estimate_usd(120, 8), rel=1e-9)


def test_cap_refuses_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path, "0.000001")
    with pytest.raises(budget.BudgetExceeded):
        budget.reserve("x" * 4000)
    assert budget.status()["calls"] == 0


def test_stop_sentinel_halts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, None)
    budget.stop_path().parent.mkdir(parents=True, exist_ok=True)
    budget.stop_path().write_text("", encoding="utf-8")
    with pytest.raises(budget.BudgetStopped):
        budget.reserve("x" * 40)


def test_failed_call_keeps_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path, None)
    reservation = budget.reserve("x" * 400)
    budget.settle(reservation, {})
    status = budget.status()
    assert status["failed_calls"] == 1
    est = status["est_usd"]
    assert isinstance(est, (int, float))
    assert float(est) == pytest.approx(reservation.usd, rel=1e-9)


def test_lock_is_exclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, None)
    budget.acquire_lock("t")
    with pytest.raises(budget.BudgetLocked):
        budget.acquire_lock("t")
    budget.release_lock("t")
    budget.acquire_lock("t")
    budget.release_lock("t")


def test_reset_zeroes_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, None)
    reservation = budget.reserve("x" * 400)
    budget.settle(reservation, {"prompt_tokens": 10, "completion_tokens": 1})
    budget.reset()
    status = budget.status()
    assert status["calls"] == 0
    est = status["est_usd"]
    assert isinstance(est, (int, float))
    assert float(est) == 0.0


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def json(self) -> dict[str, object]:
        return {
            "system_fingerprint": "fp_test",
            "usage": {"prompt_tokens": 111, "completion_tokens": 7},
            "choices": [{"message": {"content": json.dumps({"verdict": "benign"})}}],
        }


def _patch_post(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse()

    monkeypatch.setattr(client.requests, "post", fake_post)
    return calls


def test_judge_records_actual_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path, None)
    _patch_post(monkeypatch)
    _, judgment = client.judge("k", "example.test", "title: Example")
    assert judgment.ok
    status = budget.status()
    assert status["calls"] == 1
    assert status["prompt_tokens"] == 111
    assert status["completion_tokens"] == 7


def test_judge_refuses_and_sends_nothing_when_capped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path, "0.000001")
    calls = _patch_post(monkeypatch)
    with pytest.raises(budget.BudgetExceeded):
        client.judge("k", "example.test", "title: Example")
    assert calls["n"] == 0
    assert budget.status()["calls"] == 0
