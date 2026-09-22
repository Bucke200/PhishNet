"""Offline Phase 4 sweep tests: repeats, URL-scoped cache, seal validity.

Every test stubs the model call, so nothing here touches Groq or spends
money. What is asserted is exactly what a paid run must not get wrong:

- three repeats produce three real judgments per row (no cache collision),
- distinct URLs sharing an identical extract do not share a cache key,
- coverage counts only 200/400 seals (429/transport/5xx never count),
- transient failures are retried inside the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "scripts")

import p4_sweep as P  # noqa: E402

from phishnet.llm.client import Judgment  # noqa: E402

N_ROWS = 4


def _judgment(status: int, verdict: str | None = "benign") -> Judgment:
    ok = status == 200
    return Judgment(
        ok=ok,
        parsed={"verdict": verdict} if ok else None,
        fingerprint=f"fp_{status}",
        usage={"prompt_tokens": 10, "completion_tokens": 2} if ok else {},
        latency_ms=1.0,
        status=status,
        error="" if ok else f"http={status}",
    )


def _population() -> tuple[pd.DataFrame, float, float]:
    # Rows 0 and 1 deliberately share an extract: the old extract-only cache
    # key collided them; the URL-scoped key must keep them distinct.
    shared = {"page_host": "shared.test", "title": "same page"}
    rows = [
        {
            "url": f"https://row{i}.test/",
            "label": i % 2,
            "extract": shared
            if i < 2
            else {"page_host": f"h{i}.test", "title": f"t{i}"},
        }
        for i in range(N_ROWS)
    ]
    return pd.DataFrame(rows), 0.9, 0.6


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    monkeypatch.setattr(P, "RUNS", runs)
    monkeypatch.setattr(P, "CACHE", runs / "cache")
    monkeypatch.setattr(P, "population", _population)
    monkeypatch.setattr(P, "load_key", lambda: "test-key")
    monkeypatch.setattr(P.time, "sleep", lambda _s: None)
    monkeypatch.setenv("PHISHNET_LLM_BUDGET_DIR", str(tmp_path / "budget"))
    monkeypatch.setenv("PHISHNET_LLM_BUDGET_ID", "test")
    monkeypatch.delenv("PHISHNET_LLM_BUDGET_USD", raising=False)
    return runs


def _run(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, tuple[int, int]],
    run_id: str = "p4-test",
    repeats: int = 3,
) -> int:
    """Stub the call layer, emulating the in-call retry for transient states."""
    calls: dict[str, int] = {}

    def fake_backoff(
        key: str, extract: dict[str, object]
    ) -> tuple[dict[str, object], Judgment, int]:
        host = str(extract.get("page_host", ""))
        first, later = statuses.get(host, (200, 200))
        seen = calls.get(host, 0)
        calls[host] = seen + 1
        status = first if seen == 0 else later
        if status == 429 or status < 0 or status >= 500:
            status = later
        return {}, _judgment(status), 2

    monkeypatch.setattr(P, "judge_with_backoff", fake_backoff)
    return int(
        P.main(
            [
                "--sweep",
                "--run-id",
                run_id,
                "--repeats",
                str(repeats),
                "--no-lock",
                "--allow-population-drift",
            ]
        )
    )


def test_three_repeats_are_real_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _prepare(monkeypatch, tmp_path)
    assert _run(monkeypatch, {}) == 0

    meta = json.loads((runs / "p4-test" / "run.json").read_text(encoding="utf-8"))
    assert meta["run_class"] == "recorded"
    assert meta["n_repeats"] == 3
    assert meta["full_coverage"] is True
    assert meta["repeat_coverage"] == {"0": N_ROWS, "1": N_ROWS, "2": N_ROWS}

    records = [
        json.loads(line)
        for line in (runs / "p4-test" / "judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == N_ROWS * 3
    # URL-scoped keys: the two rows sharing an extract got distinct keys.
    assert len({r["cache_key"] for r in records}) == N_ROWS * 3
    assert len(list((runs / "cache").glob("*.json"))) == N_ROWS * 3
    for repeat in range(3):
        assert len({r["url"] for r in records if r["repeat_idx"] == repeat}) == N_ROWS


def test_persistent_429_never_counts_as_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _prepare(monkeypatch, tmp_path)
    assert _run(monkeypatch, {"h2.test": (429, 429)}) == 0

    meta = json.loads((runs / "p4-test" / "run.json").read_text(encoding="utf-8"))
    assert meta["run_class"] == "provisional"
    assert meta["full_coverage"] is False
    assert meta["repeat_coverage"] == {
        "0": N_ROWS - 1,
        "1": N_ROWS - 1,
        "2": N_ROWS - 1,
    }


def test_transient_5xx_then_200_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _prepare(monkeypatch, tmp_path)
    assert _run(monkeypatch, {"h2.test": (500, 200)}) == 0
    meta = json.loads((runs / "p4-test" / "run.json").read_text(encoding="utf-8"))
    assert meta["run_class"] == "recorded"


def test_valid_seal_statuses() -> None:
    assert P.is_valid_seal({"status": 200})
    assert P.is_valid_seal({"status": 400})
    assert not P.is_valid_seal({"status": 429})
    assert not P.is_valid_seal({"status": -1})
    assert not P.is_valid_seal({"status": 503})


def test_backoff_retries_transient_then_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = iter([_judgment(503), _judgment(200)])

    def fake_judge(
        key: str, page_host: str, text: str, prompt_version: str = "p4-v1"
    ) -> tuple[dict[str, object], Judgment]:
        return {}, next(seq)

    monkeypatch.setattr(P, "judge", fake_judge)
    monkeypatch.setattr(P.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "phishnet.snapshot.extract.to_model_text", lambda _extract: "text"
    )
    _, judgment, attempts = P.judge_with_backoff("k", {"page_host": "h.test"})
    assert judgment.status == 200
    assert attempts == 2


def test_backoff_caps_persistent_429_at_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_judge(
        key: str, page_host: str, text: str, prompt_version: str = "p4-v1"
    ) -> tuple[dict[str, object], Judgment]:
        return {}, _judgment(429)

    monkeypatch.setattr(P, "judge", fake_judge)
    monkeypatch.setattr(P.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "phishnet.snapshot.extract.to_model_text", lambda _extract: "text"
    )
    _, judgment, attempts = P.judge_with_backoff("k", {"page_host": "h.test"})
    assert judgment.status == 429
    assert attempts == 3
