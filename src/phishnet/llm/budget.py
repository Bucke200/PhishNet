"""LLM spend guard: cumulative ledger, optional cap, kill switch, run lock.

Two layers protect the card:

1. the provider-side spend limit (console) is the hard backstop;
2. this module stops a run *before* it crosses a local budget.

`phishnet.llm.client.judge` reserves estimated tokens before every request
and reconciles the reservation with the provider's reported `usage` after.
The ledger is cumulative on disk, so restarting a driver (or an agent
relaunching it) cannot reset the counter — that is the failure mode a
per-process `--max-calls` flag does not cover.

Usage is always tracked; a cap applies only when `PHISHNET_LLM_BUDGET_USD`
is set, so Tier-2 serving keeps running (and reporting its spend) without a
local cap unless the operator sets one.

Environment:
  PHISHNET_LLM_BUDGET_USD   cap in USD for this budget id (unset = track only)
  PHISHNET_LLM_BUDGET_ID    ledger namespace (default "default")
  PHISHNET_LLM_BUDGET_DIR   ledger/lock directory (default ".budget")

Kill switch: create `<budget dir>/STOP`; the next reserve raises
`BudgetStopped` before any request is sent. The response cache makes a halt
free to resume.

Prices are Groq's published `openai/gpt-oss-120b` rates (console, 2026-09):
$0.15 / 1M input, $0.60 / 1M output. Never prints payloads or keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60

# Estimation only: reserve a page-sized completion before the call so a fast
# loop cannot outrun the ledger between reserve and settle. Measured mean over
# the sealed Phase 4 cache was ~1,432 prompt / ~245 completion tokens.
COMPLETION_RESERVE_TOKENS = 300
CHARS_PER_TOKEN = 4

DEFAULT_BUDGET_DIR = Path(".budget")


class BudgetError(RuntimeError):
    """Base class for budget refusals; nothing was sent when raised."""


class BudgetExceeded(BudgetError):
    """The next call would cross the configured cap."""


class BudgetStopped(BudgetError):
    """The STOP sentinel exists."""


class BudgetLocked(BudgetError):
    """Another run holds the lock."""


@dataclass(frozen=True)
class Reservation:
    """Estimated spend held against the ledger for one in-flight call."""

    prompt_tokens: int
    completion_tokens: int
    usd: float


def budget_dir() -> Path:
    return Path(os.environ.get("PHISHNET_LLM_BUDGET_DIR", str(DEFAULT_BUDGET_DIR)))


def budget_id() -> str:
    return os.environ.get("PHISHNET_LLM_BUDGET_ID", "default") or "default"


def cap_usd() -> float | None:
    raw = os.environ.get("PHISHNET_LLM_BUDGET_USD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise BudgetError(f"PHISHNET_LLM_BUDGET_USD is not a number: {raw!r}") from exc


def estimate_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * INPUT_USD_PER_MILLION
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _ledger_path() -> Path:
    return budget_dir() / f"ledger-{budget_id()}.json"


def _empty_ledger() -> dict[str, object]:
    return {
        "budget_id": budget_id(),
        "calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "est_usd": 0.0,
    }


def _read_ledger() -> dict[str, object]:
    path = _ledger_path()
    if not path.exists():
        return _empty_ledger()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_ledger()
    return loaded if isinstance(loaded, dict) else _empty_ledger()


def _write_ledger(ledger: dict[str, object]) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def status() -> dict[str, object]:
    """Ledger snapshot plus the active cap, for CLIs and run headers."""
    ledger = _read_ledger()
    ledger["cap_usd"] = cap_usd()
    return ledger


def reset() -> None:
    """Zero the ledger for the current budget id (operator action only)."""
    _write_ledger(_empty_ledger())


def stop_path() -> Path:
    return budget_dir() / "STOP"


def reserve(prompt_text: str) -> Reservation:
    """Hold estimated spend, or refuse before any request is sent."""
    if stop_path().exists():
        raise BudgetStopped(f"STOP sentinel present: {stop_path()}")
    prompt_tokens = max(1, len(prompt_text) // CHARS_PER_TOKEN)
    reservation = Reservation(
        prompt_tokens=prompt_tokens,
        completion_tokens=COMPLETION_RESERVE_TOKENS,
        usd=estimate_usd(prompt_tokens, COMPLETION_RESERVE_TOKENS),
    )
    ledger = _read_ledger()
    prior = _as_float(ledger.get("est_usd"))
    cap = cap_usd()
    if cap is not None and prior + reservation.usd > cap:
        raise BudgetExceeded(
            f"next call would reach ${prior + reservation.usd:.4f} "
            f"> cap ${cap:.2f} (ledger: {_as_int(ledger.get('calls'))} calls, "
            f"${prior:.4f} spent)"
        )
    ledger["calls"] = _as_int(ledger.get("calls")) + 1
    ledger["prompt_tokens"] = (
        _as_int(ledger.get("prompt_tokens")) + reservation.prompt_tokens
    )
    ledger["completion_tokens"] = (
        _as_int(ledger.get("completion_tokens")) + reservation.completion_tokens
    )
    ledger["est_usd"] = prior + reservation.usd
    _write_ledger(ledger)
    return reservation


def settle(reservation: Reservation, usage: dict[str, object]) -> None:
    """Reconcile a reservation with the provider's reported usage.

    A call without usage (transport error, provider 5xx) keeps its
    reservation: an aborted request can still be billed, so the ledger is
    never allowed to under-count.
    """
    prompt_actual = usage.get("prompt_tokens")
    completion_actual = usage.get("completion_tokens")
    reported = isinstance(prompt_actual, (int, float)) or isinstance(
        completion_actual, (int, float)
    )
    new_prompt = _as_int(prompt_actual) if reported else reservation.prompt_tokens
    new_completion = (
        _as_int(completion_actual) if reported else reservation.completion_tokens
    )
    if new_prompt <= 0 and reported:
        new_prompt = reservation.prompt_tokens
    ledger = _read_ledger()
    if not reported:
        ledger["failed_calls"] = _as_int(ledger.get("failed_calls")) + 1
    ledger["prompt_tokens"] = max(
        0, _as_int(ledger.get("prompt_tokens")) - reservation.prompt_tokens + new_prompt
    )
    ledger["completion_tokens"] = max(
        0,
        _as_int(ledger.get("completion_tokens"))
        - reservation.completion_tokens
        + new_completion,
    )
    ledger["est_usd"] = max(
        0.0,
        _as_float(ledger.get("est_usd"))
        - reservation.usd
        + estimate_usd(new_prompt, new_completion),
    )
    _write_ledger(ledger)


def lock_path(name: str = "run") -> Path:
    return budget_dir() / f"{name}.lock"


def acquire_lock(name: str = "run") -> Path:
    """Create an exclusive run lock; raise if one already exists.

    Locks are never auto-broken: a stale lock after a crash is cleared by the
    operator (`uv run python scripts/llm_budget.py --unlock`), so a crashed
    run can neither be silently doubled nor silently ignored.
    """
    path = lock_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BudgetLocked(
            f"lock exists: {path} — another run may be active; "
            "remove it with 'uv run python scripts/llm_budget.py --unlock' "
            "only if you are sure no run is using it"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
    return path


def release_lock(name: str = "run") -> None:
    try:
        lock_path(name).unlink()
    except FileNotFoundError:
        pass
