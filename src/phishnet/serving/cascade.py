"""Serving cascade: Tier-1 thresholds -> fail-closed Tier-2 (C3).

Registered vocabulary (phase6-F): ``allow``, ``alert``, ``can't assess``.
A shortener that could not be resolved is ``can't assess`` with reason
``unresolved_shortener``.

Tier-2 mapping (`phase6-A`): the operationally shippable Phase 5 arm —
``p5-h1`` hardened prompt plus the frozen detector — with a **fail-closed**
exception handler. A valid parsed ``phishing`` judgment pins the score to
``nextafter(t_alert, +inf)``; a valid parsed ``benign``/``suspicious``
judgment keeps the Tier-1 score (below ``t_alert``, i.e. an ``allow``); and
**every other outcome** (schema 400, refusal, API error, timeout, unfetchable,
or no Tier-2 configured) maps to ``alert``.

The Retain policy (retain the Tier-1 score on failure) is deliberately
unreachable: there is no mapping, flag, or config that produces it. Phase 5
measured it at 62.3% evasion and prohibited it from production; an invariant
test blocks re-adding it, the same way the whitelist is blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ALLOW = "allow"
ALERT = "alert"
CANT_ASSESS = "can't assess"

PHISHING = "phishing"
BENIGN = "benign"
SUSPICIOUS = "suspicious"
FAILURE = "failure"

# Failure classes that must alert (never retain). Descriptive only: any
# non-valid-parsed outcome alerts, whatever its reason string.
FAILURE_REASONS = ("schema", "refusal", "api_error", "timeout", "unfetchable")


@dataclass(frozen=True)
class Tier2Outcome:
    """A Tier-2 judgment or its failure mode."""

    kind: str
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    """Final serving disposition for one URL."""

    disposition: str
    score: float | None
    reason: str
    in_band: bool


def alert_anchor(t_alert: float) -> float:
    """The single phishing anchor the cascade pins to (Phase 4 §2)."""
    return math.nextafter(t_alert, math.inf)


def decide(
    tier1_score: float | None,
    outcome: Tier2Outcome | None,
    *,
    t_alert: float,
    lower_edge: float,
    unresolved: bool = False,
    no_verdict_reason: str = "tier2_not_configured",
) -> Decision:
    """Map a Tier-1 score (+ optional Tier-2 outcome) to a disposition.

    ``no_verdict_reason`` distinguishes "no Tier-2 provider" from "provider
    configured but this URL is not in it" (e.g. the sealed demo cache holds
    only the registered pages), so a live in-band URL is not mislabeled as a
    missing configuration.
    """
    if unresolved:
        return Decision(CANT_ASSESS, None, "unresolved_shortener", in_band=False)
    if tier1_score is None:
        raise ValueError("tier1_score is required unless unresolved=True")

    score = float(tier1_score)
    if score >= t_alert:
        return Decision(ALERT, score, "tier1>=t_alert", in_band=False)
    if score < lower_edge:
        return Decision(ALLOW, score, "tier1<lower_edge", in_band=False)

    # In band: Tier-2 decides.
    if outcome is None:
        return Decision(CANT_ASSESS, None, no_verdict_reason, in_band=True)
    if outcome.kind == PHISHING:
        return Decision(ALERT, alert_anchor(t_alert), "tier2_phishing", in_band=True)
    if outcome.kind in (BENIGN, SUSPICIOUS):
        return Decision(ALLOW, score, f"tier2_{outcome.kind}", in_band=True)
    # Fail closed: any failure (including an unknown kind) alerts.
    return Decision(
        ALERT,
        alert_anchor(t_alert),
        f"tier2_failure:{outcome.reason or 'unknown'}",
        in_band=True,
    )
