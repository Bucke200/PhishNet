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
    """A Tier-2 judgment or its failure mode.

    ``trigger_type`` / ``trigger_match`` carry the observability detail for a
    failure (which token or status produced it), so the serving log can break
    false alarms down by cause.
    """

    kind: str
    reason: str = ""
    trigger_type: str = ""
    trigger_match: str = ""


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


# Mechanism-aware failure policy (2026-09-22). Active rejection (WAF/bot-wall)
# and dead links are different signals and must not share one blunt threshold.
# The value is the minimum Tier-1 score at which the mechanism alerts; None
# means the mechanism never alerts. A mechanism absent from the table is
# handled in `_mechanism_failure`.
FAILURE_MECHANISM_THRESHOLDS: dict[str, float | None] = {
    "http_403": 0.65,  # active rejection on an in-band domain -> cloaking
    "blocked": 0.65,  # WAF / bot-wall / challenge interstitial
    "dns": 0.70,  # dead / sinkholed / fast-flux infrastructure
    "refused": 0.70,
    "tls": 0.70,  # invalid or expired cert on a brand subdomain
    "http_5xx": 0.70,  # server error on a suspicious domain
    "origin_timeout": 0.80,  # tarpit / unresponsive; high ambiguity
    # Cloudflare Access / Zero Trust identity gates: a legit enterprise portal
    # (typically 0.65-0.75) is not alerted on, but the signal is spoofable, so
    # a high-scoring page presenting a fake gate still alerts.
    "auth_gateway": 0.80,
    "http_404": None,  # link rot -> never alert
    "fetcher_timeout": None,  # provider->fetcher RPC: internal, no verdict
    "fetcher_error": None,
    "fetcher_http": None,
}
# The LLM was reached but produced no verdict. Phase 5 measured schema
# refusals concentrated on phishing (0% on benign), so failing closed here is
# high-precision and stays an alert under every policy.
LLM_FAILURE_REASONS = frozenset({"schema", "refusal", "api_error"})


def _mechanism_failure(reason: str, score: float, t_alert: float) -> Decision:
    if reason in LLM_FAILURE_REASONS or reason.startswith("http="):
        return Decision(
            ALERT, alert_anchor(t_alert), f"tier2_failure:{reason}", in_band=True
        )
    threshold = FAILURE_MECHANISM_THRESHOLDS.get(reason)
    if threshold is not None and score >= threshold:
        return Decision(
            ALERT, alert_anchor(t_alert), f"tier2_failure:{reason}", in_band=True
        )
    return Decision(CANT_ASSESS, None, f"tier2_failure:{reason}", in_band=True)


def decide(
    tier1_score: float | None,
    outcome: Tier2Outcome | None,
    *,
    t_alert: float,
    lower_edge: float,
    unresolved: bool = False,
    no_verdict_reason: str = "tier2_not_configured",
    failure_floor: float | None = None,
    failure_policy: str = "closed",
) -> Decision:
    """Map a Tier-1 score (+ optional Tier-2 outcome) to a disposition.

    ``no_verdict_reason`` distinguishes "no Tier-2 provider" from "provider
    configured but this URL is not in it" (e.g. the sealed demo cache holds
    only the registered pages), so a live in-band URL is not mislabeled as a
    missing configuration.

    ``failure_floor`` implements the risk-graded fail-closed policy (T2-9,
    live-performance plan). When set, a Tier-2 *failure* alerts only if the
    Tier-1 score is at or above the floor; a failure below it becomes
    ``can't assess`` instead. When ``None`` (the registered behavior) every
    failure alerts. This is a deliberate, recorded weakening of Phase 5's
    unconditional fail-closed mapping: live measurement showed 23 of 35
    benign false alarms were fetch failures on infra hosts sitting in the low
    half of the band, and grading recovered them with no measured recall loss.

    ``failure_policy`` selects the failure rule: ``"closed"`` (registered —
    any failure alerts), ``"graded"`` (one ``failure_floor`` for all
    failures), or ``"mechanism"`` (per-mechanism thresholds in
    ``FAILURE_MECHANISM_THRESHOLDS``).
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
    # Fail closed: any failure (including an unknown kind) alerts, unless a
    # policy says otherwise. "mechanism" keys on the failure mechanism
    # (see FAILURE_MECHANISM_THRESHOLDS); "graded" uses one floor for all
    # failures; "closed" (default) is the registered alert-on-any-failure.
    reason = outcome.reason or "unknown"
    if failure_policy == "mechanism":
        return _mechanism_failure(reason, score, t_alert)
    if (
        failure_policy == "graded"
        and failure_floor is not None
        and score < failure_floor
    ):
        return Decision(
            CANT_ASSESS, None, f"tier2_failure_below_floor:{reason}", in_band=True
        )
    return Decision(
        ALERT, alert_anchor(t_alert), f"tier2_failure:{reason}", in_band=True
    )
