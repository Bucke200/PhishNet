"""Escalation-band bucketing (pure, dependency-free).

Half-open buckets so every row falls in exactly one (§1.2)::

    below: (-inf, lower_edge)   — passes through, no LLM call
    band:  [lower_edge, t_alert) — escalated to the LLM
    alert: [t_alert, +inf)       — passes through, no LLM call

Comparison direction is `score >= edge` (shared with `threshold_at_fpr`,
which walks real score values with `>=`). This module is the single home of
that convention; the cascade and the tests import it rather than restating
it. `in_band` uses the same predicate, so the fetch set (§3.1) and the
cascade can never disagree on membership.
"""

from __future__ import annotations

BELOW = "below"
BAND = "band"
ALERT = "alert"


def bucket(score: float, lower_edge: float, t_alert: float) -> str:
    """Bucket for one score under half-open edges."""
    if score >= t_alert:
        return ALERT
    if score >= lower_edge:
        return BAND
    return BELOW


def in_band(score: float, lower_edge: float, t_alert: float) -> bool:
    """True iff the score escalates (band membership, one predicate)."""
    return bucket(score, lower_edge, t_alert) == BAND
