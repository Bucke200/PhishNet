"""C3 — fail-closed Tier-2 and the forbidden Retain mapping.

Any in-band Tier-2 outcome other than a valid parsed judgment alerts; a valid
non-phishing judgment keeps the (sub-threshold) Tier-1 score. The Retain
policy — retain Tier-1 on failure — must be unreachable: Phase 5 measured it
at 62.3% evasion and prohibited it. The invariant test mirrors the whitelist
test's pattern (a quoted sentinel + a behavioral pin).
"""

from __future__ import annotations

import inspect

import pytest

from phishnet.serving import cascade
from phishnet.serving.cascade import ALERT, ALLOW, CANT_ASSESS, Tier2Outcome, decide

T_ALERT = 0.9269363298832987
LOWER = 0.6493076453312958
ANCHOR = cascade.alert_anchor(T_ALERT)
IN_BAND = (LOWER + T_ALERT) / 2


def test_tier1_above_threshold_alerts() -> None:
    d = decide(T_ALERT, None, t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALERT
    assert d.score == T_ALERT


def test_tier1_below_lower_edge_allows() -> None:
    d = decide(LOWER - 0.01, None, t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALLOW
    assert d.in_band is False


def test_in_band_phishing_pins_to_anchor() -> None:
    d = decide(IN_BAND, Tier2Outcome("phishing"), t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALERT
    assert d.score == ANCHOR
    assert d.in_band is True


def test_in_band_benign_keeps_tier1_score() -> None:
    d = decide(IN_BAND, Tier2Outcome("benign"), t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALLOW
    assert d.score == IN_BAND


def test_in_band_suspicious_keeps_tier1_score() -> None:
    d = decide(IN_BAND, Tier2Outcome("suspicious"), t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALLOW
    assert d.score == IN_BAND


@pytest.mark.parametrize("reason", cascade.FAILURE_REASONS)
def test_in_band_failure_alerts(reason: str) -> None:
    d = decide(
        IN_BAND, Tier2Outcome("failure", reason), t_alert=T_ALERT, lower_edge=LOWER
    )
    assert d.disposition == ALERT
    assert d.score == ANCHOR


def test_unknown_outcome_kind_alerts() -> None:
    """Fail closed even on an outcome shape we do not recognize."""
    d = decide(IN_BAND, Tier2Outcome("wat"), t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == ALERT


def test_in_band_without_tier2_cannot_assess() -> None:
    d = decide(IN_BAND, None, t_alert=T_ALERT, lower_edge=LOWER)
    assert d.disposition == CANT_ASSESS
    assert d.score is None


def test_unresolved_shortener_has_no_verdict_score() -> None:
    d = decide(None, None, t_alert=T_ALERT, lower_edge=LOWER, unresolved=True)
    assert d.disposition == CANT_ASSESS
    assert d.score is None
    assert d.reason == "unresolved_shortener"


def test_retain_policy_is_unreachable() -> None:
    """Retain is banned (Phase 5 §5): failures alert, never retain Tier-1.

    Sentinel: the module may discuss the policy in prose ("Retain policy"),
    but no quoted ``"retain"`` token exists, and the behavioral pin below
    shows a failure outcome returns the alert anchor, not the Tier-1 score.
    """
    source = inspect.getsource(cascade)
    assert '"retain"' not in source
    assert "'retain'" not in source
    assert not hasattr(cascade, "RETAIN")

    d = decide(
        IN_BAND, Tier2Outcome("failure", "schema"), t_alert=T_ALERT, lower_edge=LOWER
    )
    assert d.disposition == ALERT
    assert d.score != IN_BAND
    assert d.score == ANCHOR
