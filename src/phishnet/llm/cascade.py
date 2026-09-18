"""Cascade predictor (§2): Tier-1 scores with exactly one moving arm.

Only in-band rows reach the LLM; the v1 mapping has exactly one arm that
moves — `verdict == phishing` → `nextafter(t_alert, +inf)` — everything else
retains its Tier-1 score (benign, suspicious, schema violation, refusal, API
failure, and unfetchable under the default policy). Rows below the band and
rows already in alert pass through without an LLM call. No confidence gating
in v1.

Consequence (§2, repeated in the report): anchoring every phishing-verdict
row to a single float makes cascade PR-AUC/ROC-AUC partly artifactual, with
a tie block exactly at `t_alert`. Fixed-threshold recall/FPR is primary;
rank metrics are descriptive. Bucketing reuses `snapshot.bands` (one
predicate), and the phishing anchor is `math.nextafter(t_alert, +inf)`.

Unfetchable policies: `default` keeps the Tier-1 score; `alternative` routes
to human review — scored identically here (Tier-1) and counted as a separate
disposition in the report from the manifest + verdicts. Both arms are
reported for every cascade number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from phishnet.snapshot.bands import in_band

PHISHING = "phishing"


class CascadePredictor:
    """`eval.py`-compatible cascade over frozen Tier-1 scores + verdicts."""

    def __init__(
        self,
        tier1_by_url: dict[str, float],
        verdict_by_url: dict[str, str],
        t_alert: float,
        lower_edge: float,
        unfetchable_policy: str = "default",
        name: str = "cascade(tier1+llm)",
    ):
        if unfetchable_policy not in ("default", "alternative"):
            raise ValueError(f"unknown unfetchable_policy: {unfetchable_policy}")
        self.tier1_by_url = tier1_by_url
        self.verdict_by_url = verdict_by_url
        self.t_alert = t_alert
        self.lower_edge = lower_edge
        self.unfetchable_policy = unfetchable_policy
        self.name = f"{name}[{unfetchable_policy}]"
        self.anchor = math.nextafter(t_alert, math.inf)

    def score(self, urls: Sequence[str]) -> list[float]:
        out = []
        for url in urls:
            tier1 = float(self.tier1_by_url[url])
            if not in_band(tier1, self.lower_edge, self.t_alert):
                out.append(tier1)
                continue
            verdict = self.verdict_by_url.get(url)
            if verdict == PHISHING:
                out.append(self.anchor)
            else:
                out.append(tier1)
        return out


def tier1_only_predictor(
    tier1_by_url: dict[str, float],
) -> Callable[[Sequence[str]], list[float]]:
    """Plain Tier-1 reference scorer (no LLM arm) for the gap comparison."""

    class _Tier1Only:
        name = "tier1(row-a)"

        def score(self, urls: Sequence[str]) -> list[float]:
            return [float(tier1_by_url[u]) for u in urls]

    return _Tier1Only()
