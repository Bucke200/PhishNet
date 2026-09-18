"""Step-0 trigger (§3.3): pure function of the Step-0 table.

Option 2 (joined feature) is eligible only if ALL three hold:

1. train-band phishing fetch success ≥ 40%;
2. train-band class gap (benign − phishing success) ≤ 0.05;
3. test-band class gap (benign − phishing success) ≤ 0.05.

Otherwise option 1 (verdict-as-report, on the test band only). Expectation on
record: option 1 fires — a gate working, not a fetch gone badly. The 0.05
budget is Phase 3's unknown-gap budget, reused deliberately. Fetch success
means outcome == `ok`. Any wish to move a bar after seeing the rate is itself
the amendment and must say so.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

TRAIN_PHISH_MIN = 0.40
CLASS_GAP_MAX = 0.05

OPTION1 = "option-1"
OPTION2 = "option-2"


def _success_rate(frame: pd.DataFrame, label: int, era: str) -> float | None:
    subset = frame[(frame["label"] == label) & (frame["era"] == era)]
    if len(subset) == 0:
        return None
    return float((subset["outcome"] == "ok").mean())


def trigger_verdict(fetch_manifest: pd.DataFrame) -> dict[str, Any]:
    """Mechanical verdict: option, marginals, and which condition failed."""
    train_phish = _success_rate(fetch_manifest, 1, "train")
    train_benign = _success_rate(fetch_manifest, 0, "train")
    test_phish = _success_rate(fetch_manifest, 1, "test")
    test_benign = _success_rate(fetch_manifest, 0, "test")

    cond1 = train_phish is not None and train_phish >= TRAIN_PHISH_MIN
    train_gap = (
        (train_benign - train_phish)
        if train_benign is not None and train_phish is not None
        else None
    )
    test_gap = (
        (test_benign - test_phish)
        if test_benign is not None and test_phish is not None
        else None
    )
    cond2 = train_gap is not None and train_gap <= CLASS_GAP_MAX
    cond3 = test_gap is not None and test_gap <= CLASS_GAP_MAX

    option = OPTION2 if (cond1 and cond2 and cond3) else OPTION1
    scope = "all-bands-joined" if option == OPTION2 else "test-band-only"
    return {
        "option": option,
        "scope": scope,
        "train_phish_success": train_phish,
        "train_benign_success": train_benign,
        "test_phish_success": test_phish,
        "test_benign_success": test_benign,
        "train_gap": train_gap,
        "test_gap": test_gap,
        "conditions": {
            "train_phish_ge_0.40": bool(cond1),
            "train_gap_le_0.05": bool(cond2),
            "test_gap_le_0.05": bool(cond3),
        },
    }
