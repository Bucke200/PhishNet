"""Tests for the Phase 3 fixed-threshold driver (pure rules only).

Heavy paths (scoring, model I/O) run in the driver itself; what is
pinned here is the Amendment E.3 transfer rule and the drift-interval
determinism, on synthetic arrays.
"""

from __future__ import annotations

import numpy as np

from ml_training.eval_phase3 import drift_ci, transfer_verdict


def test_transfer_verdict_interval_rules() -> None:
    # Interval wholly inside the bar: fixed.
    got = transfer_verdict(0.005, 0.0058, (0.0002, 0.0009))
    assert got["verdict"] == "fixed"
    assert got["drift_pp"] == __import__("pytest").approx(0.0008)
    assert got["bar_pp"] == 0.001
    # Interval straddling the bar: indistinguishable (review correction —
    # the point-estimate rule could not conclude here).
    assert (
        transfer_verdict(0.005, 0.0058, (0.0002, 0.0014))["verdict"]
        == "indistinguishable"
    )
    # Interval wholly above the bar: not-fixed.
    bad = transfer_verdict(0.01, 0.012, (0.0011, 0.003))
    assert bad["verdict"] == "not-fixed"
    # No comparator (1% target): indistinguishable with reason.
    assert (
        transfer_verdict(0.01, 0.012, (0.0011, 0.003), None)["verdict"]
        == "indistinguishable"
    )


def test_paired_lift_own_thresholds() -> None:
    from ml_training.eval_phase3 import paired_recall_lift

    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 100)
    s_a = np.concatenate([rng.random(100), rng.random(100) + 0.3])
    s_b = np.concatenate([rng.random(100), rng.random(100) + 0.6])
    groups = np.array([f"g{i // 4}" for i in range(200)])
    got = paired_recall_lift(y, s_a, s_b, 0.5, 0.7, 50, 0, groups)
    lo, hi = got["recall_lift_ci"]
    assert lo <= hi and hi > 0  # (b) dominates by construction


def test_drift_ci_deterministic_and_sane() -> None:
    rng = np.random.default_rng(0)
    y_c = np.array([0] * 200 + [1] * 200)
    y_t = np.array([0] * 200 + [1] * 200)
    s_c = np.concatenate([rng.random(200), rng.random(200) + 0.5])
    s_t = np.concatenate([rng.random(200), rng.random(200) + 0.5])
    g_c = np.array([f"c{i // 4}" for i in range(400)])
    g_t = np.array([f"t{i // 4}" for i in range(400)])
    ci1 = drift_ci(y_c, s_c, y_t, s_t, 0.5, 50, 0, g_c, g_t)
    ci2 = drift_ci(y_c, s_c, y_t, s_t, 0.5, 50, 0, g_c, g_t)
    assert ci1 == ci2
    assert ci1[0] <= ci1[1]
