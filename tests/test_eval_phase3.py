"""Tests for the Phase 3 fixed-threshold driver (pure rules only).

Heavy paths (scoring, model I/O) run in the driver itself; what is
pinned here is the Amendment E.3 transfer rule and the drift-interval
determinism, on synthetic arrays.
"""

from __future__ import annotations

import numpy as np

from ml_training.eval_phase3 import drift_ci, transfer_verdict


def test_transfer_verdict_fixed_below_phase2_miss() -> None:
    got = transfer_verdict(0.005, 0.0058, (0.0002, 0.0014))
    assert got["verdict"] == "fixed"
    assert got["drift_pp"] == __import__("pytest").approx(0.0008)
    assert got["phase2_miss_pp"] == 0.001


def test_transfer_verdict_not_fixed_at_or_above_miss() -> None:
    assert transfer_verdict(0.005, 0.006, (0.0, 0.002))["verdict"] == "not-fixed"
    bad = transfer_verdict(0.01, 0.012, (0.001, 0.003))
    assert bad["verdict"] == "not-fixed"


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
