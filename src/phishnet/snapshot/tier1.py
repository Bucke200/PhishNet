"""Tier 1 scorer: Phase 3 row (a), headline path, loaded by asset hash.

Row (a) is the lexical + `is_hosted_tenant` ablation
(`backend/ablation_lexical_assets/`), scored through the exact Phase 3
headline path — `predictors.EnrichedGbm` on the pinned snapshot
(`data/enrichment-p3-2026-09-17.jsonl`, `run-1`) with `first_seen` from the
band CSV. A lightweight `GbmSingle` on the same weights diverges (max abs
diff ≈ 0.94 on a 200-row calib probe: different featurisation of the frozen
columns), so it must NOT substitute: band edges have to be fixed on the
scores the headline was computed from, or the band will not match it.

Band edges (§1.2) are derived from calib only, by calling
`eval.threshold_at_fpr` twice — never hard-coded, never swept::

    t_alert    = threshold_at_fpr(calib, 0.005)
    lower_edge = threshold_at_fpr(calib, 0.005 + 0.05)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import eval as E  # noqa: E402
import predictors  # noqa: E402

ROW_A_ASSETS = "backend/ablation_lexical_assets"
ROW_A_MODEL_HASH = "7b765bfc82716350555d38d01f2246215f79661803b2468097979ec7b944024f"
ROW_A_COLUMNS_HASH = "39d0e665391b06557ced4a648caf9834b76cde83ffa6637128dd5b2becacd79e"
PINNED_SNAPSHOT = "data/enrichment-p3-2026-09-17.jsonl"
PINNED_RUN = "run-1"

T_ALERT_FPR = 0.005
BAND_BENIGN_MASS = 0.05


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_row_a(first_seen_csv: str) -> predictors.EnrichedGbm:
    """Row (a) scorer in eval mode, with asset-hash assertion.

    `first_seen_csv` is the band being scored (timestamps are observation
    metadata, never labels — the same sourcing `EnrichedGbm` documents).
    Raises if the weights or vocabulary differ from the pinned hashes.
    """
    pred = predictors.EnrichedGbm(
        assets_dir=ROW_A_ASSETS,
        snapshot=PINNED_SNAPSHOT,
        run_id=PINNED_RUN,
        first_seen_csv=first_seen_csv,
    )
    fingerprint = pred.asset_fingerprint
    if fingerprint.get("model") != ROW_A_MODEL_HASH:
        raise ValueError(
            "row (a) model hash mismatch: "
            f"{fingerprint.get('model')} != {ROW_A_MODEL_HASH}"
        )
    if fingerprint.get("columns") != ROW_A_COLUMNS_HASH:
        raise ValueError("row (a) columns hash mismatch")
    return pred


def score_band(band_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """Labels and Tier-1 scores for every row of a band CSV."""
    pred = load_row_a(band_csv)
    frame = pd.read_csv(band_csv, usecols=["url", "label"])
    urls = frame["url"].astype(str).tolist()
    y = frame["label"].to_numpy(dtype=int)
    scores, _ = E.score_all(pred, urls, batch_size=512)
    return y, np.asarray(scores, dtype=float)


def band_edges(y_calib: np.ndarray, s_calib: np.ndarray) -> tuple[float, float]:
    """`(t_alert, lower_edge)` from calib only — one function, called twice."""
    t_alert = E.threshold_at_fpr(y_calib, s_calib, T_ALERT_FPR)
    lower_edge = E.threshold_at_fpr(y_calib, s_calib, T_ALERT_FPR + BAND_BENIGN_MASS)
    return float(t_alert), float(lower_edge)


def achieved_benign_band_mass(
    y_calib: np.ndarray, s_calib: np.ndarray, lower_edge: float, t_alert: float
) -> float:
    """Achieved (not nominal) benign mass in `[lower_edge, t_alert)`."""
    benign = s_calib[y_calib == 0]
    if benign.size == 0:
        raise ValueError("no benign rows in calib")
    return float(((benign >= lower_edge) & (benign < t_alert)).mean())


def edge_report(y_calib: np.ndarray, s_calib: np.ndarray) -> dict[str, Any]:
    """Edges plus the achieved-mass numbers the report states (§1.2)."""
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    return {
        "t_alert": t_alert,
        "lower_edge": lower_edge,
        "achieved_benign_band_mass": achieved_benign_band_mass(
            y_calib, s_calib, lower_edge, t_alert
        ),
        "tier1_assets": ROW_A_ASSETS,
        "tier1_model_hash": ROW_A_MODEL_HASH,
        "n_calib": int(y_calib.size),
    }
