"""Step 0 fetch set (§3.1): defined once, fetched once, frozen.

`fetch_set = step0_sample ∪ in_band(calib) ∪ in_band(test)`, where edges are
fixed on calib before any fetch. `step0_sample` (n ≈ 300–500) is stratified
by class × era (train/calib/test band) × survival stratum using the existing
`survival_stratum` and `source` columns. A row present in both components is
fetched once and reused — no row ever acquires two snapshots at two
timestamps.

Step-0 reporting (§3.2) crosses class × era × stratum × outcome with counts
and Wilson intervals; cells are descriptive, the trigger fires on marginals.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from phishnet.snapshot.bands import in_band
from phishnet.snapshot.tier1 import band_edges, score_band

SPLIT_DIR = "data/splits-p3"
STEP0_N_TARGET = 400
STEP0_N_MIN = 300
STEP0_N_MAX = 500
STEP0_SEED = 0


def _band_frame(band: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{SPLIT_DIR}/{band}.csv",
        usecols=["url", "label", "survival_stratum", "source"],
    )
    df["era"] = band
    return df


def step0_sample(rng_seed: int = STEP0_SEED) -> pd.DataFrame:
    """Stratified sample, class × era × survival stratum, n ≈ 300–500."""
    frames = [_band_frame(b) for b in ("train", "calib", "test")]
    full = pd.concat(frames, ignore_index=True)
    full["survival_stratum"] = full["survival_stratum"].fillna("unknown").astype(str)

    groups = full.groupby(["label", "era", "survival_stratum"])
    n_groups = len(groups)
    per_group = max(1, STEP0_N_TARGET // n_groups)
    rng = np.random.default_rng(rng_seed)
    parts = []
    for _, group in groups:
        take = min(len(group), per_group)
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        parts.append(full.loc[idx])
    sample = pd.concat(parts, ignore_index=True)
    assert STEP0_N_MIN <= len(sample) <= STEP0_N_MAX, (
        f"step0_sample n={len(sample)} outside [{STEP0_N_MIN}, {STEP0_N_MAX}]"
    )
    return sample


def in_band_rows(band: str, lower_edge: float, t_alert: float) -> pd.DataFrame:
    """All in-band rows of one band (Tier-1 scores, calib-fixed edges)."""
    y, scores = score_band(f"{SPLIT_DIR}/{band}.csv")
    frame = pd.read_csv(
        f"{SPLIT_DIR}/{band}.csv",
        usecols=["url", "label", "survival_stratum", "source"],
    )
    mask = np.array(
        [in_band(float(s), lower_edge, t_alert) for s in scores], dtype=bool
    )
    out = frame.loc[mask].copy()
    out["era"] = band
    out["tier1_score"] = np.asarray(scores, dtype=float)[mask]
    return out


def build_fetch_set() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch set + the calib-fixed edges that defined it."""
    y_calib, s_calib = score_band(f"{SPLIT_DIR}/calib.csv")
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    sample = step0_sample()
    inband = pd.concat(
        [
            in_band_rows("calib", lower_edge, t_alert),
            in_band_rows("test", lower_edge, t_alert),
        ],
        ignore_index=True,
    )
    combined = pd.concat([sample, inband], ignore_index=True)
    fetch_set = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    meta = {
        "t_alert": t_alert,
        "lower_edge": lower_edge,
        "n_step0_sample": int(len(sample)),
        "n_in_band": int(len(inband)),
        "n_fetch_set": int(len(fetch_set)),
        "tier1": "row-a",
    }
    return fetch_set, meta


def wilson_interval(count: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson 95% interval for a cell rate (descriptive, §3.2)."""
    if n == 0:
        return [0.0, 1.0]
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def step0_table(fetch_manifest: pd.DataFrame) -> pd.DataFrame:
    """Counts + Wilson intervals by class × era × stratum × outcome."""
    rows = []
    for keys, group in fetch_manifest.groupby(
        ["label", "era", "survival_stratum", "outcome"]
    ):
        n = len(group)
        ok = int((group["outcome"] == "ok").sum())
        lo, hi = wilson_interval(ok, n)
        rows.append(
            {
                "label": keys[0],
                "era": keys[1],
                "survival_stratum": keys[2],
                "outcome": keys[3],
                "n": n,
                "rate": ok / n if n else 0.0,
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
