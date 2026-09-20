"""C1 — serving identity: `/predict` Tier-1 == the Phase 3 headline scorer.

The container's Tier-1 path (`phishnet.serving.Tier1Servable`, a pandas-free
LightGBM fast path) must score bit-identically (max abs diff 0.0) to
`phishnet.snapshot.tier1.score_band` — the eval-mode `EnrichedGbm` path the
Phase 3 headline numbers were computed with — on every row of the calib and
test bands, one URL at a time.

Also pinned here: the fast path equals `featurise_frame` + `hosted_flag` +
`predict_proba` (C2's "bit-equal (max abs diff 0.0) to `featurise_frame`"),
and startup refuses to run on a model/column or threshold mismatch (no
degraded mode).
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phishnet.enrichment.features import HOSTED_COLUMN, hosted_flag
from phishnet.features.extraction import featurise_frame
from phishnet.serving import Tier1Servable
from phishnet.snapshot.tier1 import ROW_A_ASSETS, score_band

ROOT = Path(__file__).resolve().parents[1]
BANDS = (
    ROOT / "data" / "splits-p3" / "calib.csv",
    ROOT / "data" / "splits-p3" / "test.csv",
)


@pytest.fixture(scope="module")
def servable() -> Tier1Servable:
    return Tier1Servable()


def _band_urls(path: Path) -> list[str]:
    return pd.read_csv(path, usecols=["url"])["url"].astype(str).tolist()


def test_serving_matches_headline_on_every_row(servable: Tier1Servable) -> None:
    """C1: max abs diff 0.0 over the full calib and test bands."""
    for band in BANDS:
        y, headline = score_band(str(band))
        urls = _band_urls(band)
        assert len(urls) == len(headline)
        served = np.array([servable.score_one(u) for u in urls], dtype=float)
        diff = float(np.max(np.abs(served - headline)))
        assert diff == 0.0, f"{band.name}: max abs diff {diff}"


def test_fast_path_equals_featurise_frame(servable: Tier1Servable) -> None:
    """C2: the single-URL fast path is bit-equal to the batch pipeline."""
    urls = _band_urls(BANDS[0])[:500]
    frozen = [c for c in servable.columns if c != HOSTED_COLUMN]
    frame = featurise_frame(urls, frozen, canonicalize=servable.canonicalize)
    frame[HOSTED_COLUMN] = hosted_flag(urls)
    reference = servable._booster.predict(frame[servable.columns].to_numpy(dtype=float))
    served = np.array([servable.score_one(u) for u in urls], dtype=float)
    assert float(np.max(np.abs(served - np.asarray(reference, dtype=float)))) == 0.0


def test_explain_matches_scoring_row(servable: Tier1Servable) -> None:
    url = _band_urls(BANDS[0])[0]
    out = servable.explain_one(url, top_k=5)
    assert len(out["features"]) == 5
    row = servable.row(url)
    contributions = servable._booster.predict(row, pred_contrib=True).ravel()
    assert contributions.shape[0] == len(servable.columns) + 1
    assert out["bias"] == pytest.approx(float(contributions[-1]))


def test_startup_refuses_model_hash_mismatch(tmp_path: Path) -> None:
    assets = ROOT / ROW_A_ASSETS
    (tmp_path / "ablation_lexical_gbm_model.pkl").write_bytes(b"not a model")
    shutil.copy(
        assets / "feature_columns.pkl",
        tmp_path / "ablation_lexical_feature_columns.pkl",
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        Tier1Servable(
            assets_dir=tmp_path,
            threshold_file=ROOT / "reports" / "phase4.json",
        )


def test_startup_refuses_threshold_mismatch(tmp_path: Path) -> None:
    source = json.loads((ROOT / "reports" / "phase4.json").read_text(encoding="utf-8"))
    source["t_alert"] = source["t_alert"] + 1e-9
    tampered = tmp_path / "phase4.json"
    tampered.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RuntimeError, match="threshold t_alert"):
        Tier1Servable(threshold_file=tampered)


def test_columns_file_is_pinned_vocabulary() -> None:
    """The loaded vocabulary is the manifest-pinned 79 columns."""
    columns_path = (
        ROOT
        / "src"
        / "phishnet"
        / "urlset_ml_assets"
        / "ablation_lexical_feature_columns.pkl"
    )
    cols = pickle.loads(columns_path.read_bytes())
    assert len(cols) == 79
    assert cols[-1] == HOSTED_COLUMN
