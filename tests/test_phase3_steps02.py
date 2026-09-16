"""Phase 3 Steps 0-2: split provenance, is_https rule, cache key, stub, store.

Pins the nine-correction contract before any enrichment runs:
first_snapshot/strata from filenames, pre-committed is_https rule,
PSL-keyed cache rule with hosted na, stub == forced-100%-miss grouped by
cache key, CT pre-first_seen filter at join time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

import build_splits
import predictors
from phishnet.enrichment.key import (
    cache_key,
    check_psl_splits,
    gate_psl_snapshot,
    hosted_share,
)
from phishnet.enrichment.store import (
    append_records,
    load_pinned_run,
    na_unknown_rates,
    pre_first_seen_filter,
    seal_run,
    select_earliest_success,
)
from phishnet.enrichment.stub import UnknownStubProvider, force_miss
from phishnet.enrichment.types import EnrichedRecord


def test_snapshot_date_from_name() -> None:
    assert (
        build_splits.snapshot_date_from_name("openphish-2026-09-12.jsonl")
        == "2026-09-12"
    )
    assert (
        build_splits.snapshot_date_from_name(
            "benign-cc-CC-MAIN-2026-34-2026-09-15.jsonl"
        )
        == "2026-09-15"
    )
    assert build_splits.snapshot_date_from_name("fixture.jsonl") is None


def test_survival_stratum_boundaries() -> None:
    assert build_splits.survival_stratum(0.0) == "fresh"
    assert build_splits.survival_stratum(2.0) == "fresh"
    assert build_splits.survival_stratum(2.5) == "short"
    assert build_splits.survival_stratum(30.0) == "short"
    assert build_splits.survival_stratum(30.5) == "long"
    assert build_splits.survival_stratum(None) == "na"
    assert build_splits.survival_stratum(float("nan")) == "na"


def test_is_https_rule_drops_on_real_gap() -> None:
    # Test-era reality (benign ~0.97, phish ~0.79): gap >> 0.04 -> DROP.
    assert build_splits.should_drop_is_https(0.97, 0.789) is True
    assert build_splits.should_drop_is_https(0.91, 0.905) is False


def test_enrich_keeps_earliest_snapshot_and_strata() -> None:
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-10 08:00:00+00:00",
                "first_snapshot": "2026-09-12",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-12.jsonl",
            },
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-10 08:00:00+00:00",
                "first_snapshot": "2026-09-12",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-12.jsonl",
            },
            {
                "url": "https://benign.example/b",
                "label": 0,
                "first_seen": "2026-09-13 08:00:00+00:00",
                "first_snapshot": "2026-09-13",
                "source": "tranco:x",
                "time_basis": "crawled",
                "_snap_file": "benign-2026-09-13.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df, phase3=True)
    assert len(out) == 2
    phish = out[out.label == 1].iloc[0]
    assert phish["first_snapshot"] == "2026-09-12"  # minimum wins
    assert phish["survival_stratum"] == "fresh"  # same-file lag ~0
    assert float(phish["survival_lag_days"]) >= 0.0
    benign = out[out.label == 0].iloc[0]
    assert benign["survival_stratum"] == "na"


def test_default_enrich_stays_legacy_clean() -> None:
    """Without phase3=True, enrich adds no survival columns: pinned
    rebuilds are byte-identical with or without the flag existing."""
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-10 08:00:00+00:00",
                "first_snapshot": "2026-09-12",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-12.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df)
    assert "survival_stratum" not in out.columns
    assert "survival_lag_days" not in out.columns
    assert "snapshot_anchor" not in out.columns
    assert build_splits.LAST_ANCHORS == {}


def test_observed_basis_rows_go_to_unknown_not_fresh() -> None:
    """OpenPhish rows (time_basis observed) are live phish of unknown age."""
    df = pd.DataFrame(
        [
            {
                "url": "http://evil.example/login",
                "label": 1,
                "first_seen": "2026-09-12 13:06:02+00:00",
                "first_snapshot": "2026-09-12",
                "source": "openphish",
                "time_basis": "observed",
                "_snap_file": "openphish-2026-09-12.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df, phase3=True)
    assert out.iloc[0]["survival_stratum"] == "unknown"


def test_negative_lag_rejected() -> None:
    """A snapshot collection time preceding first_seen is broken input.

    Legacy-style rows (snapshot date, no source file): the fallback is
    end-of-day of the filename date, so a row stamped after its own
    snapshot date must refuse rather than mis-stratify. (With `_snap_file`
    present the lag is non-negative by construction: the file ts is the
    file's max first_seen, which dominates every row stamp inside it.)
    """
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-16 08:00:00+00:00",
                "first_snapshot": "2026-09-12",
                "source": "phishtank",
                "time_basis": "submitted",
            },
        ]
    )
    try:
        build_splits.enrich(df, phase3=True)
    except ValueError as e:
        assert "negative survival lag" in str(e)
    else:
        raise AssertionError("expected ValueError for negative lag")


def test_openphish_run_stamp_anchors_same_date_dump() -> None:
    """The run stamp (08:38), not the dump max (07:03), dates same-day rows.

    Verified against data/raw: every openphish file is single-valued
    (13:06:02 / 08:16:04 / 08:51:35 / 08:38:27) and each postdates its
    same-date phishtank dump max (11:02 / 06:53 / 02:37 / 07:03).
    """
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-15 07:03:46+00:00",
                "first_snapshot": "2026-09-15",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-15.jsonl",
            },
            {
                "url": "http://feeds.example/x",
                "label": 1,
                "first_seen": "2026-09-15 08:38:27+00:00",
                "first_snapshot": "2026-09-15",
                "source": "openphish",
                "time_basis": "observed",
                "_snap_file": "openphish-2026-09-15.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df, phase3=True)
    phish = out[out.url.str.contains("example.com/a")].iloc[0]
    # Lag vs the run stamp: 95 min, not ~0 (dump-max anchor) nor −7h.
    assert phish["snapshot_anchor"] == "openphish-run-stamp"
    assert abs(float(phish["survival_lag_days"]) - 95 / 1440) < 0.01
    anchors = build_splits.LAST_ANCHORS
    assert anchors["phishtank-2026-09-15.jsonl"]["method"] == ("openphish-run-stamp")
    assert anchors["phishtank-2026-09-15.jsonl"]["ts"] == ("2026-09-15T08:38:27+00:00")


def test_file_max_fallback_without_openphish() -> None:
    """No same-date openphish file: the dump max anchors, method recorded."""
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-14 02:00:00+00:00",
                "first_snapshot": "2026-09-14",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-14.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df, phase3=True)
    assert out.iloc[0]["snapshot_anchor"] == "file-max"
    assert float(out.iloc[0]["survival_lag_days"]) == 0.0


def test_same_day_late_submission_lag_nonnegative() -> None:
    """The 07:03-submitted / 08:38-collected case: lag ~+1h, never −7h."""
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a",
                "label": 1,
                "first_seen": "2026-09-15 07:03:00+00:00",
                "first_snapshot": "2026-09-15",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-15.jsonl",
            },
            {
                "url": "http://example.com/b",
                "label": 1,
                "first_seen": "2026-09-15 08:38:00+00:00",
                "first_snapshot": "2026-09-15",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-15.jsonl",
            },
        ]
    )
    out = build_splits.enrich(df, phase3=True)
    lags = out["survival_lag_days"].astype(float)
    assert bool((lags >= 0.0).all())
    # Midnight-of-filename-date would have read ≈ −7h for row a.
    assert float(lags.min()) < 1.0  # collection-time based, hours not days


def test_cache_key_hosted_vs_plain() -> None:
    key, hosted = cache_key("https://login.core.windows.net/tenant/path")
    assert hosted is True
    assert key == "login.core.windows.net"
    key2, hosted2 = cache_key("https://mail.example.com/inbox")
    assert hosted2 is False
    assert key2 == "example.com"
    # Same platform, different tenants -> different keys (no smearing).
    ka, _ = cache_key("https://a.core.windows.net/x")
    kb, _ = cache_key("https://b.core.windows.net/x")
    assert ka != kb


def test_tenant_grouping() -> None:
    from phishnet.enrichment.key import is_hosted_tenant, tenant_group

    assert is_hosted_tenant("login.core.windows.net") is True
    assert is_hosted_tenant("mail.example.com") is False
    assert is_hosted_tenant("") is False
    # Non-hosted: public registrable, unchanged from the frozen path.
    assert tenant_group("https://mail.example.com/i") == "example.com"
    # PSL private-section carriers resolve tenant-level.
    assert tenant_group("https://evil.blogspot.com/") == "evil.blogspot.com"
    assert tenant_group("https://t.azurewebsites.net/") == "t.azurewebsites.net"
    # Absent from the private section: full host keeps tenants separate.
    assert tenant_group("https://login.core.windows.net/x") == (
        "login.core.windows.net"
    )
    assert tenant_group("https://a.core.windows.net/x") != tenant_group(
        "https://b.core.windows.net/x"
    )


def test_gate_joined_rows_band_scoped(tmp_path: Path) -> None:
    from phishnet.enrichment.join import gate_joined_rows

    rows = [
        {
            "cache_key": f"p{i}.com",
            "url": f"https://p{i}.com/",
            "label": "1",
            "survival_stratum": "fresh",
            "age_known": True,
            "age_na": False,
            "ct_known": True,
            "ct_na": False,
        }
        for i in range(30)
    ] + [
        {
            "cache_key": f"b{i}.com",
            "url": f"https://b{i}.com/",
            "label": "0",
            "survival_stratum": "na",
            "age_known": True,
            "age_na": False,
            "ct_known": True,
            "ct_na": False,
        }
        for i in range(30)
    ]
    bundle = gate_joined_rows(rows, max_unknown_gap=0.05, n_boot=100, seed=0)
    assert bundle["gate"]["verdict"] == "pass"
    assert bundle["gate"]["signals"]["age"]["eligible"] is True
    assert set(bundle["gap_cis"]) == {"age", "ct"}


def test_psl_check_offline_and_diffable() -> None:
    rep = check_psl_splits(["https://login.core.windows.net/x"])
    sha = cast("str", rep["psl_snapshot_sha256"])
    assert len(sha) == 64
    rows = cast("list[dict[str, Any]]", rep["rows"])
    assert rows[0]["hosted"] is True


def test_psl_gate_refuses_on_mismatch() -> None:
    try:
        gate_psl_snapshot("0" * 64)
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("expected SystemExit(1) on PSL mismatch")
    assert len(gate_psl_snapshot(None)) == 64  # record-but-pass


def test_hosted_share_reported() -> None:
    rep = hosted_share(["https://a.core.windows.net/x", "https://mail.example.com/i"])
    assert rep["n"] == 2 and rep["n_hosted"] == 1
    assert rep["hosted_share"] == 0.5


def test_snapshot_run_keyed_seal_and_pin(tmp_path: Path) -> None:
    """A later failed re-enrichment must not overwrite an earlier success."""
    path = tmp_path / "enrichment-test.jsonl"
    good = {
        "cache_key": "example.com",
        "age_known": True,
        "domain_age_days": 365.0,
        "enriched_at": "2026-09-16T00:00:00+00:00",
    }
    append_records(path, "run-1", [good])
    # Resume same run is a no-op for known keys.
    append_records(path, "run-1", [good])
    # Later run lapses: pure-unknown failure for the same key.
    bad = {"cache_key": "example.com", "enriched_at": "2026-10-16T00:00:00+00:00"}
    append_records(path, "run-2", [bad])
    seal_run(path, "run-1")
    seal_run(path, "run-2")
    pinned = load_pinned_run(path, "run-1")
    assert pinned["example.com"]["domain_age_days"] == 365.0
    # Fallback without a pin still resolves the earliest SUCCESS, not the
    # later failure.
    all_recs = [
        *load_pinned_run(path, "run-1").values(),
        *load_pinned_run(path, "run-2").values(),
    ]
    assert select_earliest_success(all_recs)["example.com"]["domain_age_days"] == 365.0


def test_na_unknown_rates_per_class_and_stratum() -> None:
    rows = [
        {
            "label": 1,
            "survival_stratum": "fresh",
            "age_known": True,
            "age_na": False,
            "ct_known": False,
            "ct_na": False,
        },
        {
            "label": 1,
            "survival_stratum": "fresh",
            "age_known": False,
            "age_na": True,
            "ct_known": False,
            "ct_na": True,
        },
        {
            "label": 0,
            "survival_stratum": "na",
            "age_known": True,
            "age_na": False,
            "ct_known": True,
            "ct_na": False,
        },
    ]
    rep = na_unknown_rates(rows)
    assert rep["age_by_label"]["1"]["na_rate"] == 0.5
    assert rep["age_by_label"]["0"]["na_rate"] == 0.0
    # na rows are excluded from the unknown denominator: the one eligible
    # phishing row is known, so unknown reads 0, not 0.5.
    assert rep["age_by_label"]["1"]["n_eligible"] == 1
    assert rep["age_by_label"]["1"]["unknown_rate"] == 0.0
    assert rep["ct_by_survival_stratum"]["fresh"]["unknown_rate"] == 1.0


def test_train_score_parity_with_canonicalize() -> None:
    """One decision, one switch: the training path and the scoring path
    featurize identically, and scheme variants converge through both."""
    import numpy as np

    from ml_training.train_gbm import featurise as train_featurise

    pred = predictors.LegacyEnsemble(canonicalize=True)
    urls = ["https://example.com/login?x=1", "http://192.168.1.1/admin"]
    train_frame = train_featurise(urls, pred.columns, canonicalize=True)
    score_frame = pred._features(urls)
    # Scoring standardizes; training (GBM lineage) uses native units —
    # parity is proven through the inverse scaler, not raw equality.
    unscaled = pred.scaler.inverse_transform(score_frame)
    np.testing.assert_allclose(
        train_frame.to_numpy(dtype=float), unscaled, rtol=1e-9, atol=1e-9
    )
    # Scheme variants converge on both paths; default paths keep the leak.
    for featurise in (
        lambda u: train_featurise([u], pred.columns, canonicalize=True),
        lambda u: predictors.LegacyEnsemble(canonicalize=True)._features([u]),
    ):
        a = np.asarray(featurise("http://example.com/login"), dtype=float)
        b = np.asarray(featurise("https://example.com/login"), dtype=float)
        np.testing.assert_array_equal(a, b)
    plain = predictors.LegacyEnsemble()
    assert not np.array_equal(
        plain._features(["http://example.com/login"]),
        plain._features(["https://example.com/login"]),
    )


def test_join_single_rule_recorded_and_gate_on_same_rows(
    tmp_path: Path,
) -> None:
    """The join uses exactly one rule, records it, and gates on its rows."""
    from phishnet.enrichment.join import join_enrichment

    snap = tmp_path / "enrichment-join.jsonl"
    append_records(
        snap,
        "run-1",
        [
            {
                "cache_key": "example.com",
                "rdap": {
                    "creation_date": "2020-01-01T00:00:00+00:00",
                    "source": "rdap",
                },
                "ct": {"certs": [], "provider": "crt.sh-json"},
                "enriched_at": "2026-09-16T00:00:00+00:00",
            }
        ],
    )
    seal_run(snap, "run-1")
    rows = [
        {
            "url": "https://mail.example.com/i",
            "label": 1,
            "first_seen": "2026-09-15T08:00:00+00:00",
            "survival_stratum": "short",
        },
        {"url": "https://absent.example/x", "label": 0},
    ]
    joined, manifest = join_enrichment(
        rows, snap, {"rule": "pinned-run", "run_id": "run-1"}
    )
    assert manifest["selection_rule"] == "pinned-run"
    assert manifest["run_id"] == "run-1"
    assert manifest["n_keys_known"] == 1
    assert joined[0]["age_known"] is True
    # 2020-01-01 -> 2026-09-15 08:00 UTC, fractional days preserved.
    assert joined[0]["domain_age_days"] == 2449.0 + 8.0 / 24.0
    assert joined[0]["ct_known"] is True  # empty history is a real answer
    assert joined[0]["ct_cert_count_pre"] == 0
    assert joined[1]["age_known"] is False  # unknown key
    # The contamination block is computed on the joined rows themselves.
    assert manifest["contamination"]["age_by_label"]["1"]["n"] == 1
    assert manifest["contamination"]["age_by_label"]["0"]["n"] == 1
    try:
        join_enrichment(rows, snap, {"rule": "both"})
    except ValueError as e:
        assert "exactly one rule" in str(e)
    else:
        raise AssertionError("expected ValueError for mixed rule")


def test_join_derivation_fail_closed_cases() -> None:
    """Truncated CT, future creation dates, and hosted rows resolve safe."""
    from phishnet.enrichment.join import derive_row

    # Truncated history cannot prove youth: unknown, not young.
    r = derive_row(
        "https://example.com/",
        {"ct": {"certs": [], "provider": "crt.sh-json", "truncated": True}},
        "2026-09-15T00:00:00+00:00",
        False,
    )
    assert r["ct_known"] is False
    # Creation after observation: unknown, never a negative feature.
    r = derive_row(
        "https://example.com/",
        {"rdap": {"creation_date": "2027-01-01T00:00:00+00:00", "source": "rdap"}},
        "2026-09-15T00:00:00+00:00",
        False,
    )
    assert r["age_known"] is False and r["domain_age_days"] is None
    # Hosted tenant: na without any payload.
    r = derive_row("https://t.core.windows.net/", None, "2026-09-15", True)
    assert (r["age_na"], r["ct_na"]) == (True, True)
    assert (r["age_known"], r["ct_known"]) == (False, False)
    # CT certs after first_seen don't count; age is source-tagged.
    r = derive_row(
        "https://example.com/",
        {
            "rdap": {"creation_date": "2026-09-01T00:00:00+00:00", "source": "whois"},
            "ct": {
                "certs": [
                    {"entry_timestamp": "2026-09-20T00:00:00+00:00"},
                    {"entry_timestamp": "2026-09-10T00:00:00+00:00"},
                ],
                "provider": "crt.sh-json",
            },
        },
        "2026-09-15T00:00:00+00:00",
        False,
    )
    assert r["ct_cert_count_pre"] == 1 and r["ct_age_days"] == 5.0
    assert r["age_source"] == "whois" and r["domain_age_days"] == 14.0


def test_contamination_gate_needs_explicit_thresholds() -> None:
    from phishnet.enrichment.join import check_contamination

    even = {
        "age_by_label": {
            "1": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
            "0": {"n": 100, "unknown_rate": 0.12, "na_rate": 0.0},
        },
        "ct_by_label": {
            "1": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
            "0": {"n": 100, "unknown_rate": 0.12, "na_rate": 0.0},
        },
    }
    got = check_contamination(even, max_unknown_gap=0.05)
    assert got["verdict"] == "pass"
    assert got["signals"]["age"]["eligible"] is True
    assert got["signals"]["ct"]["eligible"] is True
    skewed = {
        "age_by_label": {
            "1": {"n": 100, "unknown_rate": 0.4, "na_rate": 0.0},
            "0": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
        },
        "ct_by_label": {
            "1": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
            "0": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
        },
    }
    got = check_contamination(skewed, max_unknown_gap=0.05)
    # Per-signal eligibility: the CT failure must not take age down too.
    assert got["verdict"] == "fail"
    assert got["signals"]["age"]["eligible"] is False
    assert got["signals"]["ct"]["eligible"] is True
    assert (
        check_contamination({"age_by_label": {}}, max_unknown_gap=0.05)["verdict"]
        == "unmeasurable"
    )
    # A point inside budget whose CI straddles the threshold is unresolved.
    straddled = check_contamination(
        even,
        max_unknown_gap=0.05,
        gap_cis={
            "age": {"unknown": (0.01, 0.09), "na": (0.0, 0.0)},
            "ct": {"unknown": (0.0, 0.01), "na": (0.0, 0.0)},
        },
    )
    assert straddled["signals"]["age"]["eligible"] is False
    assert "straddles" in straddled["signals"]["age"]["reason"]
    assert straddled["signals"]["ct"]["eligible"] is True
    # The na share is composition, not contamination: a large na gap
    # reports but never decides (Amendment A).
    na_heavy = {
        "age_by_label": {
            "1": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.45},
            "0": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
        },
        "ct_by_label": {
            "1": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.45},
            "0": {"n": 100, "unknown_rate": 0.1, "na_rate": 0.0},
        },
    }
    got = check_contamination(na_heavy, max_unknown_gap=0.05)
    assert got["verdict"] == "pass"
    assert got["signals"]["age"]["na_gap"] == 0.45


def test_gap_bootstrap_ci_clusters_by_key() -> None:
    from phishnet.enrichment.join import gap_bootstrap_ci

    rows = [
        {
            "cache_key": f"p{i}.com",
            "label": "1",
            "age_known": i % 2 == 0,
            "age_na": False,
        }
        for i in range(20)
    ] + [
        {"cache_key": f"b{i}.com", "label": "0", "age_known": True, "age_na": False}
        for i in range(20)
    ]
    lo, hi = gap_bootstrap_ci(rows, "age", "unknown", n_boot=200, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= 0.5 <= hi  # ~50% phish unknown vs 0% benign
    assert gap_bootstrap_ci([], "age", "unknown")[0] != 0.0  # nan pair
    try:
        gap_bootstrap_ci(rows, "age", "bogus")
    except ValueError as e:
        assert "unknown" in str(e) and "na" in str(e)
    else:
        raise AssertionError("expected ValueError for bad kind")


def test_resolve_canonicalize_follows_manifest(tmp_path: Path) -> None:
    from ml_training.train_gbm import resolve_canonicalize

    split = tmp_path / "split"
    split.mkdir()
    (split / "manifest.json").write_text(
        '{"is_https_rule": {"decision": "drop"}}', encoding="utf-8"
    )
    assert resolve_canonicalize(split, None) == (True, "manifest:drop")
    (split / "manifest.json").write_text(
        '{"is_https_rule": {"decision": "keep"}}', encoding="utf-8"
    )
    assert resolve_canonicalize(split, None) == (False, "manifest:keep")
    assert resolve_canonicalize(split, True) == (True, "flag")
    assert resolve_canonicalize(split, False) == (False, "flag")
    # Predates the rule, or unreadable manifest: Phase 2 behavior preserved.
    assert resolve_canonicalize(tmp_path, None) == (False, "absent-default")
    (split / "manifest.json").write_text("not json", encoding="utf-8")
    assert resolve_canonicalize(split, None) == (False, "absent-default")


class _DummyProbaModel:
    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        import numpy as np

        p = np.zeros((len(X), 2))
        p[:, 1] = 0.5
        return p


def _gbm_assets(tmp_path: Path, *, canonicalize: bool | None) -> Path:
    import pickle

    d = tmp_path / "gbm_assets"
    d.mkdir(exist_ok=True)
    with open(d / "gbm_model.pkl", "wb") as f:
        pickle.dump(_DummyProbaModel(), f)
    real_cols: list[str] = list(
        __import__("pickle").loads(
            Path("src/phishnet/urlset_ml_assets/feature_columns.pkl").read_bytes()
        )
    )
    with open(d / "feature_columns.pkl", "wb") as f:
        pickle.dump(real_cols, f)
    if canonicalize is not None:
        from ml_training.train_gbm import write_train_config

        write_train_config(
            d,
            canonicalize=canonicalize,
            scheme_source="manifest:drop" if canonicalize else "manifest:keep",
        )
    return d


def test_predictor_follows_sidecar_and_fingerprints_it(
    tmp_path: Path,
) -> None:
    # No sidecar (pre-Phase-3 assets): False, Phase 2 behavior preserved.
    plain = predictors.GbmSingle(
        assets_dir=str(_gbm_assets(tmp_path, canonicalize=None))
    )
    assert plain.canonicalize is False
    assert plain.asset_fingerprint["canonicalize_scheme"] == "false"
    assert plain.scheme_source == "absent-default"
    # Sidecar DROP: scoring follows training without any flag.
    drop = predictors.GbmSingle(
        assets_dir=str(_gbm_assets(tmp_path, canonicalize=True))
    )
    assert drop.canonicalize is True
    assert drop.asset_fingerprint["canonicalize_scheme"] == "true"
    assert drop.scheme_source == "manifest:drop"
    # Explicit flag wins over the sidecar either way.
    assert (
        predictors.GbmSingle(
            assets_dir=str(_gbm_assets(tmp_path, canonicalize=True)),
            canonicalize=False,
        ).canonicalize
        is False
    )


def test_native_path_parity_with_canonicalize() -> None:
    """The champion runs scaler-less: same parity check, native branch."""
    import numpy as np

    from ml_training.train_gbm import featurise as train_featurise

    pred = predictors.LegacyEnsemble(canonicalize=True)
    pred.scaler = None  # the GBM native-units branch subclasses inherit
    urls = ["https://example.com/login?x=1", "http://192.168.1.1/admin"]
    np.testing.assert_array_equal(
        train_featurise(urls, pred.columns, canonicalize=True).to_numpy(dtype=float),
        pred._features(urls),
    )


def test_measure_type_targets_pins_inputs(tmp_path: Path) -> None:
    """Quota shares are measured from recorded phishing files, never lore."""
    from build_cc_benign import measure_type_targets

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "openphish-2026-09-12.jsonl").write_text(
        '{"url": "http://a.example/"}\n{"url": "http://b.example/x"}\n',
        encoding="utf-8",
    )
    (raw / "phishtank-2026-09-12.jsonl").write_text(
        '{"url": "http://a.example/"}\n'  # duplicate across feeds: once
        '{"url": "http://c.example/x/y?q=1"}\n',
        encoding="utf-8",
    )
    (raw / "benign-2026-09-12.jsonl").write_text(
        '{"url": "http://benign.example/"}\n', encoding="utf-8"
    )
    shares, inputs = measure_type_targets(raw)
    assert shares == {"path1": 1 / 3, "query": 1 / 3, "root": 1 / 3}
    assert set(inputs["files"]) == {
        "openphish-2026-09-12.jsonl",
        "phishtank-2026-09-12.jsonl",
    }
    assert all(len(h) == 64 for h in inputs["files"].values())
    assert inputs["n_dedup_urls"] == 3
    assert inputs["mode"] == "measured"


def test_phase3_power_option_fires_by_rule() -> None:
    assert build_splits.phase3_power_option(20_000) == "option-1"
    assert build_splits.phase3_power_option(15_000) == "option-1"  # boundary passes
    assert build_splits.phase3_power_option(14_999) == "option-2-fallback"
    assert build_splits.PHASE3_BENIGN_TEST_FLOOR == 15_000


def test_champion_untouched_and_subclass_opt_in() -> None:
    """Phase 2 GbmRefit carries no stub; the enriched subclass opts in."""
    assert not hasattr(predictors.GbmRefit, "enrichment_provider")
    assert getattr(predictors.GbmRefitWithEnrichment, "model_filename", "") == (
        "refit_base.pkl"
    )
    sub = predictors.GbmRefitWithEnrichment.__new__(predictors.GbmRefitWithEnrichment)
    sub.enrichment_provider = UnknownStubProvider()
    recs = sub.enrichment_provider.lookup_many(["https://example.com/"])
    assert recs[0].age_known is False


def test_stub_unknown_and_hosted_na() -> None:
    stub = UnknownStubProvider()
    r = stub.lookup("https://mail.example.com/inbox")
    assert r.age_known is False and r.ct_known is False
    assert r.age_na is False and r.ct_na is False
    h = stub.lookup("https://login.core.windows.net/tenant")
    assert h.age_na is True and h.ct_na is True
    assert h.age_known is False and h.ct_known is False


def test_stub_equals_forced_miss_grouped_by_key() -> None:
    stub = UnknownStubProvider()
    urls = [
        "https://a.core.windows.net/x",
        "https://a.core.windows.net/y",  # same key: one miss covers both
        "https://mail.example.com/inbox",
    ]
    recs = [
        EnrichedRecord(
            cache_key="a.core.windows.net",
            age_na=True,
            ct_na=True,
        ),
        EnrichedRecord(
            cache_key="a.core.windows.net",
            age_na=True,
            ct_na=True,
        ),
        EnrichedRecord(
            cache_key="example.com",
            domain_age_days=9.0,
            age_known=True,
            ct_known=True,
        ),
    ]
    missed = force_miss(recs)
    stubbed = stub.lookup_many(urls)
    assert [(m.cache_key, m.age_known, m.age_na) for m in missed] == [
        (s.cache_key, s.age_known, s.age_na) for s in stubbed
    ]


def test_ct_filter_runs_at_join_per_row_on_entry_time() -> None:
    # Filters on CT log entry time, NOT not_before (backdatable): only the
    # entry time says the cert was publicly visible before first_seen.
    certs = [
        {
            "entry_timestamp": "2026-09-01T00:00:00+00:00",
            "not_before": "2020-01-01T00:00:00+00:00",  # backdated: ignored
        },
        {
            "entry_timestamp": "2026-09-20T00:00:00+00:00",
            "not_before": "2026-09-01T00:00:00+00:00",
        },
        {"entry_timestamp": "garbage"},
        {"not_before": "2026-09-01T00:00:00+00:00"},  # no entry time: dropped
    ]
    # Same domain, two rows with different first_seen: filter differs per row.
    assert len(pre_first_seen_filter(certs, "2026-09-10T00:00:00+00:00")) == 1
    assert len(pre_first_seen_filter(certs, "2026-09-25T00:00:00+00:00")) == 2


def test_scheme_canonicalization_removes_length_signal() -> None:
    """https:// is a char longer than http://: raw featurizing leaks scheme
    through url_length and friends, so DROP means canonicalize, not
    drop-one-column."""
    from phishnet.features.extraction import (
        canonicalize_scheme,
        comprehensive_phishing_features,
    )

    assert canonicalize_scheme("https://example.com/login") == "example.com/login"
    assert canonicalize_scheme("http://example.com/login") == "example.com/login"
    raw_http = comprehensive_phishing_features("http://example.com/login")
    raw_https = comprehensive_phishing_features("https://example.com/login")
    assert raw_http["url_length"] != raw_https["url_length"]  # the leak
    canon_http = comprehensive_phishing_features(
        canonicalize_scheme("http://example.com/login")
    )
    canon_https = comprehensive_phishing_features(
        canonicalize_scheme("https://example.com/login")
    )
    assert canon_http["url_length"] == canon_https["url_length"]
    assert canon_http["is_https"] == canon_https["is_https"] == 0


def test_scheme_gap_measured_on_train() -> None:
    """The vocabulary decision must not see the test set."""
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/",
                "https://b.example/",
                "http://c.example/",
                "http://evil.example/x",
            ],
            "label": [0, 0, 1, 1],
        }
    )
    train = df.iloc[[0, 3]]  # benign https, phish http -> gap 1.0 on train
    b_rate, p_rate = build_splits.scheme_rates(train)
    assert (b_rate, p_rate) == (1.0, 0.0)
    assert build_splits.should_drop_is_https(b_rate, p_rate) is True


def test_enrich_byte_determinism() -> None:
    """Building the enriched frame twice yields identical bytes."""
    df = pd.DataFrame(
        [
            {
                "url": "http://example.com/a?x=1",
                "label": 1,
                "first_seen": "2026-09-10 08:00:00+00:00",
                "first_snapshot": "2026-09-12",
                "source": "phishtank",
                "time_basis": "submitted",
                "_snap_file": "phishtank-2026-09-12.jsonl",
            },
            {
                "url": "https://benign.example/b",
                "label": 0,
                "first_seen": "2026-09-13 08:00:00+00:00",
                "first_snapshot": "2026-09-13",
                "source": "tranco:x",
                "time_basis": "crawled",
                "_snap_file": "benign-2026-09-13.jsonl",
            },
        ]
    )
    cols = [
        "url",
        "label",
        "first_seen",
        "first_snapshot",
        "survival_stratum",
        "registrable_domain",
        "suffix",
        "source",
    ]
    first = build_splits.enrich(df, phase3=True)[cols].to_csv(
        index=False, lineterminator="\r\n"
    )
    second = build_splits.enrich(df, phase3=True)[cols].to_csv(
        index=False, lineterminator="\r\n"
    )
    assert first == second
    assert "survival_stratum" in first and "first_snapshot" in first
