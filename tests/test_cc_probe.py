"""Offline tests for the Amendment D COUNT probe (M3) — no AWS, no boto3.

Covers the read-only guarantees and the decision inputs with synthetic
inputs only:
* the COUNT SQL pins crawl/subset/registered-domain/apex-host/200 plus
  exact root-URL equalities (no LIKE wildcards), and escapes quotes;
* candidate replay matches fetch's seeded order (one default_rng stream,
  one permutation per stratum in STRATA order), excludes definitive
  outcomes only, and never consults the network;
* tenant-skipping, the per-domain root cap, the bar decision, and the
  report/cache write guards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import build_cc_benign as B
import probe_cc_roots as P


def _mapping(n: int) -> dict[int, str]:
    return {r: f"d{r}.example" for r in range(1, n + 1)}


def test_sql_pins_predicates_and_escapes() -> None:
    sql = P.render_apex_root_count_sql("ccindex", "CC-MAIN-2026-34", "example.com")
    assert "COUNT(DISTINCT url)" in sql
    assert "crawl = 'CC-MAIN-2026-34'" in sql
    assert "subset = 'warc'" in sql
    assert "url_host_registered_domain = 'example.com'" in sql
    assert "url_host_name = 'example.com'" in sql
    assert "fetch_status = 200" in sql
    assert "LIKE" not in sql
    for root in (
        "http://example.com/",
        "https://example.com/",
        "http://example.com",
        "https://example.com",
    ):
        assert f"url = '{root}'" in sql
    evil = P.render_apex_root_count_sql("t", "c", "o'brien.example")
    assert "o''brien.example" in evil
    assert "o'brien.example" not in evil.replace("o''brien.example", "")


def test_replay_matches_fetch_seeded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny = {"s1x": (1, 5), "s4x": (6, 12)}
    monkeypatch.setattr(B, "STRATA", tiny)
    mapping = _mapping(12)
    # Independent replay of cmd_fetch's rng consumption: s1x's permutation
    # is drawn and discarded so the s4x order matches a live resume.
    rng = np.random.default_rng(0)
    pool1 = [mapping[r] for r in range(1, 6)]
    _ = rng.permutation(len(pool1))
    pool4 = [mapping[r] for r in range(6, 13)]
    expect = [pool4[int(i)] for i in rng.permutation(len(pool4))]
    cache: list[dict[str, Any]] = [
        {
            "domain": expect[0],
            "stratum": "s4x",
            "index": "CC-MAIN-2026-34",
            "note": "ok",
        },
        {
            "domain": expect[1],
            "stratum": "s4x",
            "index": None,
            "note": "no-usable-captures",
        },
        {"domain": expect[2], "stratum": "s4x", "index": None, "note": "error:boom"},
    ]
    fresh = P.replay_fresh(mapping, cache, 0, ["s4x"])
    got = [d for _, _, d in fresh]
    # Definitive (success + hard miss) excluded; transient kept for retry.
    assert got == [d for d in expect if d not in {expect[0], expect[1]}]
    assert expect[2] in got


def test_replay_seed_mismatch_exits(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"seed": 0, "domains": []}), encoding="utf-8")
    report = tmp_path / "rep.json"
    with pytest.raises(SystemExit):
        P.main(["--cache", str(cache), "--seed", "7", "--report", str(report)])


def test_report_must_not_be_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"seed": 0, "domains": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        P.main(["--cache", str(cache), "--report", str(cache)])


def test_cap_sum_and_decision() -> None:
    assert P.cap_sum([0, 3, 9, 6], 6) == 0 + 3 + 6 + 6
    assert P.decide(5900, 5900) == "D1"  # boundary counts
    assert P.decide(5899.9, 5900) == "D2"


def test_projection_math() -> None:
    proj = P.project_yield(600, 100, 1000, 5900)
    assert proj["rate_per_domain"] == pytest.approx(6.0)
    assert proj["projected_total"] == pytest.approx(600 + 6.0 * 900)
    assert proj["required_rate_on_remainder"] == pytest.approx((5900 - 600) / 900)


def test_tenant_skip() -> None:
    fresh = [
        ("s4_1k_10k", 1001, "evil-example.com"),
        ("s4_1k_10k", 1002, "clean-example.com"),
    ]
    kept, skipped = P.split_tenant_skipped(fresh, {"evil-example.com"})
    assert [d for _, _, d in kept] == ["clean-example.com"]
    assert [d for _, _, d in skipped] == ["evil-example.com"]


def test_pinned_list_matches_d01() -> None:
    assert P.PINNED_PHISH_FILES == sorted(P.PINNED_PHISH_FILES)
    assert len(P.PINNED_PHISH_FILES) == 10
    assert P.PINNED_PHISH_FILES[0] == "openphish-2026-09-12.jsonl"
    assert P.PINNED_PHISH_FILES[-1] == "phishtank-2026-09-16.jsonl"


def test_pinned_tenant_set_missing_file_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        P.pinned_tenant_set(tmp_path, ["openphish-2026-09-12.jsonl"])


def test_parse_single_count() -> None:
    assert P.parse_single_count([{"_col0": "42"}]) == 42
    with pytest.raises(ValueError):
        P.parse_single_count([])
