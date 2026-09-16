"""Tests for the Common-Crawl benign selector (offline, no network).

Covers the v2 sampling design in ``build_cc_benign.cmd_select`` using
small synthetic caches:
* per-type quotas fill from empirical index records first;
* root backfill synthesises at most one apex root per scheme actually
  observed for that domain (never a default scheme — no https fallback);
* host-form rule: apex only — www evidence never backs an apex root
  (domain skipped and counted), a www empirical root blocks apex
  synthesis via the apex-based dedup key, non-apex seeds are skipped;
* an empirical root record always wins over a synthesised duplicate;
* provenance records the synthesis rule, scheme evidence and counts.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import build_cc_benign as B


def _entry(
    domain: str,
    records: list[tuple[str, ...]],
    stratum: str = "s5_10k_100k",
    rank: int = 15000,
    index: str | None = None,
) -> dict[str, Any]:
    slim = []
    for i, r in enumerate(records):
        digest = r[2] if len(r) > 2 else f"d-{domain}-{i}"
        slim.append(
            {
                "url": r[0],
                "timestamp": r[1],
                "digest": digest,
                "mime": "text/html",
                "status": "200",
            }
        )
    return {
        "domain": domain,
        "index": index or B.CC_INDEX_PRIMARY,
        "http_status": 200,
        "n_records": len(records),
        "note": "ok",
        "attempts": 1,
        "queried_at": "2026-09-13T00:00:00+00:00",
        "stratum": stratum,
        "rank": rank,
        "records": slim,
    }


def _run_select(
    tmp_path: Path, domains: list[dict[str, Any]], **kw: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = {
        "seed": 0,
        "cc_index_primary": B.CC_INDEX_PRIMARY,
        "cc_index_fallback": B.CC_INDEX_FALLBACK,
        "tranco_csv": str(B.TRANC0_CSV),
        "tranco_sha256": B.TRANC0_SHA256,
        "domains": domains,
    }
    cp = tmp_path / "cache.json"
    cp.write_text(json.dumps(cache), encoding="utf-8")
    out = tmp_path / "benign.jsonl"
    a = Namespace(
        seed=0,
        target_n=kw.get("target_n", 8),
        productive_per_stratum=500,
        cache=str(cp),
        out=str(out),
        collapse_digest=kw.get("collapse_digest", False),
        measure_quotas_from=kw.get("measure_quotas_from"),
        stratified_quotas=kw.get("stratified_quotas", False),
        quota_files=kw.get("quota_files"),
        exclude_phishing_tenants_from=kw.get("exclude_from"),
        require_multi_crawl=kw.get("multi_crawl", False),
    )
    assert B.cmd_select(a) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    prov = json.loads(Path(str(out) + ".provenance.json").read_text(encoding="utf-8"))
    return rows, prov


def test_synthesis_uses_only_observed_schemes(tmp_path: Path) -> None:
    """https-only evidence -> https root; http-only -> http root (no default)."""
    rows, _ = _run_select(
        tmp_path,
        [
            _entry(
                "ex-c.com", [("https://ex-c.com/x/y", "20260807104456")], rank=200000
            ),
            _entry(
                "ex-h.com", [("http://ex-h.com/old", "20260807104456")], rank=300000
            ),
        ],
    )
    by_url = {r["url"]: r for r in rows}
    assert by_url["https://ex-c.com/"].get("synthesized_root") is True
    assert by_url["http://ex-h.com/"].get("synthesized_root") is True
    assert "https://ex-h.com/" not in by_url  # never default to https


def test_empirical_root_wins_over_synthesised_duplicate(tmp_path: Path) -> None:
    rows, prov = _run_select(
        tmp_path,
        [_entry("ex-a.com", [("https://ex-a.com/", "20260807104456")])],
    )
    by_url = {r["url"]: r for r in rows}
    assert "synthesized_root" not in by_url["https://ex-a.com/"]
    assert by_url["https://ex-a.com/"]["time_basis"] == "commoncrawl-index"
    assert prov["root_synthesis"]["skipped_duplicate"] >= 1


def test_provenance_records_synthesis_rule_and_counts(tmp_path: Path) -> None:
    rows, prov = _run_select(
        tmp_path,
        [_entry("ex-c.com", [("https://ex-c.com/x/y", "20260807104456")], rank=200000)],
    )
    synth = prov["root_synthesis"]
    assert "never a default scheme" in synth["rule"]
    assert synth["added"] >= 1
    # These synthetic caches predate per-domain scheme evidence, so the
    # backfill falls back to the 200-only records and says so.
    assert synth["domains_evidence_fallback_200_only"] >= 1
    assert synth["n_synthesized_roots_selected"] == sum(
        1 for r in rows if r.get("synthesized_root")
    )
    assert prov["fetch_mechanism"] == "cdx"
    assert prov["fetch_mechanism_detail"] == {"cc_query_form": B.QUERY_FORM}
    assert prov["productive_per_stratum_target"] == 500
    synth_row = next(r for r in rows if r.get("synthesized_root"))
    assert synth_row["scheme_evidence"] == {"https": 1}
    assert synth_row["time_basis"] == "domain-inferred-root"


def test_www_only_evidence_skips_domain_and_counts(tmp_path: Path) -> None:
    """Records on www/sub hosts only (no apex capture): no synthesis, counted."""
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "ex-w.com",
                [
                    ("https://www.ex-w.com/", "20260807104456"),
                    ("https://blog.ex-w.com/x", "20260807104457"),
                ],
            )
        ],
    )
    by_url = {r["url"]: r for r in rows}
    assert "https://ex-w.com/" not in by_url  # no www fallback
    assert "http://ex-w.com/" not in by_url
    assert prov["root_synthesis"]["domains_no_apex_capture"] == 1
    assert prov["root_synthesis"]["added"] == 0
    # The www URLs themselves still select normally as empirical rows.
    assert "https://www.ex-w.com/" in by_url


def test_www_empirical_root_blocks_apex_synthesis(tmp_path: Path) -> None:
    """Apex-key dedup: www empirical root blocks the apex synthesis."""
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "ex-d.com",
                [
                    ("https://www.ex-d.com/", "20260807104456"),
                    ("https://ex-d.com/deep", "20260807104457"),
                ],
            )
        ],
    )
    by_url = {r["url"]: r for r in rows}
    assert "synthesized_root" not in by_url["https://www.ex-d.com/"]
    assert "https://ex-d.com/" not in by_url  # blocked, not duplicated
    assert prov["root_synthesis"]["skipped_duplicate"] >= 1


def test_non_apex_seed_skipped_and_counted(tmp_path: Path) -> None:
    """A seed that is not a bare registrable domain is never synthesised."""
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "sub.ex-e.com",
                [("https://sub.ex-e.com/x", "20260807104456")],
            )
        ],
    )
    assert not [r for r in rows if r.get("synthesized_root")]
    assert prov["root_synthesis"]["domains_non_apex_seed_skipped"] == 1


def test_www_apex_variants_collapse_before_quotas(tmp_path: Path) -> None:
    """www + apex captures of one root collapse to one candidate (earliest)."""
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "ex-f.com",
                [
                    ("https://www.ex-f.com/", "20260807104458"),
                    ("https://ex-f.com/", "20260807104456"),
                ],
            )
        ],
        target_n=4,
    )
    roots = [r for r in rows if r["url_type"] == "root"]
    assert [r["url"] for r in roots] == ["https://ex-f.com/"]
    assert roots[0]["first_seen"].startswith("2026-08-07T10:44:56")
    stats = prov["candidate_stats"]
    assert stats["duplicates"] >= 1


def test_scheme_and_query_preserved_in_dedup_key(tmp_path: Path) -> None:
    """http/https and distinct queries are different URLs, not duplicates."""
    rows, _ = _run_select(
        tmp_path,
        [
            _entry(
                "ex-g.com",
                [
                    ("https://ex-g.com/a?x=1", "20260807104456"),
                    ("https://ex-g.com/a?x=2", "20260807104456"),
                    ("http://ex-g.com/a?x=1", "20260807104456"),
                ],
            )
        ],
        # Query quota must fit all three candidates: this tests dedup,
        # not quota selection.
        target_n=24,
    )
    queries = sorted(r["url"] for r in rows if r["url_type"] == "query")
    assert queries == [
        "http://ex-g.com/a?x=1",
        "https://ex-g.com/a?x=1",
        "https://ex-g.com/a?x=2",
    ]


def test_digest_collapse_off_by_default_on_by_flag(tmp_path: Path) -> None:
    """Same digest, different URLs: kept by default, collapsed with the flag."""
    domains = [
        _entry(
            "ex-h.com",
            [
                ("https://ex-h.com/a?utm=1", "20260807104456", "same-bytes"),
                ("https://ex-h.com/a?utm=2", "20260807104457", "same-bytes"),
            ],
        )
    ]
    rows_off, prov_off = _run_select(tmp_path, domains, target_n=12)
    assert sorted(r["url"] for r in rows_off if r["url_type"] == "query") == [
        "https://ex-h.com/a?utm=1",
        "https://ex-h.com/a?utm=2",
    ]
    assert prov_off["candidate_stats"]["digest_duplicates"] == 0
    assert prov_off["dedup"]["digest_collapse"] is False

    rows_on, prov_on = _run_select(tmp_path, domains, target_n=12, collapse_digest=True)
    assert [r["url"] for r in rows_on if r["url_type"] == "query"] == [
        "https://ex-h.com/a?utm=1"
    ]
    assert prov_on["candidate_stats"]["digest_duplicates"] == 1


def _write_phish_raw(raw: Path, urls: list[str]) -> None:
    (raw / "phishtank-2026-09-12.jsonl").write_text(
        "\n".join(json.dumps({"url": u}) for u in urls), encoding="utf-8"
    )


def test_phishing_tenant_exclusion_is_tenant_level(tmp_path: Path) -> None:
    """A benign capture on a phishing tenant drops even when the exact URL
    never appeared in a feed; clean tenants on other hosts survive."""
    raw = tmp_path / "praw"
    raw.mkdir()
    _write_phish_raw(raw, ["https://evil.blogspot.com/unrelated"])
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "seed-a.com",
                [
                    ("https://evil.blogspot.com/x", "20260807104456"),
                    ("https://seed-a.com/y", "20260807104456"),
                ],
            )
        ],
        exclude_from=str(raw),
    )
    by_url = {r["url"]: r for r in rows}
    assert "https://evil.blogspot.com/x" not in by_url  # tenant match, URL differs
    assert "https://seed-a.com/y" in by_url
    excl = prov["phishing_tenant_exclusion"]
    assert excl["enabled"] is True
    assert sum(excl["excluded_by_type"].values()) >= 1
    assert set(excl["inputs"]["files"]) == {"phishtank-2026-09-12.jsonl"}
    assert excl["inputs"]["n_tenants"] >= 1


def test_phishing_tenant_exclusion_off_by_default(tmp_path: Path) -> None:
    raw = tmp_path / "praw"
    raw.mkdir()
    _write_phish_raw(raw, ["https://evil.blogspot.com/unrelated"])
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "seed-a.com",
                [("https://evil.blogspot.com/x", "20260807104456")],
            )
        ],
    )
    assert "https://evil.blogspot.com/x" in {r["url"] for r in rows}
    assert prov["phishing_tenant_exclusion"]["enabled"] is False


def test_require_multi_crawl_keeps_durable_tenants(tmp_path: Path) -> None:
    """Tenants seen in one crawl drop; tenants in two crawls survive."""
    rows, prov = _run_select(
        tmp_path,
        [
            _entry(
                "once-a.com",
                [("https://once-a.com/a", "20260807104456")],
                index=B.CC_INDEX_PRIMARY,
            ),
            _entry(
                "twice-b.com",
                [("https://twice-b.com/a", "20260807104456")],
                index=B.CC_INDEX_PRIMARY,
            ),
            _entry(
                "twice-b.com",
                [("https://twice-b.com/b", "20260807104457")],
                index=B.CC_INDEX_FALLBACK,
            ),
        ],
        multi_crawl=True,
    )
    by_url = {r["url"]: r for r in rows}
    assert "https://once-a.com/a" not in by_url
    assert "https://twice-b.com/a" in by_url
    assert "https://twice-b.com/b" in by_url
    assert prov["multi_crawl"]["required"] is True
    assert sum(prov["multi_crawl"]["excluded_by_type"].values()) >= 1


def _write_phish_named(raw: Path, name: str, urls: list[str]) -> None:
    (raw / name).write_text(
        "\n".join(json.dumps({"url": u}) for u in urls), encoding="utf-8"
    )


def test_stratified_quotas_exclude_hosted(tmp_path: Path) -> None:
    """Non-hosted measurement drops hosted-tenant URLs after dedup."""
    raw = tmp_path / "qraw"
    raw.mkdir()
    _write_phish_named(
        raw,
        "openphish-2026-09-12.jsonl",
        [
            "https://b0.example.com/",
            "https://b1.example.com/a",
            "https://t0.vercel.app/",
            "https://t1.vercel.app/",
        ],
    )
    shares, inputs = B.measure_type_targets(raw, nonhosted=True)
    assert shares == {"path1": 0.5, "root": 0.5}
    assert inputs["stratum"] == "main-nonhosted"
    assert inputs["n_excluded_hosted"] == 2
    assert inputs["n_dedup_urls"] == 4
    shares_all, inputs_all = B.measure_type_targets(raw)
    assert shares_all["root"] == pytest.approx(0.75)
    # Default path keeps the historical provenance shape byte-identical.
    assert "stratum" not in inputs_all
    assert "n_excluded_hosted" not in inputs_all


def test_quota_files_pin_and_missing_exits(tmp_path: Path) -> None:
    raw = tmp_path / "qraw"
    raw.mkdir()
    _write_phish_named(raw, "openphish-2026-09-12.jsonl", ["https://a.example.com/"])
    _write_phish_named(raw, "phishtank-2026-09-12.jsonl", ["https://b.example.net/a"])
    shares, inputs = B.measure_type_targets(raw, files=["openphish-2026-09-12.jsonl"])
    assert set(inputs["files"]) == {"openphish-2026-09-12.jsonl"}
    assert shares == {"root": 1.0}
    with pytest.raises(SystemExit):
        B.measure_type_targets(raw, files=["openphish-2026-09-13.jsonl"])


def test_stratified_quotas_end_to_end(tmp_path: Path) -> None:
    """Select with --stratified-quotas records the non-hosted mix in provenance."""
    raw = tmp_path / "qraw"
    raw.mkdir()
    _write_phish_named(
        raw,
        "openphish-2026-09-12.jsonl",
        ["https://b0.example.com/a", "https://t0.vercel.app/"],
    )
    rows, prov = _run_select(
        tmp_path,
        [_entry("seed-a.com", [("https://seed-a.com/x", "20260807104456")])],
        target_n=4,
        measure_quotas_from=str(raw),
        stratified_quotas=True,
    )
    assert rows  # quotas still fill from the pool
    qi = prov["quota_inputs"]
    assert qi["stratum"] == "main-nonhosted"
    assert qi["shares"] == {"path1": 1.0, "pathN": 0.0, "query": 0.0, "root": 0.0}
    assert prov["type_quotas"] == {"path1": 4, "pathN": 0, "query": 0, "root": 0}
