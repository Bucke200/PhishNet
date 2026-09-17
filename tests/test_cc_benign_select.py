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
        length_bands=kw.get("length_bands", False),
        domain_cap=kw.get("domain_cap"),
        wave_manifest=kw.get("wave_manifest"),
        exclude_phishing_tenants_from=kw.get("exclude_from"),
        phishing_tenant_files=kw.get("tenant_files"),
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


def _rooty_entry(domain: str, rank: int) -> dict[str, Any]:
    """Four dedup-distinct roots: both apex schemes plus two subdomains."""
    return _entry(
        domain,
        [
            (f"https://{domain}/", "20260807104456"),
            (f"http://{domain}/", "20260807104457"),
            (f"https://a.{domain}/", "20260807104458"),
            (f"https://b.{domain}/", "20260807104459"),
        ],
        stratum="s4x",
        rank=rank,
    )


def test_fill_consumes_domains_in_seeded_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-quota pools fill from the earliest replay domains, then stop.

    D0.7.2 invariant: with 12 root rows behind a root quota of 4, the
    taken set is exactly the first replay domain's rows — later domains
    are never considered, so pool size cannot steer selection.
    """
    import numpy as np

    tiny = {"s4x": (1, 3)}
    mapping = {1: "ex-a.com", 2: "ex-b.com", 3: "ex-c.com"}
    monkeypatch.setattr(B, "STRATA", tiny)
    monkeypatch.setattr(B, "load_tranco", lambda: dict(mapping))
    pool = [mapping[r] for r in range(1, 4)]
    order = list(np.random.default_rng(0).permutation(len(pool)))
    first = pool[int(order[0])]
    rows, prov = _run_select(
        tmp_path,
        [
            _rooty_entry("ex-a.com", 1),
            _rooty_entry("ex-b.com", 2),
            _rooty_entry("ex-c.com", 3),
        ],
        target_n=12,
    )
    assert prov["type_quotas"]["root"] == 4
    taken = {r["url"] for r in rows if r["url_type"] == "root"}
    assert taken == {
        f"https://{first}/",
        f"http://{first}/",
        f"https://a.{first}/",
        f"https://b.{first}/",
    }


def test_fill_is_byte_identical_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeded-order fill reproduces byte-for-byte."""
    tiny = {"s4x": (1, 3)}
    mapping = {1: "ex-a.com", 2: "ex-b.com", 3: "ex-c.com"}
    monkeypatch.setattr(B, "STRATA", tiny)
    monkeypatch.setattr(B, "load_tranco", lambda: dict(mapping))
    domains = [
        _rooty_entry("ex-a.com", 1),
        _rooty_entry("ex-b.com", 2),
        _rooty_entry("ex-c.com", 3),
    ]
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    rows1, _ = _run_select(tmp_path / "r1", domains, target_n=12)
    rows2, _ = _run_select(tmp_path / "r2", domains, target_n=12)
    assert rows1 == rows2


def test_length_bands_measure_quartiles(tmp_path: Path) -> None:
    raw = tmp_path / "lraw"
    raw.mkdir()
    urls = [f"https://h{i}.example.com/p{i}" for i in range(8)]
    urls += [f"https://g{i}.example.com/p{i}/q{i}" for i in range(8)]
    urls += ["https://r.example.com/", "https://q.example.com/?x=1"]
    urls += ["https://t0.vercel.app/should-be-excluded"]
    _write_phish_named(raw, "openphish-2026-09-12.jsonl", urls)
    edges, inputs = B.measure_length_bands(raw, ["openphish-2026-09-12.jsonl"])
    assert inputs["stratum"] == "main-nonhosted"
    assert inputs["n_dedup_urls"] == 19
    assert sorted(edges) == ["path1", "pathN", "query", "root"]
    assert len(edges["path1"]) == 3
    assert edges["path1"] == sorted(edges["path1"])
    with pytest.raises(SystemExit):
        B.measure_length_bands(raw, ["openphish-2026-09-13.jsonl"])


def test_length_bands_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "lraw"
    raw.mkdir()
    _write_phish_named(
        raw,
        "openphish-2026-09-12.jsonl",
        [
            "https://b0.example.com/a",
            "https://b1.example.com/",
            "https://b2.example.com/a/b",
            "https://b3.example.com/?x=1",
        ],
    )
    rows, prov = _run_select(
        tmp_path,
        [
            _entry("seed-a.com", [("https://seed-a.com/x", "20260807104456")]),
            _entry("seed-b.com", [("https://seed-b.com/", "20260807104456")]),
        ],
        target_n=4,
        measure_quotas_from=str(raw),
        quota_files="openphish-2026-09-12.jsonl",
        length_bands=True,
    )
    assert prov["length_bands"]["enabled"] is True
    assert set(prov["length_bands"]["band_quotas"]) == {
        "path1",
        "pathN",
        "query",
        "root",
    }
    assert prov["length_bands"]["band_takes"]
    assert {r["url_type"] for r in rows} <= {"path1", "pathN", "query", "root"}


def _write_wave_store(
    tmp_path: Path, frame_rows: dict[str, list[Any]]
) -> tuple[Path, dict[str, Path]]:
    """Hive-partitioned Parquet store + fetch manifest, served via fake S3.

    Returns (manifest_path, keymap) where keymap maps ``pfx/primary/...``
    keys onto local part files. The frame must carry a ``stratum`` column
    (hive partition) and the wave columns (domain/url/fetch_time/
    fetch_status/content_digest/content_mime_type/url_type).
    """
    import pandas as pd

    frame = pd.DataFrame(frame_rows)
    store = tmp_path / "store"
    frame.to_parquet(store, partition_cols=["stratum"])
    keymap = {}
    for p in sorted(store.rglob("*.parquet")):
        keymap[f"pfx/primary/{p.parent.name}/{p.name}"] = p
    manifest = {
        "output": {"base": "s3://bkt/pfx/"},
        "inputs": {
            "crawls": {
                "primary": "CC-MAIN-2026-34",
                "fallback": "CC-MAIN-2026-30",
            }
        },
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    return man_path, keymap


def _fake_boto_for(keymap: dict[str, Path]) -> Any:
    """In-memory S3 stub serving local Parquet files (no network)."""

    class _FakePaginator:
        def paginate(self, Bucket: str, Prefix: str) -> Any:
            assert Bucket == "bkt"
            if Prefix == "pfx/primary/":
                return [{"Contents": [{"Key": k} for k in sorted(keymap)]}]
            return [{"Contents": []}]

    class _FakeS3:
        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"
            return _FakePaginator()

        def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
            import shutil

            shutil.copy(keymap[Key], Filename)

    class _FakeBoto:
        def client(self, name: str, **kw: Any) -> Any:
            assert name == "s3"
            return _FakeS3()

    return _FakeBoto()


def test_wave_intake_maps_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hive-partitioned Parquet -> cache-shaped entries, no S3."""
    import sys as _sys

    man_path, keymap = _write_wave_store(
        tmp_path,
        {
            "domain": ["w1.example", "w1.example", "w2.example", "w2.example"],
            "stratum": ["s4_1k_10k", "s4_1k_10k", "s5_10k_100k", "s5_10k_100k"],
            "url": [
                "https://w1.example/",
                "https://w1.example/a",
                "https://w2.example/",
                "not-a-timestamp-row",
            ],
            "fetch_time": [
                "2026-08-11 20:21:05",
                "2026-08-11 20:22:05",
                "2026-08-11 20:23:05",
                "bogus",
            ],
            "fetch_status": [200, 200, 200, 200],
            "content_digest": ["d1", "d2", "d3", "d4"],
            "content_mime_type": ["text/html"] * 4,
            "url_type": ["root", "path1", "root", "root"],
        },
    )
    monkeypatch.setitem(_sys.modules, "boto3", _fake_boto_for(keymap))
    entries = B.load_wave_entries(man_path, tmp_path / "dl")
    by_dom = {e["domain"]: e for e in entries}
    assert set(by_dom) == {"w1.example", "w2.example"}
    assert by_dom["w1.example"]["index"] == "CC-MAIN-2026-34"
    assert by_dom["w1.example"]["stratum"] == "s4_1k_10k"
    recs = {r["url"]: r for r in by_dom["w1.example"]["records"]}
    assert recs["https://w1.example/"]["timestamp"] == "20260811202105"
    assert recs["https://w1.example/"]["status"] == "200"
    # Bogus timestamp row is skipped at intake, never poisons the pool.
    w2_urls = {r["url"] for r in by_dom["w2.example"]["records"]}
    assert "not-a-timestamp-row" not in w2_urls


def test_wave_intake_records_part_hashes(tmp_path: Path) -> None:
    """Intake provenance pins per-part sha256 without re-running fetch.

    D0.6.1 registers manifest (queries, row counts, per-part sha256);
    the committed fetch manifest carries queries/counts/bytes, so the
    select provenance closes the chain at intake time (fetch-once kept).
    """
    import sys as _sys
    from unittest.mock import patch

    man_path, keymap = _write_wave_store(
        tmp_path,
        {
            "domain": ["w1.example"],
            "stratum": ["s4_1k_10k"],
            "url": ["https://w1.example/"],
            "fetch_time": ["2026-08-11 20:21:05"],
            "fetch_status": [200],
            "content_digest": ["d1"],
            "content_mime_type": ["text/html"],
            "url_type": ["root"],
        },
    )
    with patch.dict(_sys.modules, {"boto3": _fake_boto_for(keymap)}):
        parts: list[dict[str, Any]] = []
        entries = B.load_wave_entries(man_path, tmp_path / "dl", parts)
    assert len(entries) == 1
    assert len(parts) == len(keymap)
    assert all(p["sha256"] and p["size_bytes"] > 0 for p in parts)
    assert parts[0]["crawl_side"] == "primary"
    assert parts[0]["stratum_partition"] == "s4_1k_10k"


def test_phishing_tenant_files_pin(tmp_path: Path) -> None:
    """Tenant exclusion accepts the D0.1 pin; missing names refuse."""
    raw = tmp_path / "traw"
    raw.mkdir()
    _write_phish_named(raw, "openphish-2026-09-12.jsonl", ["https://evil.example/"])
    _write_phish_named(raw, "phishtank-2026-09-12.jsonl", ["https://other.example/"])
    tenants, inputs = B.phishing_tenant_set(raw, files=["openphish-2026-09-12.jsonl"])
    assert set(inputs["files"]) == {"openphish-2026-09-12.jsonl"}
    assert len(tenants) >= 1
    with pytest.raises(SystemExit):
        B.phishing_tenant_set(raw, files=["openphish-2026-09-13.jsonl"])


def test_wave_select_rederives_url_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D0.8.3: the SQL CASE bounds fetch volume only; selection re-derives types.

    A bare-trailing-'?' URL reads SQL ``query`` (strpos '?') but
    ``url_type()`` on the normalised URL reads ``path1`` (normalise drops
    the empty query). The fetched row must count under the re-derived
    type — and intake must not carry the SQL label through at all.
    """
    import sys as _sys

    man_path, keymap = _write_wave_store(
        tmp_path,
        {
            "domain": ["w1.example"],
            "stratum": ["s4_1k_10k"],
            "url": ["https://w1.example/a?"],
            "fetch_time": ["2026-08-11 20:21:05"],
            "fetch_status": [200],
            "content_digest": ["d1"],
            "content_mime_type": ["text/html"],
            # What the Athena CASE yields for this URL; must never stick.
            "url_type": ["query"],
        },
    )
    monkeypatch.setitem(_sys.modules, "boto3", _fake_boto_for(keymap))
    entries = B.load_wave_entries(man_path, tmp_path / "dl")
    assert len(entries) == 1
    assert all("url_type" not in r for e in entries for r in e["records"])
    rows, _ = _run_select(tmp_path, entries, target_n=8)
    by_url = {r["url"]: r for r in rows}
    assert by_url["https://w1.example/a"]["url_type"] == "path1"
