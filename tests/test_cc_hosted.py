"""Tests for the hosted-benign stratum (Amendment C) — offline only.

Covers the hosted fetch + select extension in ``build_cc_benign`` with a
fake Athena client and small synthetic caches (no credentials, no boto3,
no network):
* suffix-SQL renderers pin crawl/platform/subset, escape quotes, and stay
  unfiltered on fetch_status (evidence counts all rows, 200-only selection
  is client-side — same NOTE as the per-domain path);
* one platform x one crawl fetches head-first (bounded select, no count
  query), records per-query byte counts, and labels the entry stratum
  "hosted" with the platform;
* mega-platforms take the count-then-threshold path; a count failure is
  transient (never a silent unbounded download);
* hosted pools dedup on the FULL host (apex collapse would merge tenants
  of one platform into one key);
* select allocates equally per suffix with the sorted-order remainder
  rule, enforces per-tenant caps with no per-eTLD+1 cap, skips
  cross-pool duplicates, and records SQL + bytes in provenance;
* main-only selects are unchanged when --hosted-cache is absent.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import build_cc_benign as B


class _FakeAthena:
    """Minimal Athena client over literal row pages (no boto3).

    COUNT( queries get a one-row count page; selects get the row pages.
    Statistics carry DataScannedInBytes so byte accounting is exercised.
    """

    def __init__(
        self,
        rows: list[dict[str, str | None]],
        count: tuple[int, int] = (0, 0),
        fail_on: str | None = None,
        scanned_bytes: int = 123456,
    ) -> None:
        self._rows = rows
        self._count = count
        self._fail_on = fail_on
        self._scanned = scanned_bytes
        self.started: list[str] = []
        self._last_is_count = False

    def _page(self, data_rows: list[list[str]], columns: list[str]) -> dict[str, Any]:
        def cells(*vs: str) -> dict[str, Any]:
            return {"Data": [{"VarCharValue": v} for v in vs]}

        header = cells(*columns)
        return {
            "ResultSet": {
                "ResultSetMetadata": {"ColumnInfo": [{"Name": c} for c in columns]},
                "Rows": [header] + [cells(*r) for r in data_rows],
            }
        }

    def start_query_execution(
        self,
        QueryString: str,
        QueryExecutionContext: dict[str, str],
        ResultConfiguration: dict[str, str],
    ) -> dict[str, str]:
        if self._fail_on is not None and self._fail_on in QueryString:
            raise RuntimeError(f"fake failure on {self._fail_on}")
        self.started.append(QueryString)
        self._last_is_count = "COUNT(" in QueryString
        return {"QueryExecutionId": "qid-1"}

    def get_query_execution(self, QueryExecutionId: str) -> dict[str, Any]:
        assert QueryExecutionId == "qid-1"
        return {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {
                    "DataScannedInBytes": self._scanned,
                    "EngineExecutionTimeInMillis": 7,
                    "TotalExecutionTimeInMillis": 11,
                },
            }
        }

    def get_query_results(
        self, QueryExecutionId: str, NextToken: str | None = None
    ) -> dict[str, Any]:
        assert QueryExecutionId == "qid-1"
        assert NextToken is None  # single page in this fake
        if self._last_is_count:
            return self._page(
                [[str(self._count[0]), str(self._count[1])]], ["_col0", "_col1"]
            )
        cols = [
            "url",
            "fetch_time",
            "fetch_status",
            "content_digest",
            "content_mime_type",
        ]
        return self._page([[r.get(c) or "" for c in cols] for r in self._rows], cols)


def _row(url: str, status: str = "200") -> dict[str, str | None]:
    return {
        "url": url,
        "fetch_time": "2026-08-07 10:44:56.000",
        "fetch_status": status,
        "content_digest": "dg",
        "content_mime_type": "text/html",
    }


def _ctx(client: _FakeAthena) -> dict[str, Any]:
    return {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }


def test_hosted_platforms_cover_the_suffix_list() -> None:
    plats = B.hosted_platforms()
    assert len(plats) == 24
    assert plats == sorted(plats)
    for p in ("vercel.app", "blogspot.com", "bit.ly", "core.windows.net"):
        assert p in plats


def test_render_hosted_sql_pins_crawl_platform_subset() -> None:
    sql = B.render_hosted_sql("ccindex", B.CC_INDEX_PRIMARY, "vercel.app")
    assert f"crawl = '{B.CC_INDEX_PRIMARY}'" in sql
    assert "subset = 'warc'" in sql
    assert "url_host_name LIKE '%.vercel.app'" in sql
    assert "url_host_name = 'vercel.app'" in sql
    assert "fetch_status = " not in sql  # unfiltered: evidence vs selection split
    count = B.render_hosted_count_sql("ccindex", B.CC_INDEX_PRIMARY, "vercel.app")
    assert "COUNT(*)" in count and "COUNT(DISTINCT url_host_name)" in count
    assert "%.vercel.app" in count
    assert B.render_hosted_sql("t", "c", "o'brien.io").count("''") == 2


def test_render_hosted_select_sql_predicate_and_limit() -> None:
    base = B.render_hosted_sql("ccindex", B.CC_INDEX_PRIMARY, "vercel.app")
    assert (
        B.render_hosted_select_sql("ccindex", B.CC_INDEX_PRIMARY, "vercel.app") == base
    )
    sql = B.render_hosted_select_sql(
        "ccindex", B.CC_INDEX_PRIMARY, "vercel.app", 2841, 2
    )
    assert "xxhash64" in sql and "< 2841" in sql and "'2'" in sql
    head = B.render_hosted_select_sql(
        "ccindex", B.CC_INDEX_PRIMARY, "vercel.app", limit=B.SELECT_HEAD_ROWS + 1
    )
    assert head.endswith(f" LIMIT {B.SELECT_HEAD_ROWS + 1}")


def test_hosted_dedup_key_keeps_full_host() -> None:
    a = B.hosted_dedup_key("https://tenant1.vercel.app/x")
    b = B.hosted_dedup_key("https://tenant2.vercel.app/x")
    assert a != b  # tenants never merge
    assert B.hosted_dedup_key("HTTPS://Tenant1.Vercel.App/x") == a
    # Precise difference from the main key: no www-collapse (platform apex
    # vs its www, and www.<tenant> vs <tenant>, stay distinct captures).
    assert B.hosted_dedup_key("https://www.vercel.app/") != B.hosted_dedup_key(
        "https://vercel.app/"
    )
    assert B.dedup_key("https://www.vercel.app/") == B.dedup_key("https://vercel.app/")


def test_fetch_platform_crawl_head_path() -> None:
    client = _FakeAthena(
        [
            _row("https://a.vercel.app/"),
            _row("https://b.vercel.app/x"),
            _row("http://c.vercel.app/old", status="301"),
        ]
    )
    entry = B.fetch_platform_crawl("vercel.app", B.CC_INDEX_PRIMARY, _ctx(client))
    assert entry["index"] == B.CC_INDEX_PRIMARY
    assert entry["note"] == "ok"
    assert entry["stratum"] == B.HOSTED_STRATUM
    assert entry["platform"] == "vercel.app"
    assert entry["mechanism"] == "columnar"
    assert entry["sample_threshold"] is None
    assert entry["n_count_rows"] == 3  # complete head: true count
    assert entry["n_distinct_hosts"] == 3
    assert "%.vercel.app" in str(entry["query"])
    # Evidence sees the 301; records retain all statuses (the 200-only
    # selection happens in cmd_select, same as the per-domain path).
    assert entry["scheme_evidence"] == {"https": 2, "http": 1}
    assert {str(r["url"]) for r in entry["records"]} == {
        "https://a.vercel.app/",
        "https://b.vercel.app/x",
        "http://c.vercel.app/old",
    }
    # Per-query byte counts persist on the entry (provenance input).
    kinds = [q["kind"] for q in entry["query_stats"]]
    assert kinds == ["select-head"]
    assert entry["query_stats"][0]["data_scanned_bytes"] == 123456
    assert "COUNT(" not in " ".join(client.started)


def test_fetch_platform_crawl_sampled_path(monkeypatch: Any) -> None:
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 1)  # force truncation
    client = _FakeAthena(
        [_row("https://a.vercel.app/"), _row("https://b.vercel.app/x")],
        count=(60_000, 900),
    )
    entry = B.fetch_platform_crawl("vercel.app", B.CC_INDEX_PRIMARY, _ctx(client))
    assert entry["note"] == "ok"
    assert entry["n_count_rows"] == 60_000
    assert entry["n_distinct_hosts"] == 900
    assert entry["sample_threshold"] == B.sample_threshold(60_000)
    assert "xxhash64" in str(entry["query"])
    assert [q["kind"] for q in entry["query_stats"]] == [
        "select-head",
        "count",
        "select",
    ]


def test_fetch_platform_crawl_count_failure_transient(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 1)
    client = _FakeAthena(
        [_row("https://a.vercel.app/"), _row("https://b.vercel.app/x")],
        fail_on="COUNT(",
    )
    entry = B.fetch_platform_crawl(
        "vercel.app", B.CC_INDEX_PRIMARY, _ctx(client), sleep=0
    )
    assert entry["index"] is None
    assert entry["note"].startswith("unproductive(columnar:query-failed:")
    assert B._definitive(entry) is False


def _slim(url: str, ts: str = "20260807104456") -> dict[str, Any]:
    return {
        "url": url,
        "timestamp": ts,
        "digest": f"d-{url}",
        "mime": "text/html",
        "status": "200",
    }


def _hentry(platform: str, urls: list[str], index: str) -> dict[str, Any]:
    return {
        "domain": platform,
        "platform": platform,
        "index": index,
        "mechanism": "columnar",
        "query": B.render_hosted_sql("ccindex", index, platform),
        "scheme_evidence": {"https": len(urls)},
        "n_evidence_rows": len(urls),
        "http_status": 200,
        "n_records": len(urls),
        "note": "ok",
        "attempts": 1,
        "queried_at": "2026-09-13T00:00:00+00:00",
        "stratum": B.HOSTED_STRATUM,
        "query_stats": [
            {
                "crawl": index,
                "kind": "select-head",
                "data_scanned_bytes": 1000,
                "engine_ms": 1,
                "total_ms": 2,
            }
        ],
        "records": [_slim(u) for u in urls],
    }


def _run_select(
    tmp_path: Path,
    main_domains: list[dict[str, Any]],
    hosted_domains: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = {
        "seed": 0,
        "mechanism": "columnar",
        "cc_index_primary": B.CC_INDEX_PRIMARY,
        "cc_index_fallback": B.CC_INDEX_FALLBACK,
        "columnar_table_s3": B.CC_TABLE_S3,
        "columnar_athena_table": "ccindex",
        "columnar_athena_database": "ccindex",
        "columnar_sql_template": B.COLUMNAR_SQL_TEMPLATE,
        "domains": main_domains,
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
        collapse_digest=False,
        measure_quotas_from=None,
        exclude_phishing_tenants_from=None,
        require_multi_crawl=False,
        require_multi_crawl_hosted=False,
        hosted_cache=None,
        hosted_target_n=kw.get("hosted_target_n", 2000),
    )
    if hosted_domains is not None:
        hcache = {
            "seed": 0,
            "mechanism": "columnar",
            "hosted": True,
            "cc_index_primary": B.CC_INDEX_PRIMARY,
            "cc_index_fallback": B.CC_INDEX_FALLBACK,
            "platforms": B.hosted_platforms(),
            "hosted_sql_template": B.HOSTED_SQL_TEMPLATE,
            "hosted_count_template": B.HOSTED_COUNT_TEMPLATE,
            "columnar_sample_predicate_template": (
                B.COLUMNAR_SAMPLE_PREDICATE_TEMPLATE
            ),
            "columnar_hash_modulus": B.HASH_MODULUS,
            "columnar_sample_target_rows": B.SAMPLE_TARGET_ROWS,
            "domains": hosted_domains,
        }
        hp = tmp_path / "hosted.json"
        hp.write_text(json.dumps(hcache), encoding="utf-8")
        a.hosted_cache = str(hp)
        a.require_multi_crawl_hosted = kw.get("hosted_multi_crawl", False)
    assert B.cmd_select(a) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    prov = json.loads(Path(str(out) + ".provenance.json").read_text(encoding="utf-8"))
    return rows, prov


def _mentry(domain: str, urls: list[str]) -> dict[str, Any]:
    return {
        "domain": domain,
        "index": B.CC_INDEX_PRIMARY,
        "http_status": 200,
        "n_records": len(urls),
        "note": "ok",
        "attempts": 1,
        "queried_at": "2026-09-13T00:00:00+00:00",
        "stratum": "s5_10k_100k",
        "rank": 15000,
        "records": [_slim(u) for u in urls],
    }


def test_build_hosted_pools_full_host_dedup() -> None:
    pools, stats = B.build_hosted_pools(
        [
            _hentry(
                "vercel.app",
                [
                    "https://t1.vercel.app/x",
                    "https://t2.vercel.app/x",
                    "https://t1.vercel.app/x",  # exact dup collapses, earliest wins
                ],
                B.CC_INDEX_PRIMARY,
            ),
        ]
    )
    got = sorted(r["url"] for rows in pools.values() for r in rows)
    assert got == ["https://t1.vercel.app/x", "https://t2.vercel.app/x"]
    assert stats["duplicates"] == 1
    row = next(r for rows in pools.values() for r in rows)
    assert row["stratum"] == B.HOSTED_STRATUM
    assert row["seed_domain"] == "vercel.app"


def test_select_allocates_equally_per_platform(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    hosted = [
        _hentry(p, [f"https://t{i}.{p}/page{i}" for i in range(4)], B.CC_INDEX_PRIMARY)
        for p in plats
    ]
    rows, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        hosted,
        hosted_target_n=48,
    )
    h = prov["hosted_stratum"]
    assert h["enabled"] is True
    assert h["target_n"] == 48
    assert h["n_written"] == 48
    # 48 = 24 x 2: every platform exactly at quota, no shortfall.
    assert all(
        v["taken"] == 2 and v["shortfall"] == 0 for v in h["per_platform"].values()
    )
    hrows = [r for r in rows if r["popularity_stratum"] == "hosted"]
    assert len(hrows) == 48
    assert all(r["tranco_list_id"] is None and r["tranco_rank"] is None for r in hrows)
    assert all(r["seed_domain"] in plats for r in hrows)


def test_select_remainder_goes_to_first_sorted_platforms(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    hosted = [
        _hentry(p, [f"https://t{i}.{p}/p{i}" for i in range(4)], B.CC_INDEX_PRIMARY)
        for p in plats
    ]
    _, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        hosted,
        hosted_target_n=26,
    )
    per = prov["hosted_stratum"]["per_platform"]
    assert [v["quota"] for p, v in per.items()].count(2) == 2
    assert [v["quota"] for p, v in per.items()].count(1) == 22
    assert per[plats[0]]["quota"] == 2 and per[plats[1]]["quota"] == 2


def test_select_hosted_shortfall_recorded_not_patched(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    hosted = [_hentry(plats[0], ["https://only.vercel.app/"], B.CC_INDEX_PRIMARY)]
    _, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        hosted,
        hosted_target_n=48,
    )
    per = prov["hosted_stratum"]["per_platform"]
    assert per[plats[0]]["taken"] == 1
    assert per[plats[0]]["shortfall"] == 1
    assert prov["hosted_stratum"]["n_written"] == 1


def test_select_hosted_tenant_caps_no_etld1_cap(tmp_path: Path) -> None:
    # 20 distinct tenants on one platform, one row each: all survive
    # (a per-eTLD+1 cap of 25 would also pass — so also assert the
    # per-tenant-type cap binds: 7 path1 rows on ONE tenant keep 6).
    plats = B.hosted_platforms()
    p = plats[0]
    urls = [f"https://tenant{i}.{p}/u" for i in range(20)]
    urls += [f"https://solo.{p}/w{i}" for i in range(7)]
    rows, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        [_hentry(p, urls, B.CC_INDEX_PRIMARY)],
        hosted_target_n=24 * 26,
    )
    hrows = [r for r in rows if r["popularity_stratum"] == "hosted"]
    assert len(hrows) == 20 + 6
    assert prov["hosted_stratum"]["caps"]["per_etld1"] is None


def test_select_cross_pool_dedup_prefers_main(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    dup = "https://ex-a.com/shared"
    rows, _ = _run_select(
        tmp_path,
        [_mentry("ex-a.com", [dup, "https://ex-a.com/other"])],
        [_hentry(plats[0], [dup, f"https://t.{plats[0]}/u"], B.CC_INDEX_PRIMARY)],
        target_n=24,
        hosted_target_n=24,
    )
    assert sorted(r["url"] for r in rows).count(dup) == 1
    assert next(r for r in rows if r["url"] == dup)["popularity_stratum"] != "hosted"


def test_select_hosted_multi_crawl_per_pool(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    p = plats[0]
    rows, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        [
            _hentry(p, ["https://once.vercel.app/a"], B.CC_INDEX_PRIMARY),
            _hentry(p, ["https://twice-x.vercel.app/a"], B.CC_INDEX_PRIMARY),
            _hentry(p, ["https://twice-x.vercel.app/b"], B.CC_INDEX_FALLBACK),
        ],
        hosted_target_n=24,
        hosted_multi_crawl=True,
    )
    by_url = {r["url"] for r in rows}
    assert "https://once.vercel.app/a" not in by_url
    assert "https://twice-x.vercel.app/a" in by_url
    assert prov["hosted_stratum"]["multi_crawl"]["required"] is True
    # The main pool is untouched by the hosted-only flag.
    assert prov["multi_crawl"]["required"] is False


def test_select_main_only_unchanged_without_hosted_cache(tmp_path: Path) -> None:
    rows, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
    )
    assert prov["hosted_stratum"] == {"enabled": False}
    assert "hosted_sql_template" not in prov["fetch_mechanism_detail"]
    assert all(r["popularity_stratum"] != "hosted" for r in rows)


def test_select_provenance_carries_sql_and_bytes(tmp_path: Path) -> None:
    plats = B.hosted_platforms()
    rows, prov = _run_select(
        tmp_path,
        [_mentry("ex-a.com", ["https://ex-a.com/"])],
        [_hentry(plats[0], ["https://t.vercel.app/"], B.CC_INDEX_PRIMARY)],
        hosted_target_n=24,
    )
    assert prov["fetch_mechanism_detail"]["hosted_sql_template"] == (
        B.HOSTED_SQL_TEMPLATE
    )
    assert prov["fetch_mechanism_detail"]["hosted_platforms"] == plats
    assert prov["hosted_stratum"]["data_scanned_bytes_total"] == 1000
    assert prov["hosted_stratum"]["hosted_cache_sha256"]


def test_select_missing_hosted_cache_exits(tmp_path: Path) -> None:
    import pytest as _pytest

    cache = {
        "seed": 0,
        "mechanism": "columnar",
        "cc_index_primary": B.CC_INDEX_PRIMARY,
        "cc_index_fallback": B.CC_INDEX_FALLBACK,
        "domains": [],
    }
    cp = tmp_path / "cache.json"
    cp.write_text(json.dumps(cache), encoding="utf-8")
    a = Namespace(
        seed=0,
        target_n=8,
        productive_per_stratum=500,
        cache=str(cp),
        out=str(tmp_path / "o.jsonl"),
        collapse_digest=False,
        measure_quotas_from=None,
        exclude_phishing_tenants_from=None,
        require_multi_crawl=False,
        require_multi_crawl_hosted=False,
        hosted_cache=str(tmp_path / "nope.json"),
        hosted_target_n=2000,
    )
    with _pytest.raises(SystemExit):
        B.cmd_select(a)
