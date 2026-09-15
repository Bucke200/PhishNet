"""Tests for the columnar (Athena/S3) acquisition path — offline only.

A checked-in fixture holds two literal Athena GetQueryResults pages, so
no AWS credentials, no boto3, and no network are needed. The tests prove:
* paged Athena responses parse into rows (header skipped, NextToken
  followed, NULL cells tolerated);
* rows map into the exact slim-record schema the selection stage expects
  ({url, timestamp, digest, mime, status}, CC timestamp format), with the
  CDX-equivalent filters (http/https only, no robots.txt) and
  collapse-to-earliest dedupe;
* a cache built from those records feeds cmd_select unchanged
  (mechanism-blind selection + columnar provenance);
* scheme evidence counts all fetch statuses (301/302 included) while
  selection keeps 200s only, and both rates land in provenance.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import build_cc_benign as B

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cc-columnar-athena-pages.json"


class _FakeAthena:
    """Minimal Athena client over the fixture pages (no boto3).

    Dispatches on the SQL: calibration (COUNT() queries get a one-row
    count page, empty result means zero) vs row selects (fixture pages).
    """

    def __init__(
        self,
        fixture: dict[str, Any],
        count: tuple[int, int] = (11, 2),
        fail_on: str | None = None,
        fail_message: str | None = None,
    ) -> None:
        self._pages: list[dict[str, Any]] = [fixture["page1"], fixture["page2"]]
        self._count = count
        self._fail_on = fail_on
        self._fail_message = fail_message
        self.started: list[dict[str, Any]] = []
        self._last_is_count = False

    def _count_page(self) -> dict[str, Any]:
        def cells(*vs: str) -> dict[str, Any]:
            return {"Data": [{"VarCharValue": v} for v in vs]}

        return {
            "ResultSet": {
                "ResultSetMetadata": {
                    "ColumnInfo": [{"Name": "_col0"}, {"Name": "_col1"}]
                },
                "Rows": [
                    cells("_col0", "_col1"),
                    cells(str(self._count[0]), str(self._count[1])),
                ],
            }
        }

    def start_query_execution(
        self,
        QueryString: str,
        QueryExecutionContext: dict[str, str],
        ResultConfiguration: dict[str, str],
    ) -> dict[str, str]:
        if self._fail_on is not None and self._fail_on in QueryString:
            raise RuntimeError(
                self._fail_message or f"fake failure on {self._fail_on}"
            )
        self.started.append(
            {
                "sql": QueryString,
                "database": QueryExecutionContext["Database"],
                "output": ResultConfiguration["OutputLocation"],
            }
        )
        self._last_is_count = "COUNT(" in QueryString
        return {"QueryExecutionId": "qid-1"}

    def get_query_execution(self, QueryExecutionId: str) -> dict[str, Any]:
        assert QueryExecutionId == "qid-1"
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(
        self, QueryExecutionId: str, NextToken: str | None = None
    ) -> dict[str, Any]:
        assert QueryExecutionId == "qid-1"
        if self._last_is_count:
            assert NextToken is None
            return self._count_page()
        if NextToken is None:
            return self._pages[0]
        assert NextToken == "tok-1"
        return self._pages[1]


def _load_fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def test_athena_time_formats() -> None:
    assert B.athena_time_to_cc("2026-08-07 10:44:56.000") == "20260807104456"
    assert B.athena_time_to_cc("2026-08-07 10:44:58") == "20260807104458"
    assert B.athena_time_to_cc("2026-08-07") == "20260807000000"
    assert B.athena_time_to_cc("not-a-time") is None


def test_athena_paging_and_row_mapping() -> None:
    client = _FakeAthena(_load_fixture())
    rows = B.athena_query_rows(client, "SELECT ...", "ccindex", "s3://out/prefix/")
    assert len(rows) == 11  # 7 data rows on page 1 (header skipped) + 4 on page 2
    assert rows[0]["url"] == "https://example.com/"
    assert rows[0]["fetch_status"] == "200"
    assert client.started[0]["database"] == "ccindex"
    assert client.started[0]["output"] == "s3://out/prefix/"


def test_columnar_records_match_select_schema() -> None:
    client = _FakeAthena(_load_fixture())
    rows = B.athena_query_rows(client, "SELECT ...", "ccindex", "s3://out/prefix/")
    recs = sorted(B.columnar_records(rows), key=lambda r: str(r["url"]))
    assert [r["url"] for r in recs] == [
        "http://example.com/",
        "http://example.com/old",
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/redir",
        "https://sub.example.com/x?q=1",
    ]
    for r in recs:
        assert set(r) == {"url", "timestamp", "digest", "mime", "status"}
        assert len(str(r["timestamp"])) == 14 and str(r["timestamp"]).isdigit()
    by_url = {str(r["url"]): r for r in recs}
    # Duplicate captures collapse to the earliest fetch.
    assert by_url["https://example.com/"]["timestamp"] == "20260807104456"
    assert by_url["https://example.com/"]["digest"] == "abc"
    assert by_url["http://example.com/old"]["timestamp"] == "20260807000000"
    # Non-200 rows are retained with status for evidence, never selected.
    assert by_url["http://example.com/"]["status"] == "301"
    assert by_url["https://example.com/redir"]["status"] == "302"
    assert by_url["https://example.com/"]["status"] == "200"


def test_scheme_evidence_counts_all_statuses() -> None:
    client = _FakeAthena(_load_fixture())
    rows = B.athena_query_rows(client, "SELECT ...", "ccindex", "s3://out/prefix/")
    assert B.scheme_counts(rows) == {"https": 7, "http": 2}


def test_fetch_domain_columnar_splits_evidence_and_selection() -> None:
    client = _FakeAthena(_load_fixture())
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx)
    assert entry["index"] == B.CC_INDEX_PRIMARY
    assert entry["note"] == "ok"
    assert entry["mechanism"] == "columnar"
    assert entry["n_evidence_rows"] == 11
    # Small domain: the bounded head completes in one start, no count
    # query, no sampling predicate. The true count is len(rows) and the
    # host count over the full capture set is exact.
    assert entry["sample_threshold"] is None
    assert entry["n_count_rows"] == 11
    assert entry["n_distinct_hosts"] == 2
    assert entry["query"] == (
        B.render_columnar_sql("ccindex", B.CC_INDEX_PRIMARY, "example.com")
        + f" LIMIT {B.SELECT_HEAD_ROWS + 1}"
    )
    # Evidence sees the http 301; selection keeps 200s only.
    assert entry["scheme_evidence"] == {"https": 7, "http": 2}
    assert {str(r["url"]) for r in entry["records"]} == {
        "http://example.com/",
        "http://example.com/old",
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/redir",
        "https://sub.example.com/x?q=1",
    }


def test_columnar_cache_feeds_select_unchanged(tmp_path: Path) -> None:
    client = _FakeAthena(_load_fixture())
    rows = B.athena_query_rows(client, "SELECT ...", "ccindex", "s3://out/prefix/")
    recs = B.columnar_records(rows)
    evidence = B.scheme_counts(rows)
    cache = {
        "seed": 0,
        "mechanism": "columnar",
        "cc_index_primary": B.CC_INDEX_PRIMARY,
        "cc_index_fallback": B.CC_INDEX_FALLBACK,
        "columnar_table_s3": B.CC_TABLE_S3,
        "columnar_athena_table": "ccindex",
        "columnar_athena_database": "ccindex",
        "columnar_sql_template": B.COLUMNAR_SQL_TEMPLATE,
        "tranco_csv": str(B.TRANC0_CSV),
        "tranco_sha256": B.TRANC0_SHA256,
        "domains": [
            {
                "domain": "example.com",
                "index": B.CC_INDEX_PRIMARY,
                "mechanism": "columnar",
                "query": B.render_columnar_sql(
                    "ccindex", B.CC_INDEX_PRIMARY, "example.com"
                ),
                "scheme_evidence": evidence,
                "n_evidence_rows": len(rows),
                "http_status": 200,
                "n_records": len(recs),
                "note": "ok",
                "attempts": 1,
                "queried_at": "2026-09-13T00:00:00+00:00",
                "stratum": "s5_10k_100k",
                "rank": 15000,
                "records": recs,
            }
        ],
    }
    cp = tmp_path / "cache.json"
    cp.write_text(json.dumps(cache), encoding="utf-8")
    out = tmp_path / "benign.jsonl"
    a = Namespace(
        seed=0,
        target_n=4,
        productive_per_stratum=500,
        cache=str(cp),
        out=str(out),
        collapse_digest=False,
    )
    assert B.cmd_select(a) == 0
    selected = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert selected, "columnar records must survive selection"
    for row in selected:
        assert row["source"] == f"cc:{B.CC_INDEX_PRIMARY}"
        assert row["time_basis"] == "commoncrawl-index"
        assert row["tranco_list_id"] == B.TRANC0_ID
    prov = json.loads(Path(str(out) + ".provenance.json").read_text(encoding="utf-8"))
    assert prov["fetch_mechanism"] == "columnar"
    assert prov["fetch_mechanism_detail"]["columnar_sql_template"] == (
        B.COLUMNAR_SQL_TEMPLATE
    )
    comp = prov["root_synthesis"]["scheme_evidence_vs_selection"]
    # Apex-host evidence: 3 https of 5 apex rows (dup root collapsed,
    # robots excluded at mapping); selection: 3 https of 4.
    assert comp["mean_https_share_evidence_apex_unfiltered"] == pytest.approx(3 / 5)
    assert comp["mean_https_share_selection_200_only"] == pytest.approx(3 / 4)
    assert comp["n_domains_compared"] == 1


def test_sample_threshold_math() -> None:
    # At or under the target: no predicate (None).
    assert B.sample_threshold(0) is None
    assert B.sample_threshold(B.SAMPLE_TARGET_ROWS) is None
    # Just over: nearly the whole modulus, always strictly below it.
    just_over = B.sample_threshold(B.SAMPLE_TARGET_ROWS + 1)
    assert just_over is not None and just_over < B.HASH_MODULUS
    # Blogspot scale (~17.6M rows): ~2.8k slice of the modulus.
    big = B.sample_threshold(17_600_000)
    assert big == -(-B.SAMPLE_TARGET_ROWS * B.HASH_MODULUS // 17_600_000)
    assert 0 < big < B.HASH_MODULUS
    # Monotone non-increasing in count above the target.
    assert B.sample_threshold(60_000) >= B.sample_threshold(6_000_000)


def test_select_sql_byte_identical_without_threshold() -> None:
    base = B.render_columnar_sql("ccindex", B.CC_INDEX_PRIMARY, "example.com")
    assert B.render_columnar_select_sql("ccindex", B.CC_INDEX_PRIMARY, "example.com") == base
    assert (
        B.render_columnar_select_sql(
            "ccindex", B.CC_INDEX_PRIMARY, "example.com", None, 2
        )
        == base
    )


def test_select_sql_predicate_pins_seed_and_threshold() -> None:
    sql = B.render_columnar_select_sql(
        "ccindex", B.CC_INDEX_PRIMARY, "example.com", 2841, 2
    )
    assert f"crawl = '{B.CC_INDEX_PRIMARY}'" in sql
    assert "url_host_registered_domain = 'example.com'" in sql
    assert "fetch_status = " not in sql  # still unfiltered on status
    assert "xxhash64" in sql
    assert "'2'" in sql  # sample seed folded into the hashed string
    assert "< 2841" in sql
    assert B.render_columnar_select_sql("ccindex", "c", "o'brien.com", 7, 2).count(
        "''"
    ) == 1


def test_count_and_threshold_persisted_in_record(monkeypatch) -> None:
    # Shrink the head so the 11-row fixture truncates and takes the
    # count-then-threshold path.
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 10)
    client = _FakeAthena(_load_fixture(), count=(60_000, 5))
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx)
    assert entry["note"] == "ok"
    assert entry["n_count_rows"] == 60_000
    assert entry["n_distinct_hosts"] == 5
    assert entry["sample_threshold"] == B.sample_threshold(60_000)
    assert "xxhash64" in str(entry["query"])
    # Selection still runs on the returned rows, unchanged.
    assert entry["n_evidence_rows"] == 11
    assert {str(r["url"]) for r in entry["records"]} == {
        "http://example.com/",
        "http://example.com/old",
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/redir",
        "https://sub.example.com/x?q=1",
    }


def test_count_failure_is_transient_not_silent(monkeypatch) -> None:
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 10)  # force the count path
    client = _FakeAthena(_load_fixture(), fail_on="COUNT(")
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx, sleep=0)
    # No count means no bound on the result size: retry, then report
    # failure — never a silent unbounded download, never fake success.
    # The engine's message rides in the note (and hence the log line).
    assert entry["index"] is None
    assert entry["note"].startswith("unproductive(columnar:query-failed:")
    assert "fake failure" in str(entry["note"])
    assert entry["n_count_rows"] is None
    assert B._definitive(entry) is False


def test_engine_error_reaches_note_and_stays_transient(monkeypatch) -> None:
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 10)  # force the count path
    # Throttling takes the minutes-scale curve: never sleep for real here
    # (the observed-attempts test below asserts the curve itself).
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    client = _FakeAthena(
        _load_fixture(),
        fail_on="COUNT(",
        fail_message="athena FAILED: HIVE_S3_THROTTLING: S3 throttling",
    )
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx, sleep=0)
    assert "HIVE_S3_THROTTLING" in str(entry["note"])
    # A suffixed failure note must never read as a definitive outcome:
    # resume drops it and re-fetches instead of silently excluding.
    assert B._definitive(entry) is False


def test_head_sql_carries_limit() -> None:
    sql = B.render_columnar_select_sql(
        "ccindex", B.CC_INDEX_PRIMARY, "example.com", limit=B.SELECT_HEAD_ROWS + 1
    )
    assert sql.endswith(f" LIMIT {B.SELECT_HEAD_ROWS + 1}")
    assert f"crawl = '{B.CC_INDEX_PRIMARY}'" in sql


def test_retry_delay_curves() -> None:
    # Ordinary flakes: short exponential base.
    assert B._retry_delay(0, 5.0, False) == 5.0
    assert B._retry_delay(1, 5.0, False) == 10.0
    # Throttling: minutes-scale curve regardless of the base.
    assert B._retry_delay(0, 5.0, True) == 30.0
    assert B._retry_delay(1, 5.0, True) == 120.0
    assert B._retry_delay(9, 5.0, True) == 120.0  # curve end holds
    assert B._is_throttle_error("athena FAILED: HIVE_S3_THROTTLING: S3 throttling")
    assert B._is_throttle_error("ThrottlingException: Rate exceeded")
    assert not B._is_throttle_error("RuntimeError: fake failure")
    assert not B._is_throttle_error(None)


def test_throttle_backoff_observed_in_attempts(monkeypatch) -> None:
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 10)  # force the count path
    slept: list[float] = []
    monkeypatch.setattr(B.time, "sleep", slept.append)
    client = _FakeAthena(
        _load_fixture(),
        fail_on="COUNT(",
        fail_message="athena FAILED: HIVE_S3_THROTTLING: S3 throttling",
    )
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx)
    # Two primary attempts on the long curve, then the immediate fallback
    # probe — the three attempts span minutes, not seconds.
    assert slept == [30.0, 120.0]
    assert "HIVE_S3_THROTTLING" in str(entry["note"])


def test_truncated_head_without_threshold_fails_loud(monkeypatch) -> None:
    # Truncation proves count > head >> target; a agreeing count that still
    # yields no threshold is a calibration contradiction — fail loud with
    # the contradiction in the note, never proceed unfiltered.
    monkeypatch.setattr(B, "SELECT_HEAD_ROWS", 10)
    client = _FakeAthena(_load_fixture(), count=(11, 2))
    ctx = {
        "client": client,
        "table": "ccindex",
        "database": "ccindex",
        "output": "s3://out/prefix/",
        "row_cap": 5000,
        "sample_seed": 2,
    }
    entry = B.fetch_domain_columnar("example.com", ctx, sleep=0)
    assert entry["index"] is None
    assert "truncated head but no threshold" in str(entry["note"])


def test_distinct_hostnames_unit() -> None:
    rows = [
        {"url": "https://example.com/"},
        {"url": "https://EXAMPLE.com/x"},
        {"url": "https://sub.example.com/y"},
        {"url": "not a url"},
        {"url": None},
    ]
    assert B._distinct_hostnames(rows) == 2


def test_render_columnar_sql_pins_crawl_and_domain() -> None:
    sql = B.render_columnar_sql("ccindex", B.CC_INDEX_PRIMARY, "example.com")
    assert f"crawl = '{B.CC_INDEX_PRIMARY}'" in sql
    assert "url_host_registered_domain = 'example.com'" in sql
    assert "subset = 'warc'" in sql
    # Deliberately unfiltered on status: evidence counts all rows, the
    # 200-only selection happens in columnar_records.
    assert "fetch_status = " not in sql
    assert B.render_columnar_sql("ccindex", "c", "o'brien.com").count("''") == 1


def _sample_rows(n: int) -> list[dict[str, str | None]]:
    return [
        {
            "url": f"https://d{i % 7}.example.com/p{i}",
            "fetch_time": "2026-08-07 10:44:56.000",
            "fetch_status": "200",
            "content_digest": None,
            "content_mime_type": "text/html",
        }
        for i in range(n)
    ]


def test_sample_rows_passthrough_under_cap() -> None:
    rows = _sample_rows(100)
    kept, sampled = B.sample_rows(rows, 5000, seed=2)
    assert kept == rows and sampled is False


def test_sample_rows_deterministic_uniform_prefix() -> None:
    rows = _sample_rows(6000)
    first, sampled = B.sample_rows(rows, 5000, seed=2)
    second, _ = B.sample_rows(list(reversed(rows)), 5000, seed=2)
    assert sampled is True
    assert len(first) == 5000
    # Same content set regardless of input order (sort-by-URL first).
    assert {r["url"] for r in first} == {r["url"] for r in second}
    # Uniform, not earliest: the kept set is not the first 5000 inputs.
    assert {r["url"] for r in first} != {r["url"] for r in rows[:5000]}
    other, _ = B.sample_rows(rows, 5000, seed=3)
    assert {r["url"] for r in first} != {r["url"] for r in other}
