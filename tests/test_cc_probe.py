"""Offline tests for the Amendment D JOIN probe (M3, D0.5) — no AWS, no boto3.

Covers the read-only guarantees and the decision inputs with synthetic
inputs only:
* the JOIN SQL pins crawl/subset/registered-domain/apex-or-www-host/200
  plus exact root-URL equalities (no LIKE wildcards); identifiers are
  validated, literals escaped;
* the candidates DDL points at the upload location with a header skip;
* candidate replay matches fetch's seeded order (one default_rng stream,
  one permutation per stratum in STRATA order) and excludes definitive
  outcomes only;
* fallback preference is per-domain client-side; the per-domain root cap
  and the bar decision apply to the exact measured total (no projection);
* the execute path uploads/queries/drops with fake clients and writes
  the report + counts CSV; the temp table never survives by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest

import build_cc_benign as B
import probe_cc_roots as P

ROOT = Path(__file__).resolve().parents[1]


def _mapping(n: int) -> dict[int, str]:
    return {r: f"d{r}.example" for r in range(1, n + 1)}


def test_join_sql_pins_predicates() -> None:
    sql = P.render_join_sql("ccindex", "probe_cands_x", "CC-MAIN-2026-34")
    assert "FROM ccindex ci JOIN probe_cands_x c" in sql
    assert "ON ci.url_host_registered_domain = c.domain" in sql
    assert "ci.crawl = 'CC-MAIN-2026-34'" in sql
    assert "ci.subset = 'warc'" in sql
    assert "ci.fetch_status = 200" in sql
    assert "COUNT(DISTINCT ci.url)" in sql
    assert "GROUP BY c.domain" in sql
    assert "LIKE" not in sql
    # Root-URL predicate excludes '?' from the host class (a query string
    # makes it url_type "query") and tolerates fragments.
    assert "regexp_like(ci.url, '^https?://[^/?]+/?(#.*)?$')" in sql
    assert "url_host_name" not in sql


def test_join_sql_rejects_exotic_identifiers() -> None:
    with pytest.raises(SystemExit):
        P.render_join_sql("ccindex; DROP TABLE x; --", "c", "CC-MAIN-2026-34")
    with pytest.raises(SystemExit):
        P.render_join_sql("ccindex", "c", "CC-MAIN-2026-34' OR '1'='1")


def test_type_case_renders_query_first() -> None:
    case = P.render_type_case()
    positions = [case.index(f"'{t}'") for t in ("query", "root", "path1", "pathN")]
    assert positions == sorted(positions)
    assert "strpos(ci.url, '?') > 0" in case
    assert case.endswith("ELSE 'malformed' END")


def test_sql_url_type_matches_url_type_on_clean() -> None:
    from build_cc_benign import url_type

    clean = [
        "https://example.com/",
        "http://example.com",
        "https://www.example.com/",
        "https://sub.example.com/",
        "https://example.com:8443/",
        "https://example.com/a",
        "http://example.com/a/",
        "https://example.com/a/b/c",
        "https://example.com/a?x=1",
        "https://example.com/?x=1",
        "https://example.com/a/b?q=1&r=2",
        "https://example.com/#frag",
        "https://example.com/a#frag",
    ]
    for u in clean:
        assert P.sql_url_type(u) == url_type(u), u


def test_sql_url_type_documented_divergences() -> None:
    """Edge classes where the SQL mirror and url_type disagree by design."""
    from build_cc_benign import url_type

    edges = {
        # Bare trailing '?' (empty query): SQL strpos says query.
        "https://example.com/a?": "query",
        "https://example.com?": "query",
        # Semicolon path-params: urlparse splits params off the path.
        "https://example.com/;jsessionid=1": "path1",
        "https://example.com/a/;x=1": "pathN",
        # Double-slash paths: urlparse sees empty segments.
        "https://example.com//": "pathN",
        # Uppercase scheme: urlparse lowercases it, the regex does not.
        "HTTPS://example.com/": "pathN",
    }
    for u, sql_want in edges.items():
        assert P.sql_url_type(u) == sql_want, u
        assert url_type(u) != sql_want, u


def test_sql_url_type_cache_agreement() -> None:
    """99.5%+ agreement on banked records; every mismatch is documented."""
    cache = ROOT / "data" / "raw" / "cc-columnar-CC-MAIN-2026-34.json"
    if not cache.exists():
        pytest.skip("banked main cache absent (data/ git-ignored)")
    import build_splits
    from build_cc_benign import url_type

    rows: list[str] = []
    for e in json.loads(cache.read_text(encoding="utf-8"))["domains"]:
        if e.get("index") is None:
            continue
        for rec in e.get("records", []):
            u = rec.get("url") or ""
            if build_splits.normalise(u) is None or url_type(u) == "malformed":
                continue
            rows.append(u)
    rng = np.random.default_rng(7)
    sample = [rows[int(i)] for i in rng.permutation(len(rows))[:3000]]
    bad = 0
    for u in sample:
        if P.sql_url_type(u) != url_type(u):
            bad += 1
            path = urlparse(u).path
            assert (
                u.endswith("?")
                or ";" in path
                or "//" in path
                or u.startswith(("HTTP", "HTTPS"))
            ), f"undocumented divergence: {u!r}"
    assert bad / len(sample) <= 0.005


def test_xxhash_order_vectors() -> None:
    """Seeded-hash order contract for the fetch row_number bound (D0.7).

    Vectors assume Trino's xxhash64 is stock XXH64 (seed 0): the hashed
    input is url+'|'+seed, ordered by abs() of the SIGNED int64 — the
    identical integer the sampling predicate thresholds on. xxhash is an
    ephemeral dev-only dependency (uv run --with xxhash); locked CI
    skips this test.
    """
    xxhash = pytest.importorskip("xxhash")

    def orderkey(url: str, seed: int) -> int:
        raw = xxhash.xxh64(f"{url}|{seed}".encode()).intdigest()
        return abs(raw - 2**64 if raw >= 2**63 else raw)

    vectors = {
        ("https://example.com/", 2): 7104185197328256320,
        ("https://example.com/a", 2): 2521590795364187408,
        ("http://t1.vercel.app/", 2): 6279469116289946235,
        ("https://x.com/a/b?q=1", 2): 1140093894507657044,
        ("https://y.com/", 0): 2592063262246022182,
    }
    for (url, seed), want in vectors.items():
        assert orderkey(url, seed) == want
    ascending = sorted(vectors, key=vectors.get)
    assert [u for u, _ in ascending] == [
        "https://x.com/a/b?q=1",
        "https://example.com/a",
        "https://y.com/",
        "http://t1.vercel.app/",
        "https://example.com/",
    ]
    expr = P.render_order_expr(2)
    assert "concat(url, '|', '2')" in expr
    assert expr.startswith("abs(from_big_endian_64(xxhash64(")


def test_ddl_and_drop() -> None:
    ddl = P.render_candidates_ddl("probe_cands_x", "s3://bkt/pfx/x/")
    assert "CREATE EXTERNAL TABLE probe_cands_x (domain string)" in ddl
    assert "LOCATION 's3://bkt/pfx/x/'" in ddl
    assert "skip.header.line.count'='1'" in ddl
    assert P.render_drop_sql("probe_cands_x") == "DROP TABLE probe_cands_x"
    with pytest.raises(SystemExit):
        P.render_drop_sql("no-dashes-allowed")


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


def test_cap_sum_decide_and_takeable() -> None:
    assert P.cap_sum([0, 3, 9, 6], 6) == 0 + 3 + 6 + 6
    # Calibration transfer on the capped pool (reports/probe-calibration).
    assert P.takeable_roots(1000) == pytest.approx(1000 * 0.99 * 0.72)
    assert 0.0 < P.UNIT_RATIO <= 1.0 and 0.0 < P.FILL_EFFICIENCY_TAIL <= 1.0
    assert P.decide(5900, 5900) == "D1"  # boundary counts
    assert P.decide(5899, 5900) == "D2"


def test_prefer_primary() -> None:
    out = P.prefer_primary({"a": 3, "b": 0}, {"b": 2, "c": 5})
    assert out == {"a": 3, "b": 2, "c": 5}


def test_parse_domain_counts() -> None:
    rows = [
        {"domain": "a.example", "n_roots": "3"},
        {"domain": "b.example", "n_roots": "0"},
    ]
    assert P.parse_domain_counts(rows) == {"a.example": 3, "b.example": 0}
    assert P.parse_domain_counts([]) == {}


def test_candidates_csv_and_guard() -> None:
    assert (
        P.candidates_csv(["b.example", "a.example"]) == "domain\nb.example\na.example\n"
    )
    with pytest.raises(SystemExit):
        P.candidates_csv(["evil',example"])


def test_split_s3_url() -> None:
    assert P.split_s3_url("s3://bkt/pfx/sub") == ("bkt", "pfx/sub")
    with pytest.raises(SystemExit):
        P.split_s3_url("https://bkt/pfx")
    with pytest.raises(SystemExit):
        P.split_s3_url("s3://bkt")


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


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[tuple[str, str, bytes]] = []
        self.deletes: list[tuple[str, list[str]]] = []

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]:
        self.puts.append((Bucket, Key, Body))
        return {}

    def delete_objects(self, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        keys = [o["Key"] for o in Delete["Objects"]]
        self.deletes.append((Bucket, keys))
        return {}


class _FakeAthena:
    """Routes canned row pages by crawl substring; DDL/DROP get headers only.

    Primary yields 3 roots on the first uploaded domain, fallback 9 on
    the second — read back from the fake S3 upload so the counts land on
    domains the probe actually enumerated.
    """

    def __init__(self, s3: _FakeS3) -> None:
        self._s3 = s3
        self.sqls: list[str] = []
        self._qid = 0

    def _uploaded(self) -> list[str]:
        body = self._s3.puts[0][2].decode("utf-8").splitlines()
        return [line for line in body[1:] if line]

    def start_query_execution(self, QueryString: str, **_: Any) -> dict[str, Any]:
        self.sqls.append(QueryString)
        self._qid += 1
        return {"QueryExecutionId": f"q{self._qid}"}

    def get_query_execution(self, QueryExecutionId: str) -> dict[str, Any]:
        return {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {
                    "DataScannedInBytes": 1000,
                    "EngineExecutionTimeInMillis": 10,
                    "TotalExecutionTimeInMillis": 20,
                },
            }
        }

    def _page(self, rows: list[tuple[str, int]]) -> dict[str, Any]:
        def cells(*vs: str) -> dict[str, Any]:
            return {"Data": [{"VarCharValue": v} for v in vs]}

        return {
            "ResultSet": {
                "ResultSetMetadata": {
                    "ColumnInfo": [{"Name": "domain"}, {"Name": "n_roots"}]
                },
                "Rows": [cells("domain", "n_roots")]
                + [cells(d, str(n)) for d, n in rows],
            }
        }

    def get_query_results(self, QueryExecutionId: str, **_: Any) -> dict[str, Any]:
        qid = int(QueryExecutionId[1:])
        sql = self.sqls[qid - 1]
        doms = self._uploaded()
        if "CC-MAIN-2026-34" in sql:
            return self._page([(doms[0], 3)] if doms else [])
        if "CC-MAIN-2026-30" in sql:
            return self._page([(doms[1], 9)] if len(doms) > 1 else [])
        return self._page([])


def _empty_pinned_raw(raw: Path) -> None:
    raw.mkdir()
    for name in P.PINNED_PHISH_FILES:
        (raw / name).write_text("", encoding="utf-8")


def test_execute_end_to_end_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {"s4x": (1, 4)}
    monkeypatch.setattr(B, "STRATA", tiny)
    raw = tmp_path / "raw"
    _empty_pinned_raw(raw)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"seed": 0, "domains": []}), encoding="utf-8")
    report = tmp_path / "rep.json"
    s3 = _FakeS3()
    athena = _FakeAthena(s3)
    monkeypatch.setattr(P, "_boto_clients", lambda region: (s3, athena))
    rc = P.main(
        [
            "--cache",
            str(cache),
            "--raw",
            str(raw),
            "--output",
            "s3://bkt/pfx",
            "--strata",
            "s4x",
            "--bar",
            "5",
            "--root-cap",
            "6",
            "--run-id",
            "testrun",
            "--execute",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    # Uploaded the enumerated candidates, queried both crawls, cleaned up.
    assert len(s3.puts) == 1 and s3.puts[0][0] == "bkt"
    assert s3.puts[0][1] == "pfx/probe-candidates/testrun/candidates.csv"
    kinds = ["CREATE" in s for s in athena.sqls]
    assert any(kinds) and any(s.startswith("DROP TABLE") for s in athena.sqls)
    assert s3.deletes == [("bkt", ["pfx/probe-candidates/testrun/candidates.csv"])]
    rep = json.loads(report.read_text(encoding="utf-8"))
    # First uploaded domain: primary 3; second: fallback 9 -> capped 6.
    uploaded = s3.puts[0][2].decode("utf-8").splitlines()[1:]
    assert rep["selectable_capped"] == 3 + 6
    assert rep["takeable"] == pytest.approx(9 * 0.99 * 0.72)
    assert rep["decision"] == "D1"
    assert rep["floor_reading"]["floor_viable"] is True
    counts = (tmp_path / "rep.counts.csv").read_text(encoding="utf-8")
    assert f"{uploaded[0]},3" in counts and f"{uploaded[1]},9" in counts
