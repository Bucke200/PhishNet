"""Offline tests for the D1 wave fetch (D0.6.1/D0.7) — no AWS, no boto3.

* the sample is the first-D_s fresh domains in replay order per stratum;
* the UNLOAD bounds in SQL (tested CASE, rn <= 6, Parquet, partitioned);
* fallback is primary-miss only via anti-join (no second upload);
* the execute path uploads/UNLOADs/repairs/drops with fake clients and
  writes the manifest; temp tables never survive by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import build_cc_benign as B
import fetch_cc_wave as W
import probe_cc_roots as P


def test_sample_arg_parsing() -> None:
    assert W.parse_sample_arg("s4_1k_10k=100,s5_10k_100k=200") == {
        "s4_1k_10k": 100,
        "s5_10k_100k": 200,
    }
    with pytest.raises(SystemExit):
        W.parse_sample_arg("s4_1k_10k=100,nope=5")
    with pytest.raises(SystemExit):
        W.parse_sample_arg("s4_1k_10k=many")


def test_unload_bounds_in_sql() -> None:
    sql = W.render_wave_unload(
        "ccindex", "wave_cands_x", "CC-MAIN-2026-34", "s3://bkt/pfx/", 2
    )
    assert "UNLOAD (" in sql and "WITH (format = 'PARQUET'" in sql
    assert "partitioned_by = ARRAY['stratum']" in sql
    assert "row_number() OVER (PARTITION BY c.domain, (" in sql
    assert (
        "ORDER BY abs(from_big_endian_64(xxhash64(to_utf8(concat(url, '|', '2')))))"
        in sql
    )
    assert "q.rn <= 6" in sql
    assert "ci.fetch_status = 200" in sql
    assert "CASE WHEN strpos(ci.url, '?') > 0 THEN 'query'" in sql
    assert "TO 's3://bkt/pfx/'" in sql
    fb = W.render_wave_unload(
        "ccindex",
        "wave_cands_x",
        "CC-MAIN-2026-30",
        "s3://bkt/pfx/",
        2,
        extra=" AND c.domain IN (SELECT 1)",
    )
    assert "AND c.domain IN (SELECT 1)" in fb


def test_wave_table_ddl_partitioned() -> None:
    ddl = W.render_wave_table_ddl("wave_primary_x", "s3://bkt/pfx/")
    assert "PARTITIONED BY (stratum string)" in ddl
    assert "STORED AS PARQUET" in ddl
    assert "stratum string, url string" not in ddl  # stratum is the partition key


def test_sample_csv_and_guard() -> None:
    csv = W.sample_csv(
        [("s4_1k_10k", 1001, "b.example"), ("s5_10k_100k", 5, "a.example")]
    )
    assert csv == "domain,stratum\nb.example,s4_1k_10k\na.example,s5_10k_100k\n"
    with pytest.raises(SystemExit):
        W.sample_csv([("s4_1k_10k", 1, "evil',example")])


def test_missing_sql_anti_join() -> None:
    sql = W.render_missing_sql("wave_cands_x", "wave_primary_x")
    assert "LEFT JOIN (SELECT DISTINCT domain AS d FROM wave_primary_x) m" in sql
    assert "WHERE m.d IS NULL" in sql


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
    """Canned pages: missing-query returns one domain; everything else empty."""

    def __init__(self) -> None:
        self.sqls: list[str] = []

    def start_query_execution(self, QueryString: str, **_: Any) -> dict[str, Any]:
        self.sqls.append(QueryString)
        return {"QueryExecutionId": f"q{len(self.sqls)}"}

    def get_query_execution(self, QueryExecutionId: str) -> dict[str, Any]:
        return {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {
                    "DataScannedInBytes": 100,
                    "EngineExecutionTimeInMillis": 1,
                    "TotalExecutionTimeInMillis": 2,
                },
            }
        }

    def _page(self, header: list[str], rows: list[list[str]]) -> dict[str, Any]:
        def cells(*vs: str) -> dict[str, Any]:
            return {"Data": [{"VarCharValue": v} for v in vs]}

        return {
            "ResultSet": {
                "ResultSetMetadata": {"ColumnInfo": [{"Name": c} for c in header]},
                "Rows": [cells(*header)] + [cells(*r) for r in rows],
            }
        }

    def get_query_results(self, QueryExecutionId: str, **_: Any) -> dict[str, Any]:
        qid = int(QueryExecutionId[1:])
        sql = self.sqls[qid - 1]
        if sql.startswith("SELECT c.domain FROM"):
            return self._page(["domain"], [["m.example"]])
        if "GROUP BY stratum" in sql:
            return self._page(["stratum", "n"], [["s4_1k_10k", "7"]])
        return self._page(["_col0"], [])


def _tiny_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"seed": 0, "domains": []}), encoding="utf-8")
    return cache


def _empty_pinned_raw(raw: Path) -> None:
    raw.mkdir()
    for name in P.PINNED_PHISH_FILES:
        (raw / name).write_text("", encoding="utf-8")


def test_sample_takes_replay_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {"s4_1k_10k": (1, 6)}
    monkeypatch.setattr(B, "STRATA", tiny)
    monkeypatch.setattr(W, "WAVE_STRATA", ("s4_1k_10k",))
    raw = tmp_path / "raw"
    _empty_pinned_raw(raw)
    manifest = tmp_path / "man.json"
    rc = W.main(
        [
            "--cache",
            str(_tiny_cache(tmp_path)),
            "--raw",
            str(raw),
            "--output",
            "s3://bkt/pfx",
            "--sample",
            "s4_1k_10k=3",
            "--run-id",
            "uptake",
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    man = json.loads(manifest.read_text(encoding="utf-8"))
    assert man["sample"]["per_stratum"] == {"s4_1k_10k": 3}
    assert man["sample"]["n_uploaded"] == 3
    assert man["executed"] is False


def test_execute_end_to_end_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {"s4_1k_10k": (1, 6)}
    monkeypatch.setattr(B, "STRATA", tiny)
    monkeypatch.setattr(W, "WAVE_STRATA", ("s4_1k_10k",))
    raw = tmp_path / "raw"
    _empty_pinned_raw(raw)
    manifest = tmp_path / "man.json"
    s3, athena = _FakeS3(), _FakeAthena()
    monkeypatch.setattr(W, "_boto", lambda region: (s3, athena))
    rc = W.main(
        [
            "--cache",
            str(_tiny_cache(tmp_path)),
            "--raw",
            str(raw),
            "--output",
            "s3://bkt/pfx",
            "--sample",
            "s4_1k_10k=3",
            "--run-id",
            "exerun",
            "--execute",
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    assert len(s3.puts) == 1 and s3.puts[0][0] == "bkt"
    assert s3.puts[0][1] == "pfx/cc-fetch-exerun/sample/candidates.csv"
    kinds = [s.split(" ")[0] for s in athena.sqls]
    assert "UNLOAD" in kinds and "CREATE" in kinds and "MSCK" in kinds
    assert any(s.startswith("DROP TABLE") for s in athena.sqls)
    assert s3.deletes == [("bkt", ["pfx/cc-fetch-exerun/sample/candidates.csv"])]
    man = json.loads(manifest.read_text(encoding="utf-8"))
    assert man["executed"] is True
    assert man["row_counts"]["primary"] == {"s4_1k_10k": "7"}
    assert man["n_missing_primary"] == 1
    assert man["queries"]["repair_primary"] == "MSCK REPAIR TABLE wave_primary_exerun"
