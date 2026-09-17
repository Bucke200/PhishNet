"""D1 wave fetch (Amendment D, D0.6.1/D0.7): join-bounded Athena fetch.

Fetches a seeded bounded domain sample per stratum with ONE join query
per crawl — never per-domain queries. The sample (D0.7.2 counts) is the
first-D_s fresh domains in fetch-identical replay order; selection
consumes them in that order and stops when full, so the margin is
insurance, never a best-fit pool. In-SQL bounds: url_type via the
tested CASE (probe render_type_case), 200-only, row_number() over
(domain, url_type) in seeded-hash order keeping rn <= 6. Output goes to
Parquet in S3 (UNLOAD), never into the JSON cache; a manifest records
queries, locations, row counts and bytes. Dry by default (prints DDL +
SQL + locations; no boto3 import, no network). --execute uploads,
fetches, drops temp tables and writes the manifest.

Primary-first-fallback-on-miss without per-domain queries: UNLOAD the
primary join for all sampled domains, expose it as an external table,
then UNLOAD the fallback join with an anti-join for domains missing
from the primary output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_cc_benign as B
import probe_cc_roots as P

# D0.7.2 bounded samples: first-D_s fresh domains in replay order.
WAVE_SAMPLE_DEFAULTS = {"s4_1k_10k": 7963, "s5_10k_100k": 12760, "s6_100k_1M": 9320}
WAVE_STRATA = ("s4_1k_10k", "s5_10k_100k", "s6_100k_1M")
FETCH_ROW_CAP = 6  # rn <= 6 per (domain, url_type), the per-domain-type cap

_IDENT = re.compile(r"[A-Za-z0-9_]+")


def _ident(name: str, what: str) -> str:
    if not _IDENT.fullmatch(name):
        sys.exit(f"refusing: {what} {name!r} is not a plain identifier")
    return name


def render_sample_ddl(cands: str, location: str) -> str:
    """External table over the uploaded (domain, stratum) sample CSV."""
    return (
        f"CREATE EXTERNAL TABLE {_ident(cands, 'sample table')} "
        "(domain string, stratum string) "
        "ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' "
        f"STORED AS TEXTFILE LOCATION '{location.replace(chr(39), chr(39) * 2)}' "
        "TBLPROPERTIES ('skip.header.line.count'='1')"
    )


def render_wave_unload(
    table: str, cands: str, crawl: str, location: str, seed: int, extra: str = ""
) -> str:
    """UNLOAD the bounded join for one crawl to Parquet.

    ``extra`` is an additional AND predicate (the fallback anti-join).
    The url_type CASE and the seeded-hash order expression are the
    tested probe renderers — the fetch inlines them, never reinvents.
    """
    case = P.render_type_case("ci.url")
    order = P.render_order_expr(seed)
    return (
        "UNLOAD (SELECT q.domain, q.stratum, q.url, q.fetch_time, "
        "q.fetch_status, q.content_digest, q.content_mime_type, q.url_type "
        "FROM (SELECT c.domain AS domain, c.stratum AS stratum, "
        "ci.url AS url, ci.fetch_time AS fetch_time, "
        "ci.fetch_status AS fetch_status, "
        "ci.content_digest AS content_digest, "
        "ci.content_mime_type AS content_mime_type, "
        f"({case}) AS url_type, "
        "row_number() OVER "
        f"(PARTITION BY c.domain, ({case}) ORDER BY {order}) AS rn "
        f"FROM {_ident(table, 'table')} ci "
        f"JOIN {_ident(cands, 'sample table')} c "
        "ON ci.url_host_registered_domain = c.domain "
        f"WHERE ci.crawl = '{P._crawl(crawl)}' AND ci.subset = 'warc' "
        f"AND ci.fetch_status = 200{extra}) q "
        f"WHERE q.rn <= {FETCH_ROW_CAP}) "
        f"TO '{location.replace(chr(39), chr(39) * 2)}' "
        "WITH (format = 'PARQUET', partitioned_by = ARRAY['stratum'])"
    )


def render_wave_table_ddl(name: str, location: str) -> str:
    """External table exposing one UNLOAD output dir (Parquet, by stratum)."""
    return (
        f"CREATE EXTERNAL TABLE {_ident(name, 'wave table')} "
        "(domain string, url string, fetch_time string, "
        "fetch_status int, content_digest string, content_mime_type string, "
        "url_type string) PARTITIONED BY (stratum string) "
        f"STORED AS PARQUET LOCATION '{location.replace(chr(39), chr(39) * 2)}'"
    )


def render_missing_sql(cands: str, wave_primary: str) -> str:
    """Sampled domains with no primary-crawl output rows."""
    cands_q = _ident(cands, "sample table")
    wave_q = _ident(wave_primary, "wave table")
    return (
        f"SELECT c.domain FROM {cands_q} c "
        f"LEFT JOIN (SELECT DISTINCT domain AS d FROM {wave_q}) m "
        "ON m.d = c.domain WHERE m.d IS NULL"
    )


def sample_csv(triples: list[tuple[str, int, str]]) -> str:
    """(domain, stratum) CSV for upload."""
    lines = ["domain,stratum"]
    for sname, _, dom in triples:
        if any(c in dom + sname for c in (",", '"', "'", "\n")):
            sys.exit(f"refusing: sample value {dom!r}/{sname!r} is not CSV-safe")
        lines.append(f"{dom},{sname}")
    return "\n".join(lines) + "\n"


def parse_sample_arg(spec: str) -> dict[str, int]:
    """'s4_1k_10k=100,s5_10k_100k=200' -> per-stratum sample sizes."""
    out: dict[str, int] = {}
    for chunk in spec.split(","):
        name, _, num = chunk.partition("=")
        name, num = name.strip(), num.strip()
        if name not in WAVE_STRATA or not num.isdigit():
            sys.exit(f"refusing: bad --sample entry {chunk!r}")
        out[name] = int(num)
    return out


def split_s3(url: str) -> tuple[str, str]:
    """s3://bucket/prefix -> (bucket, prefix)."""
    return P.split_s3_url(url)


def _boto(region: str | None) -> tuple[Any, Any]:
    try:
        mod: Any = importlib.import_module("boto3")
    except ImportError as e:
        raise SystemExit(
            "fetch --execute needs boto3 plus AWS credentials "
            "(ephemeral: uv run --with boto3). Dry runs never import it."
        ) from e
    kw: dict[str, Any] = {"region_name": region} if region else {}
    return mod.client("s3", **kw), mod.client("athena", **kw)


def run_sql(
    athena: Any, sql: str, database: str, output: str, max_wait: float
) -> tuple[list[dict[str, str | None]], dict[str, Any]]:
    """One statement -> (rows, byte stats)."""
    stats: dict[str, Any] = {}
    rows = B.athena_query_rows(
        athena,
        sql,
        database,
        output,
        poll_interval=5.0,
        max_wait=max_wait,
        stats_out=stats,
    )
    return rows, stats


def build_parser() -> argparse.ArgumentParser:
    """CLI: dry by default; --execute runs the wave fetch."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/raw/cc-columnar-CC-MAIN-2026-34.json")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--table", default="ccindex")
    p.add_argument("--database", default="ccindex")
    p.add_argument(
        "--output",
        default="s3://phishnet-athena/wave/",
        help="s3://bucket/prefix for Athena results AND wave Parquet",
    )
    p.add_argument("--primary", default=B.CC_INDEX_PRIMARY)
    p.add_argument("--fallback", default=B.CC_INDEX_FALLBACK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample",
        default=",".join(f"{s}={WAVE_SAMPLE_DEFAULTS[s]}" for s in WAVE_STRATA),
        help="per-stratum fresh-domain sample sizes (D0.7.2 defaults)",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="hash seed for the row_number order (default: --seed + 2)",
    )
    p.add_argument("--region", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--keep-artifacts", action="store_true")
    p.add_argument("--max-wait-secs", type=float, default=7200.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--manifest", required=True, help="fetch manifest JSON output path")
    return p


def main(argv: list[str] | None = None) -> int:
    """Enumerate the sample (always); fetch it (only with --execute)."""
    a = build_parser().parse_args(argv)
    manifest_path = Path(a.manifest)
    cache_path = Path(a.cache)
    if manifest_path.resolve() == cache_path.resolve():
        sys.exit("refusing: --manifest must not be the fetch cache itself")
    _ident(str(a.table), "table")
    _ident(str(a.database), "database")
    sample_n = parse_sample_arg(str(a.sample))
    sample_seed = a.sample_seed if a.sample_seed is not None else a.seed + 2

    mapping = B.load_tranco()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("seed") != a.seed:
        sys.exit(f"cache seed {cache.get('seed')} != --seed {a.seed}")
    fresh = P.replay_fresh(mapping, cache["domains"], a.seed, list(WAVE_STRATA))
    by_stratum: dict[str, list[tuple[str, int, str]]] = {s: [] for s in WAVE_STRATA}
    for triple in fresh:
        by_stratum[triple[0]].append(triple)
    sample = [t for s in WAVE_STRATA for t in by_stratum[s][: sample_n[s]]]
    per_stratum = {s: len([t for t in sample if t[0] == s]) for s in WAVE_STRATA}

    tenants, tenant_inputs = P.pinned_tenant_set(Path(a.raw), P.PINNED_PHISH_FILES)
    from phishnet.enrichment.key import tenant_group

    kept = [t for t in sample if tenant_group(f"http://{t[2]}/") not in tenants]
    skipped = len(sample) - len(kept)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = a.run_id or f"wave_s{a.seed}_{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        sys.exit(f"refusing: --run-id {run_id!r} is not a plain identifier")
    cands = f"wave_cands_{run_id}"
    bucket, prefix = split_s3(str(a.output))
    base = f"s3://{bucket}/{prefix}/cc-fetch-{run_id}/"
    ddl = render_sample_ddl(cands, base + "sample/")
    unload_p = render_wave_unload(
        str(a.table), cands, str(a.primary), base + "primary/", sample_seed
    )
    w_primary = f"wave_primary_{run_id}"
    ddl_primary = render_wave_table_ddl(w_primary, base + "primary/")
    missing_sql = render_missing_sql(cands, w_primary)
    unload_f = render_wave_unload(
        str(a.table),
        cands,
        str(a.fallback),
        base + "fallback/",
        sample_seed,
        extra=f" AND c.domain IN ({missing_sql})",
    )
    w_fallback = f"wave_fallback_{run_id}"
    ddl_fallback = render_wave_table_ddl(w_fallback, base + "fallback/")
    drop_p = P.render_drop_sql(w_primary)
    drop_f = P.render_drop_sql(w_fallback)
    drop_c = P.render_drop_sql(cands)
    repair_p = f"MSCK REPAIR TABLE {w_primary}"
    repair_f = f"MSCK REPAIR TABLE {w_fallback}"

    print(f"fresh available: { {s: len(by_stratum[s]) for s in WAVE_STRATA} }")
    print(f"sample: {per_stratum} total={len(sample)} tenant-skipped={skipped}")
    print(f"DDL: {ddl}")
    print(f"primary UNLOAD: {unload_p}")
    print(f"fallback UNLOAD: {unload_f}")

    manifest: dict[str, Any] = {
        "fetch": "fetch_cc_wave (Amendment D D1, D0.6.1/D0.7)",
        "run_id": run_id,
        "seed": a.seed,
        "sample_seed": sample_seed,
        "inputs": {
            "cache": str(cache_path),
            "cache_sha256": B.sha256_file(cache_path),
            "table": a.table,
            "database": a.database,
            "crawls": {"primary": a.primary, "fallback": a.fallback},
            "tenant_inputs": tenant_inputs,
        },
        "sample": {
            "per_stratum": per_stratum,
            "n_sampled": len(sample),
            "n_tenant_skipped": skipped,
            "n_uploaded": len(kept),
            "candidates_table": cands,
            "candidates_location": base + "sample/",
            "ddl": ddl,
        },
        "queries": {
            "unload_primary": unload_p,
            "wave_primary_ddl": ddl_primary,
            "repair_primary": repair_p,
            "missing_sql": missing_sql,
            "unload_fallback": unload_f,
            "wave_fallback_ddl": ddl_fallback,
            "repair_fallback": repair_f,
        },
        "output": {
            "base": base,
            "primary": base + "primary/",
            "fallback": base + "fallback/",
        },
        "executed": False,
    }
    if not a.execute:
        print("dry run: nothing uploaded, no queries issued (pass --execute).")
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8", newline="\r\n"
        )
        print(f"wrote {manifest_path}")
        return 0

    s3, athena = _boto(a.region)
    csv_text = sample_csv(kept)
    csv_key = f"{prefix}/cc-fetch-{run_id}/sample/candidates.csv"
    stats_all: list[dict[str, Any]] = []
    try:
        s3.put_object(Bucket=bucket, Key=csv_key, Body=csv_text.encode("utf-8"))
        print(f"uploaded {len(kept)} domains to s3://{bucket}/{csv_key}")
        B.athena_query_rows(
            athena,
            ddl,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=float(a.max_wait_secs),
        )
        _, st_u = run_sql(
            athena, unload_p, str(a.database), str(a.output), float(a.max_wait_secs)
        )
        B.athena_query_rows(
            athena,
            ddl_primary,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=300.0,
        )
        B.athena_query_rows(
            athena,
            repair_p,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=300.0,
        )
        missing, _ = run_sql(athena, missing_sql, str(a.database), str(a.output), 600.0)
        n_missing = len(missing)
        print(f"domains missing from primary: {n_missing}")
        B.athena_query_rows(
            athena,
            ddl_fallback,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=300.0,
        )
        B.athena_query_rows(
            athena,
            repair_f,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=300.0,
        )
        _, st_f = run_sql(
            athena, unload_f, str(a.database), str(a.output), float(a.max_wait_secs)
        )
        counts_p, _ = run_sql(
            athena,
            f"SELECT stratum, COUNT(*) AS n FROM {w_primary} GROUP BY stratum",
            str(a.database),
            str(a.output),
            600.0,
        )
        counts_f, _ = run_sql(
            athena,
            f"SELECT stratum, COUNT(*) AS n FROM {w_fallback} GROUP BY stratum",
            str(a.database),
            str(a.output),
            600.0,
        )
        stats_all = [
            {"query": "unload_primary", **st_u},
            {"query": "unload_fallback", **st_f},
        ]
    finally:
        if not a.keep_artifacts:
            for stmt in (drop_p, drop_f, drop_c):
                try:
                    B.athena_query_rows(
                        athena,
                        stmt,
                        str(a.database),
                        str(a.output),
                        poll_interval=5.0,
                        max_wait=300.0,
                    )
                except Exception as e:  # cleanup best-effort; manifest survives
                    print(f"cleanup warning: {e}", file=sys.stderr)
            try:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": csv_key}]})
            except Exception as e:
                print(f"cleanup warning: {e}", file=sys.stderr)
    scanned = sum(int(s.get("data_scanned_bytes", 0)) for s in stats_all)
    manifest.update(
        {
            "executed": True,
            "n_missing_primary": n_missing,
            "row_counts": {
                "primary": {r["stratum"]: r["n"] for r in counts_p},
                "fallback": {r["stratum"]: r["n"] for r in counts_f},
            },
            "data_scanned_bytes": scanned,
            "query_stats": stats_all,
            "kept_artifacts": bool(a.keep_artifacts),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\r\n"
    )
    print(f"scanned {scanned} bytes; wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
