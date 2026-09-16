"""Exact COUNT probe for the Amendment D second-wave decision (M3).

Counts selectable new apex roots in s4-s6 beyond the banked main cache —
exactly, not by projection: the kept candidate domains are uploaded as a
small table under the Athena results bucket, and one JOIN query per crawl
partition counts DISTINCT root URLs per domain. Dry by default (prints the
DDL, both JOIN queries and the table locations; no boto3 import, no
network). ``--execute`` uploads, queries, drops the temp table and writes
the report; the fetch cache is only ever opened read-only, and results go
to ``--report`` (refused when it resolves to the cache path itself).

What counts as selectable mirrors selection: 200-status root captures
matched by ``regexp_like(url, '^https?://[^/]+/?$')`` — any host under the
domain (subdomain roots select normally; only synthesis is apex-only),
no LIKE wildcards on the host. Calibration (reports/probe-calibration.json)
shows raw DISTINCT root URLs ≈ dedup keys (UNIT 0.99) but quota-fill
competition takes only ~72% of the capped pool (roots fill last), so the
measured capped total converts to takeable roots at 0.99 x 0.72.
Primary crawl first with 2026-30 used only for domains with no 2026-34
capture (decided client-side per domain), phishing-tenant domains skipped
before upload, and the per-domain root cap applied to the measured counts.
Root synthesis is upside (<=1/domain with scheme evidence) and
deliberately uncounted.

Decision rule (D0.4/D1, D0.5): takeable fresh roots >= ``--bar``
(5,900) justifies the second fetch wave; otherwise D2. The report also
carries the 25k-floor reading (banked 3,960 + fresh takeable >= 5,900)
as context — it is NOT the decision rule.
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

import numpy as np

import build_cc_benign as B

_PINNED_DAYS = ("12", "13", "14", "15", "16")
PINNED_PHISH_FILES = sorted(
    f"{src}-2026-09-{day}.jsonl"
    for src in ("openphish", "phishtank")
    for day in _PINNED_DAYS
)

# Root-URL predicate: scheme + any single host (no '?' — a query string
# makes it url_type "query", and [^/]+ would absorb "?a=1" into the host),
# exact root path, optional fragment (normalise strips fragments, so a
# fragment-bearing root still selects). Verified against url_type over all
# 4.76M banked records: 99.92% agreement; residual mismatches are
# bare-trailing-'?' (SQL query vs url_type path — the query test below is
# deliberately strpos-ordered first), semicolon path-params, double-slash
# paths, and uppercase schemes, all documented in tests/test_cc_probe.py.
ROOT_URL_REGEX = "^https?://[^/?]+/?(#.*)?$"
PATH1_URL_REGEX = "^https?://[^/?]+/[^/?]+/?(#.*)?$"
PATHN_URL_REGEX = "^https?://[^/?]+/"

JOIN_TEMPLATE = (
    "SELECT c.domain, COUNT(DISTINCT ci.url) AS n_roots "
    "FROM {table} ci JOIN {cands} c "
    "ON ci.url_host_registered_domain = c.domain "
    "WHERE ci.crawl = '{crawl}' AND ci.subset = 'warc' "
    "AND ci.fetch_status = 200 "
    "AND regexp_like(ci.url, '{regex}') "
    "GROUP BY c.domain"
)

CANDIDATES_DDL_TEMPLATE = (
    "CREATE EXTERNAL TABLE {cands} (domain string) "
    "ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' "
    "STORED AS TEXTFILE LOCATION '{location}' "
    "TBLPROPERTIES ('skip.header.line.count'='1')"
)

BAR_DEFAULT = 5900
BANKED_TAKEABLE_ROOTS = 3960
# Calibration transfer (reports/probe-calibration.json, measured on the
# banked main pool vs the real unconstrained select take): raw DISTINCT
# root URLs convert to dedup keys at 0.9864 (rounded here, conservative
# direction for a go-decision is DOWN on the product), and quota-fill
# competition (roots fill last under the per-domain-total cap) takes
# 0.7248 of the capped tail pool. Product ~0.71.
UNIT_RATIO = 0.99
FILL_EFFICIENCY_TAIL = 0.72
_IDENT = re.compile(r"[A-Za-z0-9_]+")
_CRAWL = re.compile(r"[A-Za-z0-9_-]+")


def _ident(name: str, what: str) -> str:
    """Athena identifiers interpolate unquoted: reject anything exotic."""
    if not _IDENT.fullmatch(name):
        sys.exit(f"refusing: {what} {name!r} is not a plain identifier")
    return name


def _crawl(name: str) -> str:
    """Crawl partitions look like CC-MAIN-2026-34: allow the dash too."""
    if not _CRAWL.fullmatch(name):
        sys.exit(f"refusing: crawl {name!r} is not a plain crawl name")
    return name


def render_join_sql(table: str, cands: str, crawl: str) -> str:
    """Per-domain DISTINCT root-URL counts over one crawl partition."""
    _ident(table, "table")
    _ident(cands, "candidates table")
    return JOIN_TEMPLATE.format(
        table=table, cands=cands, crawl=_crawl(crawl), regex=ROOT_URL_REGEX
    )


def render_type_case(url_col: str = "ci.url") -> str:
    """SQL CASE classifying a URL column exactly like url_type().

    The single source the D1 fetch inlines for its per-(domain, type)
    row_number bound (D0.6.1/D0.7): query tested first by strpos (a '?'
    anywhere — url_type keys on a non-empty query, and the residual
    bare-trailing-'?' divergence is measured, not hidden), then root /
    path1 regexes, then any valid scheme+host as pathN, else malformed.
    Parity against url_type is locked in tests/test_cc_probe.py.
    """
    return (
        f"CASE WHEN strpos({url_col}, '?') > 0 THEN 'query' "
        f"WHEN regexp_like({url_col}, '{ROOT_URL_REGEX}') THEN 'root' "
        f"WHEN regexp_like({url_col}, '{PATH1_URL_REGEX}') THEN 'path1' "
        f"WHEN regexp_like({url_col}, '{PATHN_URL_REGEX}') THEN 'pathN' "
        "ELSE 'malformed' END"
    )


def sql_url_type(url: str) -> str:
    """Python mirror of render_type_case(), for parity testing only.

    Implements the CASE's semantics (strpos query test first, then the
    three regexes, then scheme+host validity) so tests can diff it
    against build_cc_benign.url_type over real URL samples.
    """
    import re as _re

    if "?" in url:
        return "query"
    if _re.match(ROOT_URL_REGEX, url):
        return "root"
    if _re.match(PATH1_URL_REGEX, url):
        return "path1"
    try:
        from urllib.parse import urlparse as _up

        p = _up(url)
        if p.scheme in ("http", "https") and p.netloc:
            return "pathN"
    except Exception:
        pass
    return "malformed"


def render_order_expr(seed: int) -> str:
    """Seeded hash order for the fetch row_number bound (D0.6.1/D0.7).

    The identical integer the sampling predicate thresholds on —
    abs(from_big_endian_64(...)) over xxhash64 of url+'|'+seed — so the
    kept prefix is a seeded uniform draw, never earliest-first. The seed
    rides inside the hashed input because Athena's xxhash64 takes a
    single varbinary argument with no seed parameter. Assumes Trino's
    xxhash64 is stock XXH64 (seed 0); pinned by vector test.
    """
    return (
        f"abs(from_big_endian_64(xxhash64(to_utf8(concat(url, '|', '{int(seed)}')))))"
    )


def render_candidates_ddl(cands: str, location: str) -> str:
    """External table over the uploaded candidate-domain CSV."""
    return CANDIDATES_DDL_TEMPLATE.format(
        cands=_ident(cands, "candidates table"), location=location.replace("'", "''")
    )


def render_drop_sql(cands: str) -> str:
    """Drop the temp candidates table after the counts are banked."""
    return f"DROP TABLE {_ident(cands, 'candidates table')}"


def replay_fresh(
    mapping: dict[int, str],
    cache_domains: list[dict[str, Any]],
    seed: int,
    wanted: list[str],
) -> list[tuple[str, int, str]]:
    """Fresh s4-s6 candidates in fetch-identical seeded order.

    Replays ``cmd_fetch`` exactly: one ``default_rng(seed)`` stream, one
    permutation per stratum in STRATA definition order (s1-s3 permutations
    are consumed and discarded so s4-s6 order matches a live resume), pool
    built rank-by-rank from the Tranco mapping. Entries with definitive
    outcomes (success or hard miss per ``_definitive``) are excluded;
    stale transient failures are NOT done — fetch would retry them.
    Returns (stratum, rank, domain) triples.
    """
    rng = np.random.default_rng(seed)
    fresh: list[tuple[str, int, str]] = []
    for sname, (lo, hi) in B.STRATA.items():
        pool = [mapping[r] for r in range(lo, hi + 1)]
        order = rng.permutation(len(pool))
        if sname not in wanted:
            continue
        done = {
            e["domain"]
            for e in cache_domains
            if e.get("stratum") == sname and B._definitive(e)
        }
        rank_of = {d: r for r, d in mapping.items() if lo <= r <= hi}
        for i in order:
            d = pool[int(i)]
            if d not in done:
                fresh.append((sname, rank_of[d], d))
    return fresh


def pinned_tenant_set(
    raw_dir: Path, files: list[str]
) -> tuple[set[str], dict[str, Any]]:
    """Tenant groups behind the pinned phishing files (D0.1).

    Same tenant semantics as ``phishing_tenant_set`` (tenant-level, URL
    never enters a benign corpus except through this exclusion) but over
    an explicit file list: a missing file is a hard error, so a grown
    ``data/raw`` can never silently shift the probe. Returns
    (tenants, inputs_record).
    """
    from phishnet.enrichment.key import tenant_group  # type: ignore[import-untyped]

    tenants: set[str] = set()
    hashes: dict[str, str] = {}
    n_rows = 0
    for name in files:
        f = raw_dir / name
        if not f.exists():
            sys.exit(f"pinned phishing file missing: {f} — refusing to proceed")
        hashes[name] = B.sha256_file(f)
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                url = json.loads(line).get("url")
            except (ValueError, AttributeError):
                continue
            if not isinstance(url, str):
                continue
            n_rows += 1
            tenants.add(tenant_group(url))
    return tenants, {
        "raw_dir": str(raw_dir),
        "files": hashes,
        "n_rows": n_rows,
        "n_tenants": len(tenants),
    }


def split_tenant_skipped(
    fresh: list[tuple[str, int, str]], tenants: set[str]
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Partition fresh candidates into (countable, tenant-skipped).

    A fresh domain whose own tenant sits in the phishing set yields zero
    selectable rows under the registered exclusion, so it never reaches
    the uploaded table — same exclusion, applied pre-query instead of
    post-fetch.
    """
    from phishnet.enrichment.key import tenant_group

    kept: list[tuple[str, int, str]] = []
    skipped: list[tuple[str, int, str]] = []
    for triple in fresh:
        if tenant_group(f"http://{triple[2]}/") in tenants:
            skipped.append(triple)
        else:
            kept.append(triple)
    return kept, skipped


def candidates_csv(domains: list[str]) -> str:
    """One-domain-per-line CSV (domains never contain commas or quotes)."""
    for d in domains:
        if "," in d or '"' in d or "'" in d or "\n" in d:
            sys.exit(f"refusing: candidate domain {d!r} is not CSV-safe")
    return "domain\n" + "".join(d + "\n" for d in domains)


def parse_domain_counts(rows: list[dict[str, str | None]]) -> dict[str, int]:
    """JOIN result rows -> {domain: n_roots} (positional, header-agnostic)."""
    out: dict[str, int] = {}
    for r in rows:
        vals = list(r.values())
        out[str(vals[0])] = int(str(vals[1]))
    return out


def prefer_primary(primary: dict[str, int], fallback: dict[str, int]) -> dict[str, int]:
    """Primary-first-fallback-on-miss, decided client-side per domain."""
    out = dict(primary)
    for d, n in fallback.items():
        if out.get(d, 0) == 0 and n > 0:
            out[d] = n
    return out


def cap_sum(counts: list[int], cap: int) -> int:
    """Measured pool: per-domain root cap applied to the per-domain counts."""
    return sum(min(c, cap) for c in counts)


def takeable_roots(
    capped_total: int,
    unit: float = UNIT_RATIO,
    fill: float = FILL_EFFICIENCY_TAIL,
) -> float:
    """Capped pool -> roots a select would actually take (calibrated)."""
    return capped_total * unit * fill


def decide(takeable: float, bar: int) -> str:
    """D1 iff takeable fresh roots reach the bar, else D2."""
    return "D1" if takeable >= bar else "D2"


def count_selectable_roots(
    entries: list[dict[str, Any]], tenants: set[str], cap: int
) -> dict[str, Any]:
    """Selectable-root counting over banked cache records (calibration).

    Mirrors what selection keeps for the root pool: 200-status only,
    normalise, url_type root, dedup_key earliest-wins (scheme + apex host
    + path + query — a www empirical root and its apex twin collapse to
    one key), phishing-tenant exclusion before counting, per-domain cap.
    Cross-type competition (per-domain-total) and the per-eTLD+1 cap are
    quota-fill effects, not pool properties, and are NOT applied: this
    counts the pool the fill draws from. Returns stage counts so the
    calibration against the real select take names its delta instead of
    hiding it. ``entries`` are cache records with url/timestamp/digest/
    status plus their entry's domain (seed) — the same slim shape
    ``columnar_records`` produces.
    """
    from phishnet.enrichment.key import tenant_group

    by_key: dict[str, dict[str, Any]] = {}
    stats = {"records_total": 0, "unselectable_status": 0, "unnormalisable": 0}
    for entry in entries:
        seed = str(entry.get("seed_domain") or entry.get("domain") or "")
        for rec in entry.get("records", []):
            stats["records_total"] += 1
            if str(rec.get("status") or "") != "200":
                stats["unselectable_status"] += 1
                continue
            norm = B.build_splits.normalise(rec.get("url") or "")
            if norm is None:
                stats["unnormalisable"] += 1
                continue
            if B.url_type(norm) != "root":
                continue
            key = B.dedup_key(norm)
            prev = by_key.get(key)
            if prev is not None:
                if str(rec.get("timestamp") or "") < str(prev["cc_timestamp"]):
                    prev.update(
                        {
                            "url": norm,
                            "cc_timestamp": rec.get("timestamp"),
                            "seed_domain": seed,
                        }
                    )
                continue
            by_key[key] = {
                "url": norm,
                "cc_timestamp": rec.get("timestamp"),
                "seed_domain": seed,
            }
    n_deduped = len(by_key)
    kept = [r for r in by_key.values() if tenant_group(r["url"]) not in tenants]
    n_after_exclusion = len(kept)
    per_domain: dict[str, int] = {}
    for r in kept:
        per_domain[r["seed_domain"]] = per_domain.get(r["seed_domain"], 0) + 1
    capped = sum(min(c, cap) for c in per_domain.values())
    return {
        "records_total": stats["records_total"],
        "unselectable_status": stats["unselectable_status"],
        "unnormalisable": stats["unnormalisable"],
        "n_deduped_roots": n_deduped,
        "n_after_tenant_exclusion": n_after_exclusion,
        "n_domains": len(per_domain),
        "selectable_capped": capped,
        "per_domain_cap": cap,
    }


def split_s3_url(url: str) -> tuple[str, str]:
    """s3://bucket/prefix -> (bucket, prefix)."""
    if not url.startswith("s3://"):
        sys.exit(f"refusing: {url!r} is not an s3:// URL")
    rest = url[len("s3://") :].strip("/")
    bucket, _, prefix = rest.partition("/")
    if not bucket or not prefix:
        sys.exit(f"refusing: {url!r} needs a bucket AND a prefix")
    return bucket, prefix


def _boto_clients(region: str | None) -> tuple[Any, Any]:
    """Lazily build (s3, athena) clients — boto3 stays execute-only."""
    try:
        boto3_mod: Any = importlib.import_module("boto3")
    except ImportError as e:
        raise SystemExit(
            "probe --execute needs boto3 plus AWS credentials "
            "(ephemeral: uv run --with boto3). No live call is made "
            "without --execute."
        ) from e
    kw: dict[str, Any] = {"region_name": region} if region else {}
    return boto3_mod.client("s3", **kw), boto3_mod.client("athena", **kw)


def run_join(
    athena: Any, sql: str, database: str, output: str, max_wait: float
) -> tuple[dict[str, int], dict[str, Any]]:
    """One JOIN query -> (per-domain counts, byte stats)."""
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
    return parse_domain_counts(rows), stats


def build_parser() -> argparse.ArgumentParser:
    """CLI: dry by default; --execute uploads and runs the two JOINs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/raw/cc-columnar-CC-MAIN-2026-34.json")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--table", default="ccindex")
    p.add_argument("--database", default="ccindex")
    p.add_argument(
        "--output",
        default="s3://phishnet-athena/probe/",
        help="s3://bucket/prefix for Athena results AND the temp "
        "candidates table (required with --execute)",
    )
    p.add_argument("--primary", default=B.CC_INDEX_PRIMARY)
    p.add_argument("--fallback", default=B.CC_INDEX_FALLBACK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strata", default="s4_1k_10k,s5_10k_100k,s6_100k_1M")
    p.add_argument("--bar", type=int, default=BAR_DEFAULT)
    p.add_argument("--root-cap", type=int, default=B.PER_DOMAIN_TYPE_CAP)
    p.add_argument("--region", default=None)
    p.add_argument(
        "--run-id",
        default=None,
        help="temp table suffix (default: UTC timestamp + seed)",
    )
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="leave the candidates table + CSV in place for audit",
    )
    p.add_argument("--max-wait-secs", type=float, default=3600.0)
    p.add_argument(
        "--execute",
        action="store_true",
        help="upload candidates and run the live JOIN queries "
        "(needs boto3 + AWS credentials)",
    )
    p.add_argument("--report", required=True, help="probe report JSON output path")
    return p


def main(argv: list[str] | None = None) -> int:
    """Enumerate (always) then JOIN (only with --execute); write --report."""
    a = build_parser().parse_args(argv)
    cache_path = Path(a.cache)
    report_path = Path(a.report)
    if report_path.resolve() == cache_path.resolve():
        sys.exit("refusing: --report must not be the fetch cache itself")
    _ident(str(a.table), "table")
    _ident(str(a.database), "database")

    mapping = B.load_tranco()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("seed") != a.seed:
        sys.exit(f"cache seed {cache.get('seed')} != --seed {a.seed}")
    strata = [s.strip() for s in str(a.strata).split(",") if s.strip()]
    fresh = replay_fresh(mapping, cache["domains"], a.seed, strata)
    per_stratum: dict[str, int] = {}
    for sname, _, _ in fresh:
        per_stratum[sname] = per_stratum.get(sname, 0) + 1

    tenants, tenant_inputs = pinned_tenant_set(Path(a.raw), PINNED_PHISH_FILES)
    kept, skipped = split_tenant_skipped(fresh, tenants)
    kept_domains = [d for _, _, d in kept]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = a.run_id or f"m3_s{a.seed}_{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        sys.exit(f"refusing: --run-id {run_id!r} is not a plain identifier")
    cands = f"probe_cands_{run_id}"
    bucket, prefix = split_s3_url(str(a.output))
    location = f"s3://{bucket}/{prefix}/probe-candidates/{run_id}/"
    ddl = render_candidates_ddl(cands, location)
    sql_primary = render_join_sql(str(a.table), cands, str(a.primary))
    sql_fallback = render_join_sql(str(a.table), cands, str(a.fallback))
    drop_sql = render_drop_sql(cands)

    print(f"fresh s4-s6 candidates: {len(fresh)} {per_stratum}")
    print(f"tenant-skipped (never uploaded): {len(skipped)}")
    print(f"upload table: {cands} ({len(kept_domains)} domains) at {location}")
    print(f"DDL: {ddl}")
    print(f"primary JOIN: {sql_primary}")
    print(f"fallback JOIN: {sql_fallback}")

    report: dict[str, Any] = {
        "probe": "probe_cc_roots join (Amendment D M3, D0.5)",
        "bar": a.bar,
        "root_cap": a.root_cap,
        "run_id": run_id,
        "inputs": {
            "cache": str(cache_path),
            "cache_sha256": B.sha256_file(cache_path),
            "cache_seed": cache.get("seed"),
            "seed": a.seed,
            "strata": strata,
            "table": a.table,
            "database": a.database,
            "crawls": {"primary": a.primary, "fallback": a.fallback},
            "candidates_table": cands,
            "candidates_location": location,
            "ddl": ddl,
            "sql_primary": sql_primary,
            "sql_fallback": sql_fallback,
            "tenant_inputs": tenant_inputs,
        },
        "enumeration": {
            "n_fresh": len(fresh),
            "per_stratum": per_stratum,
            "n_tenant_skipped": len(skipped),
            "n_uploaded": len(kept_domains),
        },
        "executed": False,
    }

    if not a.execute:
        print(
            "dry run: nothing uploaded, no queries issued "
            "(pass --execute to COUNT). "
            f"Rule: takeable (capped x {UNIT_RATIO} x {FILL_EFFICIENCY_TAIL}) "
            f">= {a.bar} -> D1, else D2."
        )
        report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\r\n"
        )
        print(f"wrote {report_path}")
        return 0

    s3, athena = _boto_clients(a.region)
    csv_text = candidates_csv(kept_domains)
    csv_key = f"{prefix}/probe-candidates/{run_id}/candidates.csv"
    stats_all: list[dict[str, Any]] = []
    try:
        s3.put_object(Bucket=bucket, Key=csv_key, Body=csv_text.encode("utf-8"))
        print(f"uploaded {len(kept_domains)} domains to s3://{bucket}/{csv_key}")
        B.athena_query_rows(
            athena,
            ddl,
            str(a.database),
            str(a.output),
            poll_interval=5.0,
            max_wait=float(a.max_wait_secs),
        )
        primary_counts, st_p = run_join(
            athena, sql_primary, str(a.database), str(a.output), float(a.max_wait_secs)
        )
        fallback_counts, st_f = run_join(
            athena, sql_fallback, str(a.database), str(a.output), float(a.max_wait_secs)
        )
        stats_all = [
            {"crawl": str(a.primary), **st_p},
            {"crawl": str(a.fallback), **st_f},
        ]
    finally:
        if not a.keep_artifacts:
            try:
                B.athena_query_rows(
                    athena,
                    drop_sql,
                    str(a.database),
                    str(a.output),
                    poll_interval=5.0,
                    max_wait=300.0,
                )
                s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": csv_key}]})
                print(f"dropped {cands} and deleted {csv_key}")
            except Exception as e:  # cleanup is best-effort; report survives
                print(f"cleanup warning: {e}", file=sys.stderr)

    final = prefer_primary(primary_counts, fallback_counts)
    measured = [final.get(d, 0) for d in kept_domains]
    capped = cap_sum(measured, a.root_cap)
    takeable = takeable_roots(capped)
    verdict = decide(takeable, a.bar)
    floor_need = max(0, a.bar - BANKED_TAKEABLE_ROOTS)
    scanned = sum(int(s.get("data_scanned_bytes", 0)) for s in stats_all)
    counts_path = report_path.with_name(report_path.stem + ".counts.csv")
    counts_path.write_text(
        "domain,n_roots\n"
        + "".join(f"{d},{final[d]}\n" for d in kept_domains if final.get(d, 0) > 0),
        encoding="utf-8",
        newline="\r\n",
    )
    report.update(
        {
            "executed": True,
            "n_domains_with_captures": sum(
                1 for d in kept_domains if final.get(d, 0) > 0
            ),
            "raw_root_url_sum": sum(measured),
            "selectable_capped": capped,
            "unit_ratio": UNIT_RATIO,
            "fill_efficiency_tail": FILL_EFFICIENCY_TAIL,
            "takeable": takeable,
            "floor_reading": {
                "banked_takeable": BANKED_TAKEABLE_ROOTS,
                "fresh_needed_for_25k": floor_need,
                "floor_viable": bool(takeable >= floor_need),
                "note": "context only: the decision rule is takeable >= bar; "
                "adopting the 25k-floor reading needs a D0 amendment first",
            },
            "data_scanned_bytes": scanned,
            "query_stats": stats_all,
            "counts_csv": str(counts_path),
            "kept_artifacts": bool(a.keep_artifacts),
            "decision": verdict,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\r\n"
    )
    print(f"capped={capped} takeable={takeable:.0f} bar={a.bar} -> {verdict}")
    print(f"scanned {scanned} bytes; wrote {report_path} + {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
