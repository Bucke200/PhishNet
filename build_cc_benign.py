"""Build a benign URL corpus from Common Crawl's URL index (no crawling).

Acquisition mechanism (default): the columnar index on S3
(s3://commoncrawl/cc-index/table/cc-main/warc/), queried per-domain with
Amazon Athena (needs boto3 + AWS credentials; see docs). The CDX
front-end (index.commoncrawl.org) remains available via --source cdx for
small probes only — it throttles bulk fetching.

Design (see module docstring for rationale):
* Popularity: seed domains drawn from the frozen Tranco 46VQX 1M list in
  six log-spaced rank strata with equal productive domains per stratum
  (approximately log-uniform across popularity). Target ~2,100 productive
  registrable domains so domain-grouped splits keep a usable test side.
* Query form: ``matchType=domain`` (``url=<domain>&matchType=domain``), so
  one query covers apex + www + subdomains. A bare ``url=<domain>/*``
  prefix query misses every subdomain capture (SURT-prefix behaviour) and
  404s even for facebook.com — that form is superseded, see v2 note below.
* URL type: per-type quotas measured from the deduplicated phishing feeds
  already in data/raw (root / path1 / pathN / query), so depth aligns by
  construction instead of post-hoc matching. Deep types (path1/pathN/query)
  are always empirical index records. Roots are empirical index records
  first; the residual root quota is filled with synthesised apex roots
  (``<scheme>://<apex>/``), one per scheme with evidence on the apex host
  itself — never a default scheme, never www (see the host-form rule:
  apex-only synthesis and apex-based dedup keys; domains with no apex
  capture at all are skipped for backfill and counted).
* Representation: every deep URL is a record returned by the CC index
  itself. No page fetching, no link following. Synthesised roots carry
  ``synthesized_root: true`` + per-domain scheme evidence, so they are
  auditable and removable.
* Scheme discipline: synthesised roots take only schemes observed for
  their domain (seeded sampling is unnecessary — at most one root per
  observed scheme). Never default to https://; the phishing reference is
  91.0% https overall / 81.9% on roots, and validation reports the benign
  rates for comparison.
* Provenance: each row carries cc_* fields plus the Tranco stratum/rank;
  a .provenance.json sidecar records inputs, hashes, seeds and counts.

Outputs (new artifacts; the frozen baseline is never touched):
  data/raw/benign-cc-<INDEX>-<date>.jsonl
  data/raw/benign-cc-<INDEX>-<date>.provenance.json
  data/raw/cc-columnar-<INDEX>.json   (cached raw index responses)

Usage:
  python build_cc_benign.py --phase all
  python build_cc_benign.py --phase fetch    # only query + cache the index
  python build_cc_benign.py --phase select   # build corpus from the cache
  python build_cc_benign.py --phase fetch --source cdx   # small probes only
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

import build_splits

TRANC0_CSV = Path("data/raw/tranco-46VQX-top1000000-2026-09-13.csv")
TRANC0_SHA256 = "4fb2f1c0644673cf2730161ee71d835916f9da0de6fdadc8b29026160fd81d2b"
TRANC0_ID = "46VQX"

CC_INDEX_PRIMARY = "CC-MAIN-2026-34"  # Aug 2026 crawl, pre-feed
CC_INDEX_FALLBACK = "CC-MAIN-2026-30"  # Jul 2026 crawl
CC_BASE = "https://index.commoncrawl.org"
UA = "PhishNetResearchBot/0.1 (+https://github.com/Bucke200/PhishNet; academic project)"
# matchType=domain covers apex + www + all subdomains in one query. The
# earlier bare-prefix form (url=<domain>/*) misses subdomain captures by
# SURT-prefix construction and 404s on domains that demonstrably have
# captures (e.g. facebook.com); caches written with it are superseded.
QUERY_FORM = "matchType=domain"

# Log-spaced rank strata over the 1M list (inclusive bounds).
STRATA = {
    "s1_top10": (1, 10),
    "s2_11_100": (11, 100),
    "s3_101_1k": (101, 1000),
    "s4_1k_10k": (1001, 10_000),
    "s5_10k_100k": (10_001, 100_000),
    "s6_100k_1M": (100_001, 1_000_000),
}

# URL-type targets measured from deduplicated phishing feeds in data/raw
# (75,833 unique normalized URLs: root .3635 / path1 .3600 / pathN .1280 /
# query .1485; zero malformed after normalisation).
TYPE_TARGETS = {"root": 0.3635, "path1": 0.3600, "pathN": 0.1280, "query": 0.1485}

SEED_DEFAULT = 0
TARGET_N_DEFAULT = 12_000
# 700: bootstrap over the 26-domain pilot yield distribution (heavy tail:
# median 2 capped roots/domain) meets the 4,362 root quota with P>=0.9
# even with the top-3 infra giants removed (N=2,900: mean 4,672, p10
# 4,564); N=2,100 fails that weak-tail case outright (mean 3,383).
# s3 pool (900 candidates) caps P at ~729, so 700 fits everywhere.
PRODUCTIVE_PER_STRATUM_DEFAULT = 700  # s1/s2 pools cap below this (10/90)
CANDIDATES_PER_STRATUM = 2000
# Per-domain row cap on raw Athena rows (pre-mapping). Micro-measured on
# 3 head domains: earliest-first ordering skews URL-type mix by up to ~5pp
# (samsung query-ward) vs a seeded uniform sample, so the cap samples
# uniformly: sort by URL (deterministic content order, NOT retrieval
# order, which Athena does not guarantee) then take a seeded permutation
# prefix. Retrieval wall time is unchanged — the cap bounds cache bytes
# and select-stage memory, not pagination.
ROW_CAP_PER_DOMAIN_DEFAULT = 5000
PER_DOMAIN_TYPE_CAP = 6
PER_DOMAIN_TOTAL_CAP = 16
PER_ETLD1_CAP = 25
WORKERS_CDX_DEFAULT = 1  # index.commoncrawl.org throttles bursts; stay polite
WORKERS_COLUMNAR_DEFAULT = 5  # within the Athena concurrent-query quota
CACHE_DEFAULT_CDX = "data/raw/cc-index-CC-MAIN-2026-34-matchdomain.json"
CACHE_DEFAULT_COLUMNAR = "data/raw/cc-columnar-CC-MAIN-2026-34.json"

# Columnar index on S3 (primary mechanism). Table setup: CREATE DATABASE,
# then the flat-schema DDL from commoncrawl/cc-index-table
# (src/sql/athena/cc-index-create-table-flat.sql) pointed at
# CC_TABLE_S3, then MSCK REPAIR TABLE. Column reference:
# https://data.commoncrawl.org/cc-index/table/cc-main/index.html
CC_TABLE_S3 = "s3://commoncrawl/cc-index/table/cc-main/warc/"
ATHENA_SUBSET = "warc"
ATHENA_TABLE_DEFAULT = "ccindex"
ATHENA_DATABASE_DEFAULT = "ccindex"
COLUMNAR_SQL_TEMPLATE = (
    "SELECT url, fetch_time, fetch_status, content_digest, content_mime_type "
    "FROM {table} WHERE crawl = '{crawl}' AND subset = 'warc' "
    "AND url_host_registered_domain = '{domain}'"
)
# NOTE: no fetch_status predicate here, deliberately. Scheme evidence is
# counted over ALL returned rows (200s, 301/302 redirects, errors), while
# URL selection keeps only fetch_status = 200 (see columnar_records).
# Filtering evidence to 200s would hide http->https redirects and make the
# "observed scheme" measure the filter instead of what the domain serves.


def _definitive(entry: dict[str, Any]) -> bool:
    """True iff the cached outcome needs no retry (success or hard miss).

    Timeouts, connection errors and 5xx are transient (e.g. server-side
    throttling after bursts) and must be retried, never mistaken for
    "domain has no captures".
    """
    if entry.get("index") is not None:
        return True
    note = str(entry.get("note", ""))
    return note in ("no-usable-captures", "http-404") or note.endswith(
        ("(primary:no-usable-captures)", "(primary:http-404)")
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tranco() -> dict[int, str]:
    raw = TRANC0_CSV.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRANC0_SHA256:
        sys.exit(f"sha256 mismatch for {TRANC0_CSV} — refusing to proceed")
    mapping: dict[int, str] = {}
    for line in raw.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rank_raw, sep, domain = line.partition(",")
        if sep:
            try:
                mapping[int(rank_raw.strip())] = domain.strip()
            except ValueError:
                continue
    return mapping


def url_type(url: str) -> str:
    """Same buckets used to characterize the phishing feeds."""
    try:
        p = urlparse(url)
    except Exception:
        return "malformed"
    if p.scheme not in ("http", "https") or not p.netloc:
        return "malformed"
    segs = [s for s in p.path.split("/") if s]
    if not segs:
        return "root" if not p.query else "query"
    if p.query:
        return "query"
    return "path1" if len(segs) == 1 else "pathN"


def registrable(host_url: str) -> str:
    e = build_splits.EXTRACT(urlparse(host_url).hostname or "")
    return f"{e.domain}.{e.suffix}".lower() if e.suffix and e.domain else ""


def query_index(
    domain: str, index: str, timeout: int = 60
) -> tuple[int | None, list[dict[str, Any]], str]:
    """One CC index query. Returns (http_status, records, note)."""
    url = (
        f"{CC_BASE}/{index}-index?url={domain}"
        f"&output=json&filter=status:200&collapse=urlkey"
        f"&matchType=domain"
    )
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    except requests.RequestException as e:
        return None, [], f"request-error:{type(e).__name__}"
    if r.status_code != 200:
        return r.status_code, [], f"http-{r.status_code}"
    records = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        u = rec.get("url", "")
        if not u.startswith(("http://", "https://")):
            continue
        if u.endswith("/robots.txt"):
            continue
        records.append(rec)
    if not records:
        return r.status_code, [], "no-usable-captures"
    return r.status_code, records, "ok"


def fetch_domain(domain: str, retries: int = 2, sleep: float = 1.0) -> dict[str, Any]:
    """Query primary index (with retries), then fallback, else give up."""
    attempts = 0
    for attempt in range(retries):
        attempts += 1
        status, records, note = query_index(domain, CC_INDEX_PRIMARY)
        if note == "ok":
            return _result(domain, CC_INDEX_PRIMARY, status, records, note, attempts)
        if status == 404 or note == "no-usable-captures":
            break
        time.sleep(sleep * (attempt + 1))
    status, records, note = query_index(domain, CC_INDEX_FALLBACK)
    attempts += 1
    index = CC_INDEX_FALLBACK if note == "ok" else None
    if note != "ok":
        note = f"unproductive(primary:{note})"
    time.sleep(0.5)  # politeness gap between index queries
    return _result(domain, index, status, records, note, attempts)


def _result(
    domain: str,
    index: str | None,
    status: int | None,
    records: list[dict[str, Any]],
    note: str,
    attempts: int,
    mechanism: str = "cdx",
    query: str | None = None,
    scheme_evidence: dict[str, int] | None = None,
    n_evidence_rows: int | None = None,
    rows_sampled: bool | None = None,
) -> dict[str, Any]:
    slim = [
        {
            "url": r.get("url"),
            "timestamp": r.get("timestamp"),
            "digest": r.get("digest"),
            "mime": r.get("mime"),
            "status": r.get("status"),
        }
        for r in records
    ]
    return {
        "domain": domain,
        "index": index,
        "mechanism": mechanism,
        "query": query if query is not None else QUERY_FORM,
        "scheme_evidence": scheme_evidence,
        "n_evidence_rows": n_evidence_rows,
        "rows_sampled": rows_sampled,
        "http_status": status,
        "n_records": len(slim),
        "note": note,
        "attempts": attempts,
        "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": slim,
    }


def cc_time_to_iso(ts: str) -> str:
    return (
        datetime.strptime(ts, "%Y%m%d%H%M%S")
        .replace(tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
    )


def render_columnar_sql(table: str, crawl: str, domain: str) -> str:
    """One-domain Athena query over the pinned crawl partition."""
    safe = domain.replace("'", "''")
    return COLUMNAR_SQL_TEMPLATE.format(table=table, crawl=crawl, domain=safe)


def athena_time_to_cc(ts: str) -> str | None:
    """Athena TIMESTAMP ('YYYY-MM-DD HH:MM:SS[.fff]') -> CC 'YYYYmmddHHMMSS'."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts.strip(), fmt).strftime("%Y%m%d%H%M%S")
        except (ValueError, AttributeError):
            continue
    return None


def scheme_counts(rows: list[dict[str, str | None]]) -> dict[str, int]:
    """Per-scheme counts over raw index rows (any fetch status).

    Evidence input to the root backfill: a 301 over http means the domain
    serves http even though no http URL is selectable. Counted over the
    unfiltered query rows, never over the 200-only selection.
    """
    out: dict[str, int] = {}
    for row in rows:
        scheme = urlparse(row.get("url") or "").scheme
        if scheme in ("http", "https"):
            out[scheme] = out.get(scheme, 0) + 1
    return out


def sample_rows(
    rows: list[dict[str, str | None]], cap: int, seed: int
) -> tuple[list[dict[str, str | None]], bool]:
    """Deterministic uniform subsample to cap rows. Returns (rows, sampled).

    Sort-by-URL first so the result depends only on content + seed, never
    on Athena's unguaranteed page order; then a seeded permutation prefix.
    Under the cap the input is returned untouched.
    """
    if len(rows) <= cap:
        return rows, False
    ordered = sorted(rows, key=lambda r: r.get("url") or "")
    rng = np.random.default_rng(seed)
    take = sorted(int(i) for i in rng.permutation(len(ordered))[:cap])
    return [ordered[i] for i in take], True


def columnar_records(rows: list[dict[str, str | None]]) -> list[dict[str, Any]]:
    """Athena result rows -> slim records in the select-stage schema.

    Same {url, timestamp, digest, mime, status} shape as the CDX path, so
    selection is mechanism-blind. All fetch statuses are retained with the
    status preserved per row: URL selection keeps 200s only, while scheme
    evidence is counted over apex-host rows of any status (a 301 over http
    means the domain serves http). Mirrors the CDX filters (http/https
    only, no robots.txt) and collapses repeat captures of one URL to the
    earliest fetch (the CDX collapse=urlkey equivalent).
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = row.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        if url.endswith("/robots.txt"):
            continue
        ts = athena_time_to_cc(row.get("fetch_time") or "")
        if ts is None:
            continue
        prev = best.get(url)
        if prev is not None and str(prev["timestamp"]) <= ts:
            continue
        best[url] = {
            "url": url,
            "timestamp": ts,
            "digest": row.get("content_digest"),
            "mime": row.get("content_mime_type"),
            "status": str(row.get("fetch_status") or ""),
        }
    return list(best.values())


def _boto3_athena(region: str | None) -> Any:
    """Lazily build an Athena client (boto3 stays a fetch-only import)."""
    try:
        boto3_mod: Any = importlib.import_module("boto3")
    except ImportError as e:
        raise SystemExit(
            "columnar fetch needs boto3 plus AWS credentials "
            "(pip install boto3; configure a profile/role with Athena + S3 "
            "read on s3://commoncrawl). No live call is made by --phase select."
        ) from e
    if region:
        return boto3_mod.client("athena", region_name=region)
    return boto3_mod.client("athena")


def athena_query_rows(
    client: Any,
    sql: str,
    database: str,
    output: str,
    poll_interval: float = 2.0,
    max_wait: float = 600.0,
    stats_out: dict[str, Any] | None = None,
) -> list[dict[str, str | None]]:
    """Run one Athena query and return rows as name->value dicts (paged).

    If stats_out is given, it is filled with DataScannedInBytes,
    EngineExecutionTimeInMillis and TotalExecutionTimeInMillis from the
    succeeded execution (pilot/cost accounting; no effect otherwise).
    """
    qid = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output},
    )["QueryExecutionId"]
    waited = 0.0
    while True:
        execution = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = execution["Status"]
        if state["State"] == "SUCCEEDED":
            if stats_out is not None:
                qstats = execution.get("Statistics", {})
                stats_out.update(
                    {
                        "data_scanned_bytes": qstats.get("DataScannedInBytes", 0),
                        "engine_ms": qstats.get("EngineExecutionTimeInMillis", 0),
                        "total_ms": qstats.get("TotalExecutionTimeInMillis", 0),
                    }
                )
            break
        if state["State"] in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"athena {state['State']}: {state.get('StateChangeReason')}"
            )
        time.sleep(poll_interval)
        waited += poll_interval
        if waited >= max_wait:
            try:
                client.stop_query_execution(QueryExecutionId=qid)
            finally:
                raise TimeoutError(
                    f"athena query still {state['State']} after {max_wait}s"
                )
    out: list[dict[str, str | None]] = []
    token: str | None = None
    first_page = True
    while True:
        kwargs: dict[str, Any] = {"QueryExecutionId": qid}
        if token is not None:
            kwargs["NextToken"] = token
        resp = client.get_query_results(**kwargs)
        page = resp["ResultSet"]
        if first_page:
            columns = [c["Name"] for c in page["ResultSetMetadata"]["ColumnInfo"]]
            first_page = False
        data_rows = page["Rows"]
        if token is None:
            data_rows = data_rows[1:]  # first row of first page is the header
        for r in data_rows:
            cells = r.get("Data", [])
            out.append(
                {
                    name: (cell.get("VarCharValue") if cell else None)
                    for name, cell in zip(columns, cells, strict=False)
                }
            )
        token = resp.get("NextToken", None)
        if token is None:
            break
    return out


def fetch_domain_columnar(
    domain: str,
    ctx: dict[str, Any],
    retries: int = 2,
    sleep: float = 2.0,
    query_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Query the pinned crawl partition (with retries), then fallback, else give up.

    Each crawl is a single unfiltered query: scheme evidence is counted
    over all returned rows while URL selection keeps 200s only, so the
    evidence never inherits the selection filter (see COLUMNAR_SQL_TEMPLATE).
    If query_stats is given, one entry per executed crawl query (crawl,
    data_scanned_bytes, engine_ms, total_ms) is appended for accounting.
    """

    def _crawl(
        crawl: str,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, int], int, bool]:
        """(selectable records or None on transport failure, scheme evidence
        over all rows, raw row count, whether the row cap sampled)."""
        try:
            stats: dict[str, Any] = {}
            rows = athena_query_rows(
                ctx["client"],
                render_columnar_sql(ctx["table"], crawl, domain),
                ctx["database"],
                ctx["output"],
                stats_out=stats,
            )
            if query_stats is not None:
                query_stats.append({"crawl": crawl, **stats})
        except Exception:
            return None, {}, 0, False
        evidence = scheme_counts(rows)
        n_rows = len(rows)
        rows, sampled = sample_rows(
            rows,
            int(ctx.get("row_cap") or ROW_CAP_PER_DOMAIN_DEFAULT),
            int(ctx.get("sample_seed") or 0),
        )
        return columnar_records(rows), evidence, n_rows, sampled

    attempts = 0
    for attempt in range(retries):
        attempts += 1
        recs, evidence, n_rows, sampled = _crawl(CC_INDEX_PRIMARY)
        if recs is None:
            time.sleep(sleep * (attempt + 1))
            continue
        if recs:
            return _result(
                domain,
                CC_INDEX_PRIMARY,
                200,
                recs,
                "ok",
                attempts,
                mechanism="columnar",
                query=render_columnar_sql(ctx["table"], CC_INDEX_PRIMARY, domain),
                scheme_evidence=evidence,
                n_evidence_rows=n_rows,
                rows_sampled=sampled,
            )
        break  # definitive miss on primary -> fallback (mirrors the CDX 404 path)
    attempts += 1
    recs, evidence, n_rows, sampled = _crawl(CC_INDEX_FALLBACK)
    if recs:
        return _result(
            domain,
            CC_INDEX_FALLBACK,
            200,
            recs,
            "ok",
            attempts,
            mechanism="columnar",
            query=render_columnar_sql(ctx["table"], CC_INDEX_FALLBACK, domain),
            scheme_evidence=evidence,
            n_evidence_rows=n_rows,
            rows_sampled=sampled,
        )
    if recs is None:
        return _result(
            domain,
            None,
            None,
            [],
            "unproductive(columnar:query-failed)",
            attempts,
            mechanism="columnar",
            scheme_evidence=evidence,
            n_evidence_rows=n_rows,
            rows_sampled=sampled,
        )
    return _result(
        domain,
        None,
        200,
        [],
        "no-usable-captures",
        attempts,
        mechanism="columnar",
        scheme_evidence=evidence,
        n_evidence_rows=n_rows,
        rows_sampled=sampled,
    )


def cmd_fetch(a: argparse.Namespace) -> int:
    cache_path = Path(a.cache)
    mapping = load_tranco()
    rng = np.random.default_rng(a.seed)
    if a.sample_seed is None:
        a.sample_seed = a.seed + 2  # distinct stream from fetch/select
    if cache_path.exists() and not a.refetch:
        # Resume: keep definitive outcomes (successes + hard misses), drop
        # stale transient failures for retry, then fill strata still short
        # of quota. Seeded candidate order is reproduced identically, so
        # resumed domains match a fresh run's order.
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("seed") != a.seed:
            sys.exit(
                f"cache seed {cache.get('seed')} != --seed {a.seed} "
                "(pass --refetch for a fresh seed, or reuse the cached seed)"
            )
        if cache.get("mechanism", "cdx") != a.source:
            sys.exit(
                f"cache mechanism {cache.get('mechanism', 'cdx')!r} != "
                f"--source {a.source!r} (pass --refetch, or reuse the "
                "matching --source)"
            )
        print(
            f"resuming from existing cache {cache_path} "
            f"({len(cache['domains'])} entries)"
        )
    else:
        cache = {
            "seed": a.seed,
            "mechanism": a.source,
            "cc_index_primary": CC_INDEX_PRIMARY,
            "cc_index_fallback": CC_INDEX_FALLBACK,
            "tranco_csv": str(TRANC0_CSV),
            "tranco_sha256": TRANC0_SHA256,
            "domains": [],
        }
        if a.source == "columnar":
            cache["columnar_table_s3"] = CC_TABLE_S3
            cache["columnar_athena_table"] = a.athena_table
            cache["columnar_athena_database"] = a.athena_database
            cache["columnar_sql_template"] = COLUMNAR_SQL_TEMPLATE
            cache["row_cap_per_domain"] = a.row_cap
            cache["sample_seed"] = a.sample_seed

    def save_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)

    if a.source == "columnar":
        if not a.athena_output:
            sys.exit("--source columnar needs --athena-output (s3:// results bucket)")
        ctx: dict[str, Any] = {
            "client": _boto3_athena(a.athena_region),
            "table": a.athena_table,
            "database": a.athena_database,
            "output": a.athena_output,
            "row_cap": a.row_cap,
            "sample_seed": a.sample_seed,
        }
        fetch_fn: Callable[[str], dict[str, Any]] = functools.partial(
            fetch_domain_columnar, ctx=ctx
        )
    else:
        print(
            "CDX source: probe-only — index.commoncrawl.org throttles bulk "
            "fetching. Use --source columnar for the full run."
        )
        fetch_fn = fetch_domain

    total_calls = 0
    for sname, (lo, hi) in STRATA.items():
        pool = [mapping[r] for r in range(lo, hi + 1)]
        order = rng.permutation(len(pool))
        # Tiny head strata hold fewer domains than the quota; enumerate them.
        wanted = min(a.productive_per_stratum, len(pool))
        got = sum(
            1
            for e in cache["domains"]
            if e.get("stratum") == sname and e.get("index") is not None
        )
        examined = 0
        batch: list[dict[str, Any]] = []
        done_ranks = {
            e["domain"]
            for e in cache["domains"]
            if e.get("stratum") == sname and _definitive(e)
        }
        # Drop stale transient-failure entries; they will be re-fetched.
        cache["domains"] = [
            e
            for e in cache["domains"]
            if not (e.get("stratum") == sname and not _definitive(e))
        ]
        # Examine candidates in seeded order until the stratum quota of
        # productive domains is met (or candidates run out).
        cand_idx = 0
        while cand_idx < len(pool) and pool[int(order[cand_idx])] in done_ranks:
            cand_idx += 1
        pending: dict[Future[dict[str, Any]], tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            while got < wanted and cand_idx < min(CANDIDATES_PER_STRATUM, len(pool)):
                while (
                    len(pending) < a.workers * 2
                    and got + len(pending) < wanted + 50
                    and cand_idx < min(CANDIDATES_PER_STRATUM, len(pool))
                ):
                    d = pool[int(order[cand_idx])]
                    cand_idx += 1
                    while (
                        cand_idx < len(pool)
                        and pool[int(order[cand_idx])] in done_ranks
                    ):
                        cand_idx += 1
                    if d in done_ranks:
                        continue
                    pending[ex.submit(fetch_fn, d)] = (d, sname)
                    examined += 1
                if not pending:
                    break
                for fut in as_completed(list(pending)):
                    d, sn = pending.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:  # never lose the cache on error
                        res = _result(d, None, None, [], f"error:{e}", 0)
                    res["stratum"] = sn
                    batch.append(res)
                    cache["domains"].append(res)
                    total_calls += res.get("attempts", 0)
                    if res["index"] is not None:
                        got += 1
                    if (len(batch) % 25 == 0) or got >= wanted:
                        print(
                            f"{sname}: productive={got}/{wanted} "
                            f"examined={examined} api_calls~{total_calls}",
                            flush=True,
                        )
                    if got >= wanted:
                        break
                # cancel stragglers once quota met
                if got >= wanted:
                    for fut in pending:
                        fut.cancel()
                    pending = {}
                    break
        # Assign ranks (linear scan once per stratum for determinism).
        rank_of = {dd: r for r, dd in mapping.items() if lo <= r <= hi}
        for res in batch:
            res["rank"] = rank_of.get(res["domain"])
        save_cache()  # incremental: an abort never loses fetched strata
        print(
            f"{sname}: done productive={got}/{wanted} examined={examined} "
            f"api_calls~{total_calls} (cache saved)",
            flush=True,
        )
    print(f"cache complete: {cache_path} ({len(cache['domains'])} entries)")
    return 0


def _shuffle(rng: np.random.Generator, rows: list[Any]) -> None:
    """Seeded in-place shuffle (numpy stubs reject list[dict] directly)."""
    rng.shuffle(rows)


def dedup_key(url: str) -> str:
    """Canonical identity for quota counting: scheme + apex host + path + query.

    Collapses the www./apex variance (four netloc_len characters that vary
    arbitrarily by domain) and root trailing slashes, so a heavily
    re-crawled domain cannot eat its stratum's quota with near-identical
    captures. Deliberately preserves scheme (collapsing http/https would
    game the scheme gap gate by construction) and query strings (they
    distinguish landing URLs), and preserves path case (paths may be
    case-sensitive server-side).
    """
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{p.port}" if p.port else ""
    segs = [s for s in p.path.split("/") if s]
    path = "/" + "/".join(segs) if segs else ""
    key = f"{p.scheme}://{host}{port}{path}"
    if p.query:
        key += f"?{p.query}"
    return key


def _collapse_digest(
    per_type: dict[str, list[dict[str, Any]]],
    empirical_roots: set[tuple[str, str]],
    stats: dict[str, Any],
) -> None:
    """Opt-in: drop records sharing a content_digest (earliest kept).

    Same bytes under different URLs (tracking params, session IDs,
    boilerplate error pages) consume quota without adding diversity.
    Runs after canonical dedup, per URL type, earliest timestamp wins.
    Root removals also leave empirical_roots so backfill stays consistent.
    """
    for t, rows in per_type.items():
        seen: dict[str, dict[str, Any]] = {}
        kept: list[dict[str, Any]] = []
        for row in rows:
            digest = row.get("cc_digest")
            if not digest:
                kept.append(row)
                continue
            prev = seen.get(str(digest))
            if prev is None:
                seen[str(digest)] = row
                kept.append(row)
                continue
            stats["digest_duplicates"] += 1
            if row["cc_timestamp"] < prev["cc_timestamp"]:
                kept[kept.index(prev)] = row
                seen[str(digest)] = row
                if t == "root":
                    old_key = (prev["etld1"], urlparse(prev["url"]).scheme)
                    if not any(
                        r["etld1"] == old_key[0]
                        and urlparse(r["url"]).scheme == old_key[1]
                        for r in kept
                    ):
                        empirical_roots.discard(old_key)
                    empirical_roots.add((row["etld1"], urlparse(row["url"]).scheme))
        per_type[t] = kept


def cmd_select(a: argparse.Namespace) -> int:
    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed + 1)  # distinct stream from fetching
    quotas = {t: int(round(a.target_n * p)) for t, p in TYPE_TARGETS.items()}
    # Fix rounding drift on the largest bucket.
    quotas["root"] += a.target_n - sum(quotas.values())

    # Gather normalized, deduplicated candidates per type. Deduplication
    # happens here, before quota counting: one record per canonical key
    # (earliest capture wins), so re-crawls and www/apex variants never
    # consume quota slots.
    per_type: dict[str, list[dict[str, Any]]] = {t: [] for t in quotas}
    seen_keys: set[str] = set()
    by_key: dict[str, dict[str, Any]] = {}
    # Apex-keyed empirical roots: (registrable host, scheme). A www root
    # and an apex root are the same host for backfill purposes (see the
    # apex rule below), so either one blocks synthesis of that apex root.
    empirical_roots: set[tuple[str, str]] = set()
    stats: dict[str, Any] = {
        "records_total": 0,
        "unselectable_status": 0,
        "unnormalisable": 0,
        "duplicates": 0,
        "digest_duplicates": 0,
        "per_domain_kept": {},
    }
    dom_kept: dict[str, int] = {}
    etld1_kept: dict[str, int] = {}
    for entry in cache["domains"]:
        if entry.get("index") is None:
            continue
        for rec in entry["records"]:
            stats["records_total"] += 1
            if str(rec.get("status") or "") != "200":
                # Retained for scheme evidence only; redirects and error
                # pages must never enter the dataset.
                stats["unselectable_status"] += 1
                continue
            norm = build_splits.normalise(rec["url"])
            if norm is None:
                stats["unnormalisable"] += 1
                continue
            t = url_type(norm)
            if t == "malformed":
                stats["unnormalisable"] += 1
                continue
            try:
                first_seen = cc_time_to_iso(rec["timestamp"])
            except (ValueError, TypeError, KeyError):
                stats["unnormalisable"] += 1
                continue
            key = dedup_key(norm)
            prev = by_key.get(key)
            if prev is not None:
                stats["duplicates"] += 1
                if rec["timestamp"] < prev["cc_timestamp"]:
                    prev.update(
                        {
                            "url": norm,
                            "first_seen": first_seen,
                            "cc_index": entry["index"],
                            "cc_index_url": rec["url"],
                            "cc_timestamp": rec["timestamp"],
                            "cc_digest": rec.get("digest"),
                            "seed_domain": entry["domain"],
                            "stratum": entry["stratum"],
                        }
                    )
                continue
            etld1 = registrable(norm)
            row = {
                "url": norm,
                "url_type": t,
                "first_seen": first_seen,
                "cc_index": entry["index"],
                "cc_index_url": rec["url"],
                "cc_timestamp": rec["timestamp"],
                "cc_digest": rec.get("digest"),
                "seed_domain": entry["domain"],
                "stratum": entry["stratum"],
                "etld1": etld1,
            }
            by_key[key] = row
            seen_keys.add(key)
            per_type[t].append(row)
            if t == "root":
                empirical_roots.add((etld1, urlparse(norm).scheme))
    if a.collapse_digest:
        _collapse_digest(per_type, empirical_roots, stats)
    for rows in per_type.values():
        _shuffle(rng, rows)

    # Root backfill apex rule (committed): synthesised roots are apex
    # roots, and the deduplication key is apex-based. Rationale: the www.
    # prefix is four characters of netloc_len on the feature that carried
    # the original leak, varying arbitrarily by domain. So:
    # * synthesis builds <scheme>://<apex>/ only, one per scheme with
    #   evidence on the apex host itself (any fetch status — an http 301
    #   counts as http served). No www fallback: a domain with no apex
    #   capture at all is skipped for backfill (its 200 URLs still select
    #   normally) and the skip is counted;
    # * a synthesised apex root is a duplicate if ANY empirical root
    #   exists for that (apex host, scheme) — www or apex host alike.
    # Caches predating unfiltered retention (all CDX caches) fall back to
    # 200-only records with the bias this implies, and say so in
    # domains_evidence_fallback_200_only.
    synth_stats: dict[str, Any] = {
        "domains_productive": 0,
        "domains_non_apex_seed_skipped": 0,
        "domains_no_apex_capture": 0,
        "domains_no_scheme_evidence": 0,
        "domains_evidence_fallback_200_only": 0,
        "candidates": 0,
        "skipped_duplicate": 0,
        "added": 0,
    }
    ev_https: list[float] = []  # per-domain https share, apex evidence
    sel_https: list[float] = []  # per-domain https share, 200-only selection rows
    for entry in cache["domains"]:
        if entry.get("index") is None:
            continue
        synth_stats["domains_productive"] += 1
        seed_host = str(entry.get("domain", "")).lower()
        ext = build_splits.EXTRACT(seed_host)
        apex = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix and ext.domain else ""
        if not apex or apex != seed_host:
            synth_stats["domains_non_apex_seed_skipped"] += 1
            continue
        if entry.get("n_evidence_rows") is None:
            synth_stats["domains_evidence_fallback_200_only"] += 1
        schemes: dict[str, int] = {}
        has_apex_capture = False
        earliest: str | None = None
        for rec in entry["records"]:
            u = rec.get("url", "")
            if (urlparse(u).hostname or "").lower() != apex:
                continue
            has_apex_capture = True
            sc = urlparse(u).scheme
            if sc in ("http", "https"):
                schemes[sc] = schemes.get(sc, 0) + 1
            ts = rec.get("timestamp")
            if isinstance(ts, str) and (earliest is None or ts < earliest):
                try:
                    cc_time_to_iso(ts)
                except (ValueError, TypeError):
                    continue
                earliest = ts
        if not has_apex_capture:
            synth_stats["domains_no_apex_capture"] += 1
            continue
        if schemes:
            ev_https.append(schemes.get("https", 0) / sum(schemes.values()))
        sel_schemes: dict[str, int] = {}
        for rec in entry["records"]:
            if str(rec.get("status") or "") != "200":
                continue
            u = rec.get("url", "")
            if urlparse(u).scheme in ("http", "https"):
                sc = urlparse(u).scheme
                sel_schemes[sc] = sel_schemes.get(sc, 0) + 1
        if sel_schemes:
            sel_https.append(sel_schemes.get("https", 0) / sum(sel_schemes.values()))
        if not schemes or earliest is None:
            synth_stats["domains_no_scheme_evidence"] += 1
            continue
        for scheme in sorted(schemes):
            synth_stats["candidates"] += 1
            norm = build_splits.normalise(f"{scheme}://{apex}/")
            if norm is None or url_type(norm) != "root":
                continue
            cand_key = dedup_key(norm)
            if cand_key in seen_keys or (registrable(norm), scheme) in empirical_roots:
                synth_stats["skipped_duplicate"] += 1
                continue
            seen_keys.add(cand_key)
            per_type["root"].append(
                {
                    "url": norm,
                    "url_type": "root",
                    "first_seen": cc_time_to_iso(earliest),
                    "cc_index": entry["index"],
                    "cc_index_url": None,
                    "cc_timestamp": earliest,
                    "cc_digest": None,
                    "seed_domain": entry["domain"],
                    "stratum": entry["stratum"],
                    "etld1": registrable(norm),
                    "synthesized_root": True,
                    "scheme_evidence": dict(schemes),
                }
            )
            synth_stats["added"] += 1
    # Synthesised roots join the same shuffle + capped selection below, so
    # they compete under identical per-domain / per-eTLD+1 caps.
    _shuffle(rng, per_type["root"])

    # Pass 1: per-domain-type cap + per-domain total cap + per-eTLD+1 cap.
    # Pass 2 (only for shortfalls): relax the per-domain-type cap.
    selected: list[dict[str, Any]] = []
    for type_cap in (PER_DOMAIN_TYPE_CAP, 10**9):
        for t, rows in per_type.items():
            need = quotas[t] - sum(1 for s in selected if s["url_type"] == t)
            if need <= 0:
                continue
            per_dom_type: dict[tuple[str, str], int] = {}
            for row in rows:
                if row.get("_taken"):
                    continue
                dom_key = (str(row["seed_domain"]), t)
                if per_dom_type.get(dom_key, 0) >= type_cap:
                    continue
                if dom_kept.get(row["seed_domain"], 0) >= PER_DOMAIN_TOTAL_CAP:
                    continue
                if etld1_kept.get(row["etld1"], 0) >= PER_ETLD1_CAP:
                    continue
                row["_taken"] = True
                selected.append(row)
                per_dom_type[dom_key] = per_dom_type.get(dom_key, 0) + 1
                dom_kept[row["seed_domain"]] = dom_kept.get(row["seed_domain"], 0) + 1
                etld1_kept[row["etld1"]] = etld1_kept.get(row["etld1"], 0) + 1
                need -= 1
                if need <= 0:
                    break
        if all(
            sum(1 for s in selected if s["url_type"] == t) >= q
            for t, q in quotas.items()
        ):
            break

    rank_of_domain = {e["domain"]: e.get("rank") for e in cache["domains"]}
    rows_out: list[dict[str, Any]] = []
    for s in selected:
        s = {k: v for k, v in s.items() if not k.startswith("_")}
        synth = bool(s.pop("synthesized_root", False))
        scheme_ev = s.pop("scheme_evidence", None)
        out_row: dict[str, Any] = {
            "url": s["url"],
            "label": 0,
            "first_seen": s["first_seen"],
            "source": f"cc:{s['cc_index']}",
            "time_basis": ("domain-inferred-root" if synth else "commoncrawl-index"),
            "cc_index": s["cc_index"],
            "cc_index_url": s["cc_index_url"],
            "cc_timestamp": s["cc_timestamp"],
            "cc_digest": s["cc_digest"],
            "tranco_list_id": TRANC0_ID,
            "tranco_rank": rank_of_domain.get(s["seed_domain"]),
            "popularity_stratum": s["stratum"],
            "seed_domain": s["seed_domain"],
            "url_type": s["url_type"],
            "seed": a.seed,
        }
        if synth:
            out_row["synthesized_root"] = True
            out_row["scheme_evidence"] = scheme_ev
        rows_out.append(out_row)
    rows_out.sort(key=lambda r: (r["first_seen"], r["url"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows_out:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    from collections import Counter

    got_types = Counter(r["url_type"] for r in rows_out)
    got_strata = Counter(r["popularity_stratum"] for r in rows_out)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov = {
        "generator": "build_cc_benign.py",
        "generated_at": generated_at,
        "seed": a.seed,
        "target_n": a.target_n,
        "n_written": len(rows_out),
        "cc_index_primary": CC_INDEX_PRIMARY,
        "cc_index_fallback": CC_INDEX_FALLBACK,
        "fetch_mechanism": cache.get("mechanism", "cdx"),
        "root_synthesis": {
            "host_form_rule": "apex only: synthesised roots are "
            "<scheme>://<apex>/, and the dedup key is (apex host, scheme) "
            "so a www empirical root blocks apex synthesis. No www "
            "fallback: domains with no apex capture at all are skipped "
            "for backfill (their 200 URLs still select normally).",
            "rule": "apex root per scheme with evidence on the apex host "
            "itself (any fetch status; caches predating unfiltered "
            "retention fall back to 200-only records) — never a default "
            "scheme; empirical record wins ties via apex-key dedup; dated "
            "at the domain's earliest apex capture",
            "phishing_reference_is_https": {"overall": 0.9105, "roots": 0.8191},
            **synth_stats,
            "scheme_evidence_vs_selection": {
                "mean_https_share_evidence_apex_unfiltered": (
                    sum(ev_https) / len(ev_https) if ev_https else None
                ),
                "mean_https_share_selection_200_only": (
                    sum(sel_https) / len(sel_https) if sel_https else None
                ),
                "mean_delta_evidence_minus_selection": (
                    (sum(ev_https) / len(ev_https) - sum(sel_https) / len(sel_https))
                    if ev_https and sel_https
                    else None
                ),
                "n_domains_compared": min(len(ev_https), len(sel_https)),
            },
            "n_synthesized_roots_selected": sum(
                1 for r in rows_out if r.get("synthesized_root")
            ),
        },
        "tranco_csv": str(TRANC0_CSV),
        "tranco_sha256": TRANC0_SHA256,
        "tranco_id": TRANC0_ID,
        "cache_file": str(a.cache),
        "cache_sha256": sha256_file(Path(a.cache)),
        "strata": {k: list(v) for k, v in STRATA.items()},
        "productive_per_stratum_target": a.productive_per_stratum,
        "type_targets_measured_from_phishing": dict(TYPE_TARGETS),
        "type_quotas": quotas,
        "dedup": {
            "canonical_key": "scheme + apex-host (www stripped) + path "
            "(root slash stripped) + query; earliest capture wins",
            "digest_collapse": bool(a.collapse_digest),
        },
        "type_counts": {t: got_types.get(t, 0) for t in quotas},
        "stratum_counts": {k: got_strata.get(k, 0) for k in STRATA},
        "caps": {
            "per_domain_type": PER_DOMAIN_TYPE_CAP,
            "per_domain_total": PER_DOMAIN_TOTAL_CAP,
            "per_etld1": PER_ETLD1_CAP,
        },
        "candidate_stats": stats,
        "fetch_mechanism_detail": (
            {
                "columnar_table_s3": cache.get("columnar_table_s3"),
                "columnar_athena_table": cache.get("columnar_athena_table"),
                "columnar_athena_database": cache.get("columnar_athena_database"),
                "columnar_sql_template": cache.get("columnar_sql_template"),
            }
            if cache.get("mechanism") == "columnar"
            else {"cc_query_form": QUERY_FORM}
        ),
        "row_cap_per_domain": cache.get("row_cap_per_domain"),
        "sample_seed": cache.get("sample_seed"),
        "scheme_handling": {
            "post_hoc_scheme_filter": False,
            "rule": "no subsampling, reweighting, or filtering on scheme at "
            "selection time; the root backfill takes every observed scheme "
            "per domain. If a corpus lands outside SCHEME_RATE_GAP_MAX, fix "
            "the sampling design (strata, query, quotas) — not a post-hoc "
            "scheme filter, which would pass the gate while trading the "
            "distortion into depth or rank strata. Record any such design "
            "change here.",
        },
        "n_domains_contributing": len(dom_kept),
        "output": str(out),
        "output_sha256": sha256_file(out),
    }
    prov_path = out.with_suffix(out.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8", newline="\r\n")
    print(f"wrote {out} ({len(rows_out)} rows) + {prov_path}")
    print(f"type quotas={quotas} got={dict(got_types)}")
    print(f"strata got={dict(got_strata)} domains={len(dom_kept)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["fetch", "select", "all"], default="all")
    p.add_argument(
        "--source",
        choices=["columnar", "cdx"],
        default="columnar",
        help="acquisition mechanism (default: columnar Athena over S3; "
        "cdx is the throttled front-end, small probes only)",
    )
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--target-n", type=int, default=TARGET_N_DEFAULT)
    p.add_argument(
        "--productive-per-stratum", type=int, default=PRODUCTIVE_PER_STRATUM_DEFAULT
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="fetch concurrency (default: 5 columnar, 1 cdx)",
    )
    p.add_argument("--cache", default=None)
    p.add_argument(
        "--out",
        default=None,
        help="default: data/raw/benign-cc-<primary>-<today>.jsonl",
    )
    p.add_argument("--refetch", action="store_true")
    p.add_argument(
        "--collapse-digest",
        action="store_true",
        help="opt-in: collapse same-content_digest URLs (earliest kept), "
        "counted as digest_duplicates in provenance",
    )
    p.add_argument("--athena-table", default=ATHENA_TABLE_DEFAULT)
    p.add_argument("--athena-database", default=ATHENA_DATABASE_DEFAULT)
    p.add_argument(
        "--athena-output",
        default=None,
        help="s3:// bucket/prefix for Athena results (required to fetch)",
    )
    p.add_argument("--athena-region", default=None)
    p.add_argument("--row-cap", type=int, default=ROW_CAP_PER_DOMAIN_DEFAULT)
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="seed for the per-domain row subsample (default: --seed + 2)",
    )
    a = p.parse_args(argv)
    if a.workers is None:
        a.workers = (
            WORKERS_COLUMNAR_DEFAULT if a.source == "columnar" else WORKERS_CDX_DEFAULT
        )
    if a.cache is None:
        a.cache = (
            CACHE_DEFAULT_COLUMNAR if a.source == "columnar" else CACHE_DEFAULT_CDX
        )
    if a.out is None:
        today = datetime.now(timezone.utc).date().isoformat()
        a.out = f"data/raw/benign-cc-{CC_INDEX_PRIMARY}-{today}.jsonl"
    if Path(a.out).exists():
        sys.exit(f"refusing to overwrite existing {a.out}")
    if a.phase in ("fetch", "all") and cmd_fetch(a) != 0:
        return 1
    if a.phase in ("select", "all") and cmd_select(a) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
