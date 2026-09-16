"""Read-only COUNT probe for the Amendment D second-wave decision (M3).

Measures how many selectable new apex roots s4-s6 hold beyond the banked
main cache, without writing any fetch state. Dry by default: candidate
enumeration + SQL + byte estimate only, no boto3 import, no network.
Live COUNTs require ``--execute`` (explicit) and still never touch the
fetch cache — the cache is opened read-only to replay seeded candidate
order, and results go to ``--report`` (refused when it resolves to the
cache path itself).

What counts as selectable mirrors selection, conservatively: 200-status
apex-host root captures (exact root-URL equalities, no LIKE wildcards),
primary crawl first with fallback only on zero (primary-first-
fallback-on-miss), phishing-tenant domains skipped before counting, and
the per-domain root cap applied to the sum. Root synthesis is upside
(<=1/domain with scheme evidence) and deliberately uncounted.

Decision rule (D0.4/D1): projected selectable new apex roots >= ``--bar``
(5,900) justifies the second fetch wave; otherwise D2.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

APEX_ROOT_COUNT_TEMPLATE = (
    "SELECT COUNT(DISTINCT url) "
    "FROM {table} WHERE crawl = '{crawl}' AND subset = 'warc' "
    "AND url_host_registered_domain = '{domain}' "
    "AND url_host_name = '{domain}' "
    "AND fetch_status = 200 "
    "AND (url = 'http://{domain}/' OR url = 'https://{domain}/' "
    "OR url = 'http://{domain}' OR url = 'https://{domain}')"
)

# Upper-bound byte calibration: the 26-domain pilot measured ~52 MB mean
# per full domain SELECT; a COUNT ships one narrow column over the same
# predicate, so actuals land below. Measured bytes replace the estimate
# in every --execute report via stats_out.
BYTE_UPPER_MB_PER_QUERY = 52.0
COST_PER_TB_USD = 5.0
BAR_DEFAULT = 5900


def render_apex_root_count_sql(table: str, crawl: str, domain: str) -> str:
    """COUNT(DISTINCT url) of 200-status apex root captures for one domain."""
    safe = domain.replace("'", "''")
    return APEX_ROOT_COUNT_TEMPLATE.format(table=table, crawl=crawl, domain=safe)


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
    selectable rows under the registered exclusion, so it is skipped
    before any COUNT — same exclusion, applied pre-query instead of
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


def parse_single_count(rows: list[dict[str, str | None]]) -> int:
    """First data row of the one-column COUNT query -> row count."""
    if not rows:
        raise ValueError("empty count result")
    return int(str(list(rows[0].values())[0]))


def cap_sum(counts: list[int], cap: int) -> int:
    """Selectable projection: per-domain root cap applied to the sum."""
    return sum(min(c, cap) for c in counts)


def project_yield(
    counted_capped: int, n_counted: int, n_total: int, bar: int
) -> dict[str, Any]:
    """Linear projection from a counted prefix onto the full set.

    Returns the measured sum, the projected total at the measured rate,
    and the hit rate the uncounted remainder would need for the bar —
    so a capped run still answers whether D1 is reachable.
    """
    rate = counted_capped / n_counted if n_counted else 0.0
    projected = counted_capped + rate * (n_total - n_counted)
    rest = n_total - n_counted
    need = (bar - counted_capped) / rest if rest else 0.0
    return {
        "counted_capped": counted_capped,
        "n_counted": n_counted,
        "n_total": n_total,
        "rate_per_domain": rate,
        "projected_total": projected,
        "required_rate_on_remainder": need,
    }


def decide(projected_total: float, bar: int) -> str:
    """D1 iff the projection reaches the bar, else D2."""
    return "D1" if projected_total >= bar else "D2"


def estimate_bytes(n_queries: int) -> dict[str, float]:
    """Upper-bound scan/cost estimate for n COUNT queries (dry-run only)."""
    mb = n_queries * BYTE_UPPER_MB_PER_QUERY
    return {
        "n_queries": float(n_queries),
        "bytes_upper_mb": mb,
        "cost_upper_usd": mb / 1_048_576 * COST_PER_TB_USD,
    }


def count_domain(
    client: Any,
    database: str,
    output: str,
    table: str,
    domain: str,
    primary: str,
    fallback: str,
) -> tuple[int, int, bool]:
    """COUNT(DISTINCT url) apex roots: primary, fallback only on zero.

    Returns (count, data_scanned_bytes, fallback_used).
    """
    stats: dict[str, Any] = {}
    rows = B.athena_query_rows(
        client,
        render_apex_root_count_sql(table, primary, domain),
        database,
        output,
        stats_out=stats,
    )
    n = parse_single_count(rows)
    scanned = int(stats.get("DataScannedInBytes", 0))
    used_fallback = False
    if n == 0:
        stats2: dict[str, Any] = {}
        rows2 = B.athena_query_rows(
            client,
            render_apex_root_count_sql(table, fallback, domain),
            database,
            output,
            stats_out=stats2,
        )
        n = parse_single_count(rows2)
        scanned += int(stats2.get("DataScannedInBytes", 0))
        used_fallback = True
    return n, scanned, used_fallback


def build_parser() -> argparse.ArgumentParser:
    """CLI: dry by default; --execute performs the live COUNTs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/raw/cc-columnar-CC-MAIN-2026-34.json")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--table", default="ccindex")
    p.add_argument("--database", default="ccindex")
    p.add_argument(
        "--output",
        default=None,
        help="s3:// Athena results bucket (required with --execute)",
    )
    p.add_argument("--primary", default=B.CC_INDEX_PRIMARY)
    p.add_argument("--fallback", default=B.CC_INDEX_FALLBACK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strata", default="s4_1k_10k,s5_10k_100k,s6_100k_1M")
    p.add_argument("--bar", type=int, default=BAR_DEFAULT)
    p.add_argument("--root-cap", type=int, default=B.PER_DOMAIN_TYPE_CAP)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument(
        "--max-counts",
        type=int,
        default=0,
        help="0 = count every kept candidate; else a seeded-order prefix",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="run the live COUNT queries (needs boto3 + AWS credentials)",
    )
    p.add_argument("--report", required=True, help="probe report JSON output path")
    return p


def main(argv: list[str] | None = None) -> int:
    """Enumerate (always) then COUNT (only with --execute); write --report."""
    a = build_parser().parse_args(argv)
    cache_path = Path(a.cache)
    report_path = Path(a.report)
    if report_path.resolve() == cache_path.resolve():
        sys.exit("refusing: --report must not be the fetch cache itself")
    if a.execute and not a.output:
        sys.exit("--execute needs --output (s3:// Athena results bucket)")

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

    to_count = kept if not a.max_counts else kept[: a.max_counts]
    # Fallback-on-zero doubles at most: every kept domain costs 1-2 queries.
    n_queries_max = 2 * len(to_count)
    est = estimate_bytes(n_queries_max)

    print(f"fresh s4-s6 candidates: {len(fresh)} {per_stratum}")
    print(f"tenant-skipped (no COUNT): {len(skipped)}")
    print(f"to COUNT: {len(to_count)} (max queries {n_queries_max})")
    print(
        f"byte upper bound: {est['bytes_upper_mb']:.0f} MB "
        f"(~${est['cost_upper_usd']:.2f} at $5/TB)"
    )
    print(f"SQL template: {APEX_ROOT_COUNT_TEMPLATE}")
    if to_count:
        ex_sql = render_apex_root_count_sql(a.table, a.primary, to_count[0][2])
        print(f"example: {ex_sql}")

    report: dict[str, Any] = {
        "probe": "probe_cc_roots (Amendment D M3)",
        "bar": a.bar,
        "root_cap": a.root_cap,
        "inputs": {
            "cache": str(cache_path),
            "cache_sha256": B.sha256_file(cache_path),
            "cache_seed": cache.get("seed"),
            "seed": a.seed,
            "strata": strata,
            "table": a.table,
            "database": a.database,
            "crawls": {"primary": a.primary, "fallback": a.fallback},
            "sql_template": APEX_ROOT_COUNT_TEMPLATE,
            "tenant_inputs": tenant_inputs,
        },
        "enumeration": {
            "n_fresh": len(fresh),
            "per_stratum": per_stratum,
            "n_tenant_skipped": len(skipped),
            "n_to_count": len(to_count),
        },
        "byte_estimate_upper": est,
        "executed": False,
    }

    if not a.execute:
        print(
            "dry run: no queries issued (pass --execute to COUNT). "
            f"Rule: projected selectable >= {a.bar} -> D1, else D2."
        )
        report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\r\n"
        )
        print(f"wrote {report_path}")
        return 0

    client = B._boto3_athena(None)
    counts: list[int] = []
    scanned_total = 0
    n_fallback = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        pending = {
            ex.submit(
                count_domain,
                client,
                a.database,
                a.output,
                a.table,
                dom,
                a.primary,
                a.fallback,
            ): (sname, dom)
            for sname, _, dom in to_count
        }
        for fut in as_completed(pending):
            sname, dom = pending[fut]
            try:
                n, scanned, used_fb = fut.result()
            except Exception as e:  # never lose counted progress on error
                print(
                    f"{sname}: {dom} query failed ({e}) — counted as 0", file=sys.stderr
                )
                n, scanned, used_fb = 0, 0, False
            counts.append(n)
            scanned_total += scanned
            n_fallback += 1 if used_fb else 0
            print(
                f"{sname}: {dom} apex_roots={n} "
                f"(counted={len(counts)}/{len(to_count)})",
                flush=True,
            )

    measured = cap_sum(counts, a.root_cap)
    proj = project_yield(measured, len(counts), len(kept), a.bar)
    verdict = decide(proj["projected_total"], a.bar)
    report.update(
        {
            "executed": True,
            "counts_raw_sum": sum(counts),
            "counts_capped_sum": measured,
            "data_scanned_bytes": scanned_total,
            "n_fallback_queries": n_fallback,
            "projection": proj,
            "decision": verdict,
        }
    )
    report_path.write_text(
        __import__("json").dumps(report, indent=2), encoding="utf-8", newline="\r\n"
    )
    print(
        f"measured selectable={measured} projected={proj['projected_total']:.0f} "
        f"bar={a.bar} -> {verdict}"
    )
    print(f"scanned {scanned_total} bytes; wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
