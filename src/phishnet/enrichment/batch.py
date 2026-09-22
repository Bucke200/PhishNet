"""Batch enrichment runner: cache keys -> sealed snapshot run.

For each key: hosted tenants are marked na WITHOUT querying (the
platform's age/certs are not the tenant's — see key.py); every other key
runs the RDAP chain then the CT chain, each under its own hard timeout.
A single key's failure is recorded on its row, never raised: one bad
domain must not abort a 40k-key run.

Run promptly after each feed snapshot — every day of delay widens the
phishing unknown-rate gap as registrations lapse (the contamination gate
catches the damage, timing prevents it). Seal the run before hashing or
joining it; an open file can never be pinned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from phishnet.enrichment import ct, rdap
from phishnet.enrichment.key import cache_key, is_hosted_tenant
from phishnet.enrichment.store import (
    append_records,
    run_keys,
    seal_run,
    write_run_meta,
)

SIGNALS: tuple[str, ...] = ("age", "ct")


def enrich_key(
    key: str,
    bootstrap: dict[str, list[str]] | None,
    *,
    ct_timeout: int = ct.CRT_TIMEOUT_S,
    rdap_timeout: int = rdap.RDAP_TIMEOUT_S,
    query_hosted: bool = False,
    ct_fetch: Callable[..., dict[str, Any]] | None = None,
    age_fetch: Callable[..., dict[str, Any]] | None = None,
    signals: tuple[str, ...] = SIGNALS,
) -> dict[str, Any]:
    """Raw snapshot row for one cache key (fetchers injectable for tests).

    ``signals`` selects which providers run; a skipped signal stores
    ``None`` (derives to unknown at join time, never na). Hosted tenants
    always resolve na without querying, whatever signals are requested.
    """
    if is_hosted_tenant(key) and not query_hosted:
        return {"cache_key": key, "hosted": True, "rdap": None, "ct": None}
    age = (
        (age_fetch or rdap.fetch_age)(key, bootstrap, rdap_timeout)
        if "age" in signals
        else None
    )
    history = (ct_fetch or ct.fetch_ct)(key, ct_timeout) if "ct" in signals else None
    return {"cache_key": key, "hosted": False, "rdap": age, "ct": history}


def _crash_row(key: str, error: Exception) -> dict[str, Any]:
    """Fail-closed record for a key whose fetchers raised unexpectedly.

    The batch contract is that one bad key never aborts the run; an
    unexpected exception therefore becomes an unknown (both lookups
    unresolved), never a missing row and never a raised error.
    """
    return {
        "cache_key": key,
        "hosted": False,
        "rdap": {
            "creation_date": None,
            "source": None,
            "server": None,
            "error": f"enrich-crashed: {type(error).__name__}: {error}",
        },
        "ct": {
            "certs": None,
            "provider": None,
            "truncated": False,
            "error": "not-attempted",
        },
    }


def enrich_keys(
    urls: list[str],
    snapshot: Path,
    run_id: str,
    *,
    bootstrap: dict[str, list[str]] | None = None,
    skip_hosted_queries: bool = True,
    ct_timeout: int = ct.CRT_TIMEOUT_S,
    rdap_timeout: int = rdap.RDAP_TIMEOUT_S,
    progress_every: int = 500,
    workers: int = 1,
    checkpoint_every: int = 0,
    signals: tuple[str, ...] = SIGNALS,
    shuffle_seed: int = 0,
    population_manifest_sha: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Enrich every distinct cache key behind urls into one snapshot run.

    Keys already stored under ``run_id`` are skipped (resume is a no-op
    re-run). Returns the seal sidecar. Prerequisites (bootstrap fetch,
    na-gate thresholds) are the caller's; this function only records what
    it used in the run meta.

    ``workers`` > 1 fetches keys on a thread pool (network-bound latency
    dominates). Fetches run in seeded-shuffle order (``shuffle_seed``) so
    availability windows stay uncorrelated with class; ``seal_run`` sorts
    by key before hashing, so the sealed hash is reproducible regardless
    of fetch order. ``checkpoint_every`` > 0 appends completed rows every
    N keys, so an interrupted 40k-key run resumes from its checkpoint
    instead of from zero (``append_records`` dedups by key).
    ``signals`` selects providers (a skipped signal stores ``None``);
    ``population_manifest_sha`` pins which population this run enriches.
    """
    keys: set[str] = set()
    for u in urls:
        key, _hosted = cache_key(u)
        if key:
            keys.add(key)
    stored = run_keys(snapshot, run_id)
    ordered = sorted(keys - stored)
    import numpy as np

    # Index back into `ordered` so fetch keys stay plain `str`
    # (numpy scalars would poison the JSONL snapshot).
    rng = np.random.default_rng(shuffle_seed)
    fetch_order = [ordered[i] for i in rng.permutation(len(ordered))]
    skipped = len(keys) - len(ordered)
    if skipped:
        print(
            f"  resume: {skipped} keys already stored, fetching {len(ordered)}",
            flush=True,
        )

    def _one(key: str) -> dict[str, Any]:
        try:
            return enrich_key(
                key,
                bootstrap,
                ct_timeout=ct_timeout,
                rdap_timeout=rdap_timeout,
                query_hosted=not skip_hosted_queries,
                signals=signals,
            )
        except Exception as e:  # fail-closed per key, never abort the run
            return _crash_row(key, e)

    def _stream() -> Iterator[dict[str, Any]]:
        if workers < 2:
            for key in fetch_order:
                yield _one(key)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                yield from ex.map(_one, fetch_order)

    rows: list[dict[str, Any]] = []
    for done, row in enumerate(_stream(), 1):
        rows.append(row)
        if progress_every and done % progress_every == 0:
            print(f"  enriched {done}/{len(ordered)} keys", flush=True)
        if checkpoint_every and len(rows) >= checkpoint_every:
            append_records(snapshot, run_id, rows)
            rows = []
    append_records(snapshot, run_id, rows)
    write_run_meta(
        snapshot,
        run_id,
        {
            "n_keys": len(ordered),
            "ct_timeout_s": ct_timeout,
            "rdap_timeout_s": rdap_timeout,
            "skip_hosted_queries": skip_hosted_queries,
            "workers": workers,
            "checkpoint_every": checkpoint_every,
            "signals": list(signals),
            "shuffle_seed": shuffle_seed,
            "population_manifest_sha": population_manifest_sha,
            "reason": reason,
        },
    )
    sidecar = seal_run(snapshot, run_id)
    print(
        f"sealed run {run_id}: {sidecar['n_records']} records, "
        f"sha256 {sidecar['sha256'][:12]}"
    )
    return sidecar
