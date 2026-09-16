"""Versioned enrichment snapshot store (separate artifact from splits).

Splits are deterministic and byte-reproducible from raw files; RDAP/CT
responses are not (a re-query next month returns different data — the same
class of problem as docs/WAIVERS.md). So raw provider responses live in
their own snapshot, keyed exactly like the live cache will be keyed, with
`enriched_at` timestamps and a hash recorded in repro/hashes.json.
The enriched feature table is built by joining this snapshot onto the
split — re-enriching later never creates a new population.

Anti-leak rule (storage layer): records are keyed by (cache key, run id)
and each run is SEALED before hashing. A later re-enrichment whose lookup
fails (domain lapsed — the takedown case) must never overwrite an earlier
successful lookup, so `load_snapshot` keeps no "last wins" semantics:
the join reads one explicitly pinned run, or the earliest successful
lookup per key under `select_earliest_success`. An open (unsealed) file
can never be pinned.

CT rule: the snapshot holds FULL certificate history per key; the
pre-first_seen filter runs at JOIN time per row (same domain recurs with
different first_seen values — filtering at query time would be wrong for
every row after the first).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def snapshot_path(index: str, date: str) -> str:
    return f"data/enrichment-{index}-{date}.jsonl"


def sealed_sidecar(path: Path, run_id: str) -> Path:
    return path.with_name(f"{path.name}.run-{run_id}.sealed.json")


def run_meta_path(path: Path, run_id: str) -> Path:
    return path.with_name(f"{path.name}.run-{run_id}.meta.json")


def write_run_meta(path: Path, run_id: str, meta: dict[str, Any]) -> None:
    """Record what a run used (timeouts, provider order, bootstrap sha).

    Written before sealing; sealed runs are immutable, so the meta is the
    audit trail of the conditions the unknown-rate gate must be read under.
    """
    from datetime import datetime, timezone

    run_meta_path(path, run_id).write_text(
        json.dumps(
            {
                **meta,
                "run_id": run_id,
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def append_records(path: Path, run_id: str, records: list[dict[str, Any]]) -> None:
    """Append raw provider records for one run (resumable within the run).

    Every record carries its `run_id`; re-running the same run id skips
    keys already stored under it (a no-op resume), and never touches other
    runs' rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    known: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("run_id") == run_id:
                        known.add(str(r.get("cache_key")))
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            if str(r.get("cache_key")) in known:
                continue
            row = {
                **r,
                "run_id": run_id,
                "enriched_at": r.get("enriched_at", stamp),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            known.add(str(row.get("cache_key")))


def seal_run(path: Path, run_id: str) -> dict[str, Any]:
    """Seal one run: hash its records and write the sidecar.

    Only a sealed run may be pinned (in repro/hashes.json) or joined.
    Sealing is idempotent; the sidecar records count + sha256 + time.
    """
    h = hashlib.sha256()
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("run_id") == run_id:
                h.update(json.dumps(r, sort_keys=True).encode("utf-8") + b"\n")
                n += 1
    sidecar = {
        "snapshot": path.name,
        "run_id": run_id,
        "n_records": n,
        "sha256": h.hexdigest(),
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sealed_sidecar(path, run_id).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sidecar


def run_keys(path: Path, run_id: str) -> set[str]:
    """Cache keys already stored under a run (resume skips them pre-fetch).

    Reads keys only, not payloads: a resume must not re-hit the network
    for keys it already holds.
    """
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("run_id") == run_id and r.get("cache_key"):
                keys.add(str(r["cache_key"]))
    return keys


def load_pinned_run(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    """Load one explicitly pinned run, keyed by cache key."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("run_id") == run_id:
                out[str(r["cache_key"])] = r
    return out


def _successful(r: dict[str, Any]) -> bool:
    """A lookup counts as successful when it resolved or is explicitly na.

    Raw batch rows carry ``rdap``/``ct`` payloads (plus a ``hosted`` flag);
    derived-style rows carry ``*_known``/``*_na`` flags. Either shape
    resolves here. Pure-unknown rows (all lookups failed) are never
    "earlier truth" — they are the absence of data, and must not shadow a
    real lookup from another run in either direction.
    """
    if r.get("hosted"):
        return True  # explicitly na: resolved, not missing
    rdap_payload = r.get("rdap") or {}
    ct_payload = r.get("ct") or {}
    if rdap_payload.get("creation_date") is not None:
        return True
    if ct_payload.get("certs") is not None or ct_payload.get("truncated"):
        return True
    return bool(
        r.get("age_known") or r.get("ct_known") or r.get("age_na") or r.get("ct_na")
    )


def select_earliest_success(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Earliest successful lookup per cache key (documented join fallback).

    Used only when no single run is pinned: for each key, the successful
    record with the smallest `enriched_at` wins; keys with no successful
    record anywhere map to unknown. A later failure (lapsed domain) can
    never displace an earlier success — the takedown leak stays closed
    through the storage layer.
    """
    best: dict[str, dict[str, Any]] = {}
    for r in sorted(records, key=lambda d: str(d.get("enriched_at", ""))):
        key = str(r.get("cache_key"))
        if key in best or not _successful(r):
            continue
        best[key] = r
    return best


def na_unknown_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Contamination gate input: na/unknown rates per class and stratum.

    `rows` are joined rows carrying `label`, `survival_stratum`,
    `age_known`/`ct_known`, `age_na`/`ct_na`. na rows are EXCLUDED from
    the unknown rate (denominator and numerator): unknown means a failed
    lookup and nothing else — hosted tenants must not spend the unknown
    budget. The na share is gated on its own (`na_rate`): a hosted list
    curated from phishing would show up there as a per-class na gap,
    however good the AUC looks. Thresholds live in
    docs/phase3-preregistration.md, committed before the bulk run; this
    pins the measurement so the gate cannot be redefined around data.
    """
    out: dict[str, Any] = {}
    for signal in ("age", "ct"):
        for group_key in ("label", "survival_stratum"):
            groups: dict[str, dict[str, float]] = {}
            seen: set[str] = set()
            for r in rows:
                seen.add(str(r.get(group_key)))
            for g in sorted(seen):
                sub = [r for r in rows if str(r.get(group_key)) == g]
                n = len(sub)
                na = sum(1 for r in sub if r.get(f"{signal}_na"))
                eligible = [r for r in sub if not r.get(f"{signal}_na")]
                n_eligible = len(eligible)
                unk = sum(1 for r in eligible if not r.get(f"{signal}_known"))
                groups[g] = {
                    "n": n,
                    "n_eligible": n_eligible,
                    "unknown_rate": (unk / n_eligible) if n_eligible else 0.0,
                    "na_rate": (na / n) if n else 0.0,
                }
            out[f"{signal}_by_{group_key}"] = groups
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pre_first_seen_filter(
    certs: list[dict[str, Any]], first_seen_iso: str
) -> list[dict[str, Any]]:
    """Join-time filter: keep only certificates publicly visible pre-cutoff.

    Filters on the CT log entry time (`entry_timestamp`), NOT `not_before`:
    not_before can be backdated, and only the entry time says the
    certificate was publicly visible before first_seen. Entries without a
    parseable entry time are dropped (fail-closed toward fewer certs, never
    toward future knowledge). Runs per row at join time, never at query
    time — the same domain recurs with different first_seen values.
    """
    import pandas as pd

    cutoff = pd.Timestamp(first_seen_iso, tz="UTC")
    out: list[dict[str, Any]] = []
    for c in certs:
        ts_raw = c.get("entry_timestamp")
        if ts_raw is None:
            continue
        try:
            ts = pd.Timestamp(ts_raw, tz="UTC")
        except Exception:
            continue
        if ts < cutoff:
            out.append(c)
    return out
