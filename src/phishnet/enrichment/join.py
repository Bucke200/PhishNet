"""Join the enrichment snapshot onto split rows (Step 4 contract, pinned now).

Exactly ONE selection rule per join — either a pinned run or the
earliest-success fallback — recorded in the manifest fragment the join
returns. The unknown/na-rate gate MUST run on these same joined rows
(`na_unknown_rates(joined)`); gating on any other selection would check
different data than the model trains on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from phishnet.enrichment.key import cache_key
from phishnet.enrichment.store import (
    load_pinned_run,
    na_unknown_rates,
    select_earliest_success,
)
from phishnet.enrichment.types import EnrichedRecord


def _all_records(path: Path) -> list[dict[str, Any]]:
    import json

    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def join_enrichment(
    urls: list[str], snapshot: Path, selection: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join snapshot records onto URLs under exactly one selection rule.

    ``selection`` is ``{"rule": "pinned-run", "run_id": ...}`` (a sealed
    run — the only choice for a published population) or ``{"rule":
    "earliest-success"}`` (diagnostic fallback). Anything else raises:
    the join must never silently mix rules. Keys with no record resolve
    to unknown (all ``*_known`` False, hosted tenants na).
    """
    rule = selection.get("rule")
    if rule == "pinned-run":
        run_id = selection.get("run_id")
        if not run_id:
            raise ValueError("pinned-run selection needs a run_id")
        table = load_pinned_run(snapshot, str(run_id))
    elif rule == "earliest-success":
        table = select_earliest_success(_all_records(snapshot))
    else:
        raise ValueError(
            f"join needs exactly one rule "
            f"('pinned-run' or 'earliest-success'), got {rule!r}"
        )
    joined: list[dict[str, Any]] = []
    for u in urls:
        key, hosted = cache_key(u)
        rec = table.get(key)
        if rec is None:
            rec = {
                "cache_key": key,
                "age_known": False,
                "ct_known": False,
                "age_na": hosted,
                "ct_na": hosted,
            }
        joined.append({"url": u, "cache_key": key, **rec})
    known = sum(1 for r in joined if r.get("age_known") or r.get("ct_known"))
    manifest = {
        "snapshot": snapshot.name,
        "selection_rule": rule,
        **({"run_id": selection["run_id"]} if rule == "pinned-run" else {}),
        "n_urls": len(urls),
        "n_keys": len({r["cache_key"] for r in joined}),
        "n_keys_known": known,
        # The gate runs on THESE rows (same selection the model trains on).
        "contamination": na_unknown_rates(
            [
                {
                    "label": r.get("label", "unknown"),
                    "survival_stratum": r.get("survival_stratum", "unknown"),
                    "age_known": bool(r.get("age_known")),
                    "age_na": bool(r.get("age_na")),
                    "ct_known": bool(r.get("ct_known")),
                    "ct_na": bool(r.get("ct_na")),
                }
                for r in joined
            ]
        ),
    }
    return joined, manifest


def to_record(row: dict[str, Any]) -> EnrichedRecord:
    """Narrow a joined row to the serving schema (drops join metadata)."""
    return EnrichedRecord(
        cache_key=str(row.get("cache_key", "")),
        domain_age_days=row.get("domain_age_days"),
        age_known=bool(row.get("age_known")),
        age_na=bool(row.get("age_na")),
        age_source=row.get("age_source"),
        ct_age_days=row.get("ct_age_days"),
        ct_cert_count_pre=row.get("ct_cert_count_pre"),
        ct_known=bool(row.get("ct_known")),
        ct_na=bool(row.get("ct_na")),
        ct_provider=row.get("ct_provider"),
    )
