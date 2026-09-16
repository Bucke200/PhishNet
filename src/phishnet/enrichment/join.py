"""Join the enrichment snapshot onto split rows (Step 4 contract, pinned now).

Exactly ONE selection rule per join — either a pinned run or the
earliest-success fallback — recorded in the manifest fragment the join
returns. Raw snapshot payloads (RDAP creation dates, full CT histories)
are derived per row against the row's own ``first_seen`` (same domain
recurs with different stamps): hosted tenants resolve na without ever
querying, failed lookups resolve unknown, truncated CT histories resolve
unknown (a capped history cannot prove youth). Negative ages (creation
after observation) resolve unknown — fail-closed, never a negative
feature.

The unknown/na-rate gate MUST run on these same joined rows
(``check_contamination`` on the returned ``contamination`` block);
gating on any other selection would check different data than the model
trains on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from phishnet.enrichment.key import cache_key
from phishnet.enrichment.store import (
    load_pinned_run,
    na_unknown_rates,
    pre_first_seen_filter,
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


def _days_between(later_iso: Any, earlier_iso: Any) -> float | None:
    import pandas as pd

    try:
        later = pd.Timestamp(later_iso, tz="UTC")
        earlier = pd.Timestamp(earlier_iso, tz="UTC")
    except Exception:
        return None
    if pd.isna(later) or pd.isna(earlier):
        return None
    return (later - earlier).total_seconds() / 86400.0


def derive_row(
    url: str,
    raw: dict[str, Any] | None,
    first_seen_iso: Any,
    hosted: bool,
) -> dict[str, Any]:
    """Feature fields for one joined row (raw payload + row timestamp)."""
    base: dict[str, Any] = {"url": url}
    if hosted:
        return {
            **base,
            "age_known": False,
            "age_na": True,
            "ct_known": False,
            "ct_na": True,
        }
    raw = raw or {}
    rdap_payload = raw.get("rdap") or {}
    ct_payload = raw.get("ct") or {}
    out: dict[str, Any] = {**base, "age_na": False, "ct_na": False}
    # Age: creation date must parse and must not postdate observation.
    age_days = (
        _days_between(first_seen_iso, rdap_payload.get("creation_date"))
        if rdap_payload.get("creation_date") and first_seen_iso
        else None
    )
    if age_days is not None and age_days >= 0:
        out.update(
            {
                "domain_age_days": age_days,
                "age_known": True,
                "age_source": rdap_payload.get("source"),
            }
        )
    else:
        out.update({"domain_age_days": None, "age_known": False, "age_source": None})
    # CT: the lookup must have succeeded (certs list present, even empty);
    # truncated histories cannot prove youth and resolve unknown.
    certs = ct_payload.get("certs")
    if certs is None or ct_payload.get("truncated"):
        out.update(
            {
                "ct_age_days": None,
                "ct_cert_count_pre": None,
                "ct_known": False,
                "ct_provider": None,
            }
        )
        return out
    pre = pre_first_seen_filter(certs, str(first_seen_iso)) if first_seen_iso else []
    stamps = cast(
        "list[str]",
        [c.get("entry_timestamp") for c in pre if c.get("entry_timestamp")],
    )
    earliest = min(stamps, default=None)
    ct_age = _days_between(first_seen_iso, earliest) if earliest else None
    out.update(
        {
            "ct_age_days": ct_age,
            "ct_cert_count_pre": len(pre),
            "ct_known": True,
            "ct_provider": ct_payload.get("provider"),
        }
    )
    return out


def check_contamination(
    contamination: dict[str, Any],
    *,
    max_unknown_gap: float,
    max_na_gap: float,
) -> dict[str, Any]:
    """Gate verdict on per-class unknown/na gaps (thresholds explicit).

    Compares the phishing ("1") vs benign ("0") rows of the
    ``*_by_label`` blocks: a signal whose lookups fail (or go na)
    noticeably more often on one class is contaminated and stays out of
    the headline, however good its AUC looks. Thresholds have no defaults
    — they are set when the rebuild lands, not redefined around data.
    A missing class reads "unmeasurable" (fail-closed, never a pass).
    """
    gaps: dict[str, dict[str, float]] = {}
    verdict = "pass"
    for signal in ("age", "ct"):
        block = contamination.get(f"{signal}_by_label", {})
        phish = block.get("1")
        benign = block.get("0")
        if not phish or not benign:
            verdict = "unmeasurable"
            gaps[signal] = {"unknown_gap": float("nan"), "na_gap": float("nan")}
            continue
        unknown_gap = abs(phish["unknown_rate"] - benign["unknown_rate"])
        na_gap = abs(phish["na_rate"] - benign["na_rate"])
        gaps[signal] = {"unknown_gap": unknown_gap, "na_gap": na_gap}
        if unknown_gap > max_unknown_gap or na_gap > max_na_gap:
            verdict = "fail"
    return {
        "verdict": verdict,
        "max_unknown_gap": max_unknown_gap,
        "max_na_gap": max_na_gap,
        "gaps": gaps,
    }


def join_enrichment(
    rows: list[dict[str, Any]], snapshot: Path, selection: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join snapshot records onto split rows under exactly one rule.

    ``rows`` carry ``url`` plus ``label``/``first_seen``/``survival_stratum``
    when known (missing provenance groups under "unknown" in the gate
    input). ``selection`` is ``{"rule": "pinned-run", "run_id": ...}`` (a
    sealed run — the only choice for a published population) or ``{"rule":
    "earliest-success"}`` (diagnostic fallback). Anything else raises.
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
    for row in rows:
        url = str(row.get("url", ""))
        key, hosted = cache_key(url)
        rec = table.get(key)
        if rec is None:
            rec = {"cache_key": key}
        joined.append(
            {
                "cache_key": key,
                "label": str(row.get("label", "unknown")),
                "survival_stratum": str(row.get("survival_stratum", "unknown")),
                **derive_row(url, rec, row.get("first_seen"), hosted),
            }
        )
    known = sum(1 for r in joined if r.get("age_known") or r.get("ct_known"))
    manifest = {
        "snapshot": snapshot.name,
        "selection_rule": rule,
        **({"run_id": selection["run_id"]} if rule == "pinned-run" else {}),
        "n_urls": len(rows),
        "n_keys": len({r["cache_key"] for r in joined}),
        "n_keys_known": known,
        # The gate runs on THESE rows (same selection the model trains on).
        "contamination": na_unknown_rates(
            [
                {
                    "label": r["label"],
                    "survival_stratum": r["survival_stratum"],
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
