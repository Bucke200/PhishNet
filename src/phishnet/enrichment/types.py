"""Shared enrichment schema (batch + future live + stub).

One record per cache key. Three states per signal, never conflated:

* value present  -> `<field>` set, `<field>_known=True`
* lookup failed  -> `<field>` None, `<field>_known=False` (never "nonexistent")
* not applicable -> `<field>` None, `<field>_na=True` (hosted-platform rows;
  the platform's age/cert is not the tenant's — see key.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnrichedRecord:
    cache_key: str
    # Domain age (RDAP creation date vs row first_seen; days or None).
    domain_age_days: float | None = None
    age_known: bool = False
    age_na: bool = False
    age_source: str | None = None  # provenance only (rdap/whois), never a feature
    # Certificate history (earliest pre-first_seen issuance vs first_seen).
    ct_age_days: float | None = None
    ct_cert_count_pre: int | None = None
    ct_known: bool = False
    ct_na: bool = False
    ct_provider: str | None = None  # provenance only, never a feature
    providers: tuple[str, ...] = field(default_factory=tuple)


UNKNOWN = EnrichedRecord(cache_key="")
