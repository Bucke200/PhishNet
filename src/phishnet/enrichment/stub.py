"""Unknown-stub provider: the serving stand-in for this phase.

Always returns unknown with all `*_known=False` (and `*_na` set for hosted
tenants, mirroring the batch rule that hosted rows are not-applicable
rather than failed). Grouping is by *cache key*: one miss covers every URL
under the key, matching the forced-miss simulation in eval (which must also
group by cache key, not by registrable domain, or the two disagree on
hosted tenants).

A test pins stub output == forced-100%-miss transform output so the
cold-start number and the serving behaviour share one contract.
"""

from __future__ import annotations

from phishnet.enrichment.key import cache_key
from phishnet.enrichment.types import EnrichedRecord


class UnknownStubProvider:
    name = "unknown_stub"

    def lookup(self, url: str) -> EnrichedRecord:
        key, hosted = cache_key(url)
        return EnrichedRecord(
            cache_key=key,
            age_na=hosted,
            ct_na=hosted,
        )

    def lookup_many(self, urls: list[str]) -> list[EnrichedRecord]:
        return [self.lookup(u) for u in urls]


def force_miss(records: list[EnrichedRecord]) -> list[EnrichedRecord]:
    """Forced-100%-miss transform: the eval-side twin of the stub.

    Used by the cold-start simulation (0/50/100% miss, grouped by cache
    key): at 100% every record becomes the stub's record for its key.
    """
    return [
        EnrichedRecord(cache_key=r.cache_key, age_na=r.age_na, ct_na=r.ct_na)
        for r in records
    ]
