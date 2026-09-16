"""Provider interface shared by batch enrichment, the stub, and Phase-6 live."""

from __future__ import annotations

from typing import Protocol

from phishnet.enrichment.types import EnrichedRecord


class EnrichmentProvider(Protocol):
    """Anything that can resolve a URL to its enrichment record.

    Implementations: batch (RDAP/crt.sh snapshot builder, Step 3),
    stub (all-unknown, this phase's serving stand-in), live (Phase 6).
    Misses are domain-grouped at the *cache-key* level: every enriched
    field for a key is unknown together (see stub.py).
    """

    name: str

    def lookup(self, url: str) -> EnrichedRecord: ...

    def lookup_many(self, urls: list[str]) -> list[EnrichedRecord]: ...
