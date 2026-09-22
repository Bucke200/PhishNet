"""Shortener resolution for serving (C5, phase6-D).

Phase 5 measured the `is_shortened` leak: rewriting benign URLs onto covered
shorteners alerts 99.0% at `t05`, and forcing the flag to 0 still alerts
60.4% — most of the effect is the short-host / random-slug URL shape, not the
flag. Zeroing a trained feature is also serving/training skew. The registered
remediation is therefore to **follow the redirect and score the final URL**;
if it cannot be resolved the disposition is `can't assess`, never a score.

Method (phase6-D): GET, aborting after the response headers and before any
body is read (HEAD is rejected by several covered hosts and `goo.gl` is
dead), ≤ 5 hops, 2 s total budget, loop detection. The host list is imported
from the extractor so the resolver and the `is_shortened` feature cannot
drift.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin

import requests

from phishnet.enrichment.key import host_of, registrable_domain
from phishnet.features.extraction import SHORTENER_DOMAINS


class HttpResponse(Protocol):
    """Minimal response surface the resolver reads (headers only)."""

    status_code: int
    headers: Mapping[str, str]

    def close(self) -> None: ...


class HttpSession(Protocol):
    """Minimal session surface, satisfied by ``requests.Session``."""

    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...


MAX_HOPS = 5
TOTAL_BUDGET_S = 2.0
UNRESOLVED = "unresolved_shortener"
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True)
class Resolution:
    """Outcome of a shortener resolution attempt."""

    final_url: str | None
    hops: int
    reason: str

    @property
    def resolved(self) -> bool:
        return self.final_url is not None


def is_shortener(url: str) -> bool:
    """True iff the URL's host is a covered shortener (suffix match)."""
    host = host_of(url)
    if not host:
        return False
    if host in SHORTENER_DOMAINS:
        return True
    registered = registrable_domain(host)
    return registered in SHORTENER_DOMAINS


def resolve(
    url: str,
    *,
    max_hops: int = MAX_HOPS,
    budget: float = TOTAL_BUDGET_S,
    session: HttpSession | None = None,
) -> Resolution:
    """Follow redirects to the first non-shortener URL, or fail unresolved.

    A response with no redirect while still on a shortener host is a failure
    (dead link or interstitial), not a final destination: scoring the
    shortener URL itself is exactly the leak this avoids. Reads no body.
    """
    http = session if session is not None else requests.Session()
    deadline = time.monotonic() + budget
    current = url
    seen: set[str] = set()

    for hop in range(max_hops):
        if current in seen:
            return Resolution(None, hop, UNRESOLVED)
        seen.add(current)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Resolution(None, hop, UNRESOLVED)
        try:
            response = http.get(
                current,
                allow_redirects=False,
                stream=True,
                timeout=min(remaining, budget),
            )
        except requests.RequestException:
            return Resolution(None, hop, UNRESOLVED)
        try:
            location = response.headers.get("Location")
            if response.status_code in _REDIRECT_STATUS and location:
                nxt = urljoin(current, location)
                if not is_shortener(nxt):
                    return Resolution(nxt, hop + 1, "")
                current = nxt
                continue
            if not is_shortener(current):
                return Resolution(current, hop, "")
            return Resolution(None, hop, UNRESOLVED)
        finally:
            response.close()

    return Resolution(None, max_hops, UNRESOLVED)
