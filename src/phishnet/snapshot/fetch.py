"""Single governed fetch per row (fetch once, freeze, §5).

One row → one `FetchResult`: raw HTML (bytes, never sent to the model),
final URL, HTTP status, redirect chain, and an outcome code from the §3.2
taxonomy (`ok`, `dns_fail`, `conn_refused`, `tls_fail`, `timeout`,
`http_4xx`, `http_5xx`, `parked`). Transport failures are classified from
the exception/request state, never collapsed to a boolean. `parked` is an
explicit content heuristic recorded here (registrar-parking phrases in an
otherwise-fetchable page), kept as its own outcome so the Step-0 table can
show it rather than absorbing it into `ok`.

Bodies live under `data/snapshots-p4/` (gitignored); only the manifest
(`reports/snapshot-manifest-p4.json`) is committed. Dual hashes —
`sha256(raw_html)` and `sha256(canonical_extract)` — are stored separately
per row.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

import requests

OUTCOMES = (
    "ok",
    "dns_fail",
    "conn_refused",
    "tls_fail",
    "timeout",
    "http_4xx",
    "http_5xx",
    "parked",
)

PARKED_PHRASES = (
    "domain is parked",
    "this domain is for sale",
    "buy this domain",
    "parked domain",
    "domain parking",
)

UA = "PhishNet-phase4-snapshot/1.0 (+research corpus; single fetch)"


@dataclass
class FetchResult:
    url: str
    final_url: str | None
    status: int | None
    redirect_chain: list[str] = field(default_factory=list)
    outcome: str = "timeout"
    raw_html: bytes = b""
    fetched_at: str = ""
    latency_ms: float = 0.0

    def raw_hash(self) -> str:
        return hashlib.sha256(self.raw_html).hexdigest()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def looks_parked(html: str) -> bool:
    lowered = html.lower()
    return any(phrase in lowered for phrase in PARKED_PHRASES)


def classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = f"{exc}".lower()
    if "ssl" in name or "tls" in text or "certificate" in text:
        return "tls_fail"
    if "connect" in name and "refus" in text:
        return "conn_refused"
    if "timeout" in name or "timed out" in text:
        return "timeout"
    if "dns" in text or "name resolution" in text or "getaddrinfo" in text:
        return "dns_fail"
    if "refused" in text:
        return "conn_refused"
    return "timeout"


def fetch_once(url: str, timeout: int = 20) -> FetchResult:
    """Fetch one URL exactly once; never retried, never re-fetched."""
    from datetime import datetime, timezone

    t0 = time.time()
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=timeout,
            allow_redirects=True,
        )
        latency_ms = (time.time() - t0) * 1000.0
        chain = [r.url for r in resp.history] + [resp.url]
        fetched_at = datetime.now(timezone.utc).isoformat()
        if resp.status_code >= 500:
            outcome = "http_5xx"
        elif resp.status_code >= 400:
            outcome = "http_4xx"
        else:
            body = resp.content or b""
            try:
                text = body.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            outcome = "parked" if looks_parked(text) else "ok"
        return FetchResult(
            url=url,
            final_url=resp.url,
            status=resp.status_code,
            redirect_chain=chain,
            outcome=outcome,
            raw_html=resp.content or b"",
            fetched_at=fetched_at,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000.0
        return FetchResult(
            url=url,
            final_url=None,
            status=None,
            redirect_chain=[],
            outcome=classify_exception(exc),
            raw_html=b"",
            fetched_at=_now_iso(),
            latency_ms=latency_ms,
        )


def page_host(url: str) -> str:
    """Lowercased host of a URL (empty string when unparsable)."""
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def to_record(result: FetchResult, extract_hash: str) -> dict:
    """Manifest row: hashes + metadata, never the body."""
    record = asdict(result)
    record.pop("raw_html")
    record["sha256_raw_html"] = result.raw_hash()
    record["sha256_canonical_extract"] = extract_hash
    return record
