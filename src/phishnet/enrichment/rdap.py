"""Domain-age fetching: RDAP primary, port-43 WHOIS fallback.

Chain per cache key (see docs/point-in-time.md):

1. Find the RDAP server through the IANA bootstrap registry, query it.
2. For TLDs with no RDAP service, fall back to port-43 WHOIS with a
   small creation-date parser.
3. Take only the creation date. The ``source`` (rdap/whois) is recorded
   per row as provenance — never a feature (fallback coverage differs by
   TLD and phishing clusters on particular TLDs, so it would proxy the
   label). A failed lookup means unknown, never "doesn't exist".

One hard timeout per provider; enrich promptly after each feed snapshot
— every day of delay widens the phishing unknown-rate gap as
registrations lapse.
"""

from __future__ import annotations

import re
import socket
from typing import Any

import requests

IANA_BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"

RDAP_TIMEOUT_S = 10
RDAP_RETRIES = 2
# Registered per-server concurrency: at most this many simultaneous RDAP
# requests to one server (Amendment E review). With mostly-.com keys, an
# unbounded pool would point every worker at Verisign at once.
RDAP_PER_SERVER_LIMIT = 2
WHOIS_TIMEOUT_S = 10
WHOIS_PORT = 43
# Bounded raw-text retention for audit (full responses are unbounded).
WHOIS_RAW_LIMIT = 8000

# Creation-date lines across RDAP-fallback WHOIS servers. Deliberately
# narrow (creation only): expiry/updated dates are not collected, so they
# cannot leak into features later.
CREATION_PATTERNS = (
    r"(?im)^\s*creation date\s*:\s*(.+?)\s*$",
    r"(?im)^\s*created\s*:\s*(.+?)\s*$",
    r"(?im)^\s*registered on\s*:\s*(.+?)\s*$",
    r"(?im)^\s*registration time\s*:\s*(.+?)\s*$",
    r"(?im)^\s*domain registration date\s*:\s*(.+?)\s*$",
)


def fetch_bootstrap(timeout: int = RDAP_TIMEOUT_S) -> dict[str, list[str]]:
    """IANA bootstrap registry: TLD suffix -> RDAP server URLs."""
    r = requests.get(IANA_BOOTSTRAP_URL, timeout=timeout)
    r.raise_for_status()
    services = r.json().get("services", [])
    out: dict[str, list[str]] = {}
    for suffixes, urls in services:
        for suffix in suffixes:
            out[suffix.lower()] = [u.rstrip("/") + "/" for u in urls]
    return out


def rdap_servers_for(domain: str, bootstrap: dict[str, list[str]]) -> list[str]:
    """RDAP servers for a domain's TLD, longest-suffix match first."""
    labels = domain.lower().strip(".").split(".")
    for i in range(len(labels)):
        suffix = ".".join(labels[i:])
        if suffix in bootstrap:
            return bootstrap[suffix]
    return []


class ServerLimits:
    """Per-server concurrency gates for RDAP HTTP calls.

    One semaphore per server URL, taken around the single call about to
    contact that server and released immediately after. Only ever one
    slot is held at a time, so no acquisition ordering is needed and
    deadlock is impossible by construction. Registrar servers are
    discovered mid-lookup, so slots are resolved per request, never
    reserved upfront.
    """

    def __init__(self, limit: int = RDAP_PER_SERVER_LIMIT):
        import threading

        self._limit = limit
        self._lock = threading.Lock()
        self._slots: dict[str, Any] = {}
        self._threading = threading

    def slot(self, server: str) -> Any:
        """Context manager holding one slot on ``server``."""
        key = server.rstrip("/").lower()
        with self._lock:
            sem = self._slots.get(key)
            if sem is None:
                sem = self._threading.Semaphore(self._limit)
                self._slots[key] = sem
        return sem


DEFAULT_LIMITS = ServerLimits()


def parse_rdap_creation(payload: dict[str, Any]) -> str | None:
    """The ``registration`` event's date, or None when absent."""
    for event in payload.get("events", []) or []:
        if str(event.get("eventAction", "")).lower() == "registration":
            date = event.get("eventDate")
            if date:
                return str(date)
    return None


def rdap_lookup(
    domain: str,
    bootstrap: dict[str, list[str]],
    timeout: int = RDAP_TIMEOUT_S,
    retries: int = RDAP_RETRIES,
    limits: ServerLimits | None = None,
) -> dict[str, Any]:
    """Query RDAP for a domain's creation date (with retry).

    Each HTTP call runs under that server's concurrency slot
    (``limits``, default shared). The port-43 WHOIS fallback has no
    gate: it fires only on RDAP misses, at low volume by construction.
    """
    gates = limits or DEFAULT_LIMITS
    servers = rdap_servers_for(domain, bootstrap)
    if not servers:
        return {
            "creation_date": None,
            "source": None,
            "server": None,
            "error": "no-rdap-service",
        }
    last_error: str | None = None
    for attempt in range(retries + 1):
        for server in servers:
            try:
                with gates.slot(server):
                    r = requests.get(
                        f"{server}domain/{domain}",
                        timeout=timeout,
                        headers={"Accept": "application/rdap+json"},
                    )
                if r.status_code == 404:
                    return {
                        "creation_date": None,
                        "source": None,
                        "server": server,
                        "error": "rdap-404",
                    }
                r.raise_for_status()
                return {
                    "creation_date": parse_rdap_creation(r.json()),
                    "source": "rdap",
                    "server": server,
                    "error": None,
                }
            except (requests.RequestException, ValueError) as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
        if attempt < retries:
            import time

            time.sleep(2**attempt)
    return {
        "creation_date": None,
        "source": None,
        "server": servers[0],
        "error": last_error or "rdap-failed",
    }


def _whois_query(server: str, query: str, timeout: int) -> str:
    with socket.create_connection((server, WHOIS_PORT), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall((query + "\r\n").encode("utf-8", errors="replace"))
        chunks = []
        while True:
            try:
                data = s.recv(4096)
            except TimeoutError:
                break
            if not data:
                break
            chunks.append(data)
            if sum(len(c) for c in chunks) > WHOIS_RAW_LIMIT * 4:
                break
    return b"".join(chunks).decode("utf-8", errors="replace")


def _referral_server(tld: str, timeout: int) -> str | None:
    """IANA whois referral for a TLD (``whois: <server>`` line)."""
    try:
        text = _whois_query("whois.iana.org", tld, timeout)
    except (TimeoutError, OSError):
        return None
    m = re.search(r"(?im)^whois:\s*(\S+)\s*$", text)
    return m.group(1).strip() if m else None


def parse_whois_creation(text: str) -> str | None:
    """First creation-date match across the known server phrasings."""
    for pattern in CREATION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def whois_lookup(domain: str, timeout: int = WHOIS_TIMEOUT_S) -> dict[str, Any]:
    """Port-43 WHOIS fallback: IANA referral, then creation date."""
    tld = domain.lower().strip(".").split(".")[-1]
    server = _referral_server(tld, timeout)
    if server is None:
        # Conventional gTLD host as a second resort (no query on failure).
        server = f"{tld}.whois-servers.net"
    try:
        text = _whois_query(server, domain, timeout)
    except (TimeoutError, OSError) as e:
        return {
            "creation_date": None,
            "source": None,
            "server": server,
            "error": f"{type(e).__name__}: {e}",
        }
    creation = parse_whois_creation(text)
    return {
        "creation_date": creation,
        "source": "whois" if creation else None,
        "server": server,
        "error": None if creation else "whois-no-creation-date",
        "raw_excerpt": text[:WHOIS_RAW_LIMIT],
    }


def fetch_age(
    domain: str,
    bootstrap: dict[str, list[str]] | None,
    timeout: int = RDAP_TIMEOUT_S,
    limits: ServerLimits | None = None,
) -> dict[str, Any]:
    """Full chain for one registrable domain: RDAP, then WHOIS fallback.

    IP literals skip both (no domain registration exists) and return
    unknown. The caller stores the payload raw; age-in-days is derived at
    join time against the row's first_seen (same domain recurs with
    different first_seen values).
    """
    if re.fullmatch(r"[0-9a-fA-F:.]+", domain) and (
        ":" in domain or re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", domain)
    ):
        return {
            "creation_date": None,
            "source": None,
            "server": None,
            "error": "ip-literal",
        }
    if bootstrap:
        result = rdap_lookup(domain, bootstrap, timeout, limits=limits)
        if result.get("creation_date") or result.get("error") == "rdap-404":
            return result
    else:
        result = {"error": "no-bootstrap"}
    fallback = whois_lookup(domain, timeout=WHOIS_TIMEOUT_S)
    fallback["rdap_error"] = result.get("error")
    return fallback
