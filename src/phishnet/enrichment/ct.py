"""Certificate-history fetching: crt.sh primary, Postgres/Spotter fallback.

Chain per cache key (see docs/point-in-time.md):

1. crt.sh JSON endpoint, with backoff on 429/5xx/timeouts.
2. On repeated failure, crt.sh's public Postgres interface (bulk-safe;
   needs ``psycopg`` + ``CRT_PG_DSN`` — optional dependency, imported
   lazily so the default install stays lean).

No third provider: Cert Spotter was evaluated as the candidate last
resort and rejected — its issuances API exposes ``notBefore`` but no
log-entry time, so its rows cannot feed the entry-timestamp filter
without reintroducing the backdating the filter exists to exclude.
A provider that cannot prove public visibility pre-first_seen is not a
fallback; it is a different (weaker) signal.

The snapshot stores FULL history per key; the pre-first_seen filter
runs at JOIN time per row (``store.pre_first_seen_filter`` on
``entry_timestamp`` — never ``not_before``, which can be backdated).
Hosted-platform rows skip querying entirely (marked na at the batch
layer): the platform's wildcard cert would otherwise masquerade as the
tenant's history.
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests

CRT_JSON_URL = "https://crt.sh/?q=%.{domain}&output=json"

CRT_TIMEOUT_S = 20
CRT_RETRIES = 3
# Bound per-key history (giants like googleapis); truncation is recorded
# so the join can treat capped histories as unknown rather than young.
CT_ROW_CAP = 20_000

_CERT_FIELDS = ("id", "entry_timestamp", "not_before", "not_after", "common_name")


def _project(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in _CERT_FIELDS}


def crtsh_lookup(
    domain: str,
    timeout: int = CRT_TIMEOUT_S,
    retries: int = CRT_RETRIES,
) -> dict[str, Any]:
    """Full issuance history for %.domain via crt.sh JSON (with backoff)."""
    url = CRT_JSON_URL.format(domain=quote(domain, safe=""))
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(
                url, timeout=timeout, headers={"Accept": "application/json"}
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"http-{r.status_code}"
            else:
                r.raise_for_status()
                certs = [_project(c) for c in r.json()]
                truncated = len(certs) >= CT_ROW_CAP
                return {
                    "certs": certs[:CT_ROW_CAP],
                    "provider": "crt.sh-json",
                    "truncated": truncated,
                    "error": None,
                }
        except (requests.RequestException, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        "certs": None,
        "provider": None,
        "truncated": False,
        "error": last_error or "crtsh-failed",
    }


def postgres_lookup(domain: str, dsn: str | None = None) -> dict[str, Any]:
    """Bulk-safe fallback via crt.sh's public Postgres interface."""
    dsn = dsn or os.environ.get("CRT_PG_DSN")
    if not dsn:
        return {
            "certs": None,
            "provider": None,
            "truncated": False,
            "error": "no-pg-dsn",
        }
    try:
        psycopg = __import__("psycopg")
    except ImportError:
        return {
            "certs": None,
            "provider": None,
            "truncated": False,
            "error": "psycopg-not-installed",
        }
    try:
        query = (
            "SELECT c.id, c.ENTRY_TIMESTAMP, c.NOT_BEFORE, c.NOT_AFTER, "
            "cn.COMMON_NAME FROM certificate c "
            "JOIN certificate_identity ci ON c.id = ci.certificate_id "
            "JOIN ca_name cn ON ci.ca_name_id = cn.id "
            "WHERE ci.NAME_VALUE ILIKE %s"
        )
        with psycopg.connect(dsn, connect_timeout=20) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (f"%.{domain}",))
                rows = cur.fetchmany(CT_ROW_CAP + 1)
    except Exception as e:  # driver/network failure: recorded, not raised
        return {
            "certs": None,
            "provider": None,
            "truncated": False,
            "error": f"{type(e).__name__}: {e}",
        }
    truncated = len(rows) > CT_ROW_CAP
    certs = [
        {
            "id": r[0],
            "entry_timestamp": str(r[1]) if r[1] is not None else None,
            "not_before": str(r[2]) if r[2] is not None else None,
            "not_after": str(r[3]) if r[3] is not None else None,
            "common_name": r[4],
        }
        for r in rows[:CT_ROW_CAP]
    ]
    return {
        "certs": certs,
        "provider": "crt.sh-postgres",
        "truncated": truncated,
        "error": None,
    }


def fetch_ct(
    domain: str,
    timeout: int = CRT_TIMEOUT_S,
    *,
    allow_postgres: bool = True,
) -> dict[str, Any]:
    """Full chain for one key: crt.sh JSON, then Postgres on failure.

    Postgres runs only with ``CRT_PG_DSN`` set (else its "error" records
    why it didn't run). Each fallback records the earlier error, so the
    provenance shows the whole chain, not just the winner.
    """
    result = crtsh_lookup(domain, timeout)
    if result.get("certs") is not None:
        return result
    errors = [result.get("error")]
    if allow_postgres:
        result = postgres_lookup(domain)
        if result.get("certs") is not None:
            result["crtsh_error"] = errors[0]
            return result
        errors.append(result.get("error"))
    return {
        "certs": None,
        "provider": None,
        "truncated": False,
        "error": "; ".join(e for e in errors if e),
    }
