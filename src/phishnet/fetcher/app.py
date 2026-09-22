"""Tier-2 page fetcher, a separate image from the Tier-1 server.

Serving decision 2 keeps Playwright out of the Tier-1 image so the latency
number is not measured beside a browser. This service fetches one URL and
returns the frozen canonical extract (raw HTML is never returned). It renders
with Playwright when available and falls back to a plain HTTP fetch; the
extract is the same pure function either way, so the model sees the same
shape.

**Structured outcome contract (2026-09-22).** The endpoint always answers HTTP
200 with a discriminated body, so a target-side failure is never confused with
a fetcher-side RPC failure:

  * success: ``{"ok": true, "url", "final_url", "extract"}``
  * target failure: ``{"ok": false, "url", "stage": "origin_fetch",
    "error": <mechanism>, "status_code": <int|null>, "detail": <str>}``

``error`` is one of ``http_403``, ``http_404``, ``http_5xx``, ``blocked``
(WAF/bot-wall/challenge interstitial), ``dns``, ``refused``, ``tls``,
``origin_timeout``, ``other``. The mechanism is what the serving failure
policy keys on, so it must survive the RPC boundary instead of collapsing to
the caller's own exception class.

Live mode is opt-in: `LiveTier2Provider` posts here only when
`PHISHNET_FETCHER_URL` (and a key) are set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import requests
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from phishnet.snapshot.extract import canonical_extract

UA = "PhishNet-phase6-fetcher/1.0 (+research; single fetch)"
# Origin fetch budget: an origin that has not answered in this long is dead,
# tarpitting, or down (distinct from the provider->fetcher RPC timeout).
TIMEOUT_S = float(os.getenv("PHISHNET_FETCH_TIMEOUT", "8"))
RENDER = os.getenv("PHISHNET_FETCHER_RENDER", "1") == "1"

# WAF / bot-wall interstitials: Cloudflare, Akamai, DataDome and
# PerimeterX/HUMAN return HTTP 200 with a JS challenge, so the status check
# cannot catch them. A title alone misses interstitials whose title is the
# origin domain or empty (Cloudflare "Under Attack", DataDome, PerimeterX),
# so the visible text and the challenge-specific script/iframe tokens are
# checked too. A challenge page is classified `blocked`, never handed to the
# LLM as page content.
CHALLENGE_TEXT_MARKERS = (
    "just a moment",
    "attention required",
    "suspected phishing",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "verify you are human",
    "verifying you are human",
    "checking your browser",
    "please wait while we verify",
    "you have been blocked",
    "ddos protection by cloudflare",
    "performance & security by cloudflare",
)
CHALLENGE_HTML_MARKERS = (
    "cf-browser-verification",
    "challenge-platform",
    "__cf_chl",
    "cf_chl_opt",
    "captcha-delivery.com",  # DataDome challenge iframe
    "px-captcha",  # PerimeterX / HUMAN challenge
)
# Akamai's reference structure is unambiguous, unlike the loose "Reference #"
# / "Access Denied" text that generic application 403s also carry.
AKAMAI_REFERENCE_RE = re.compile(
    r"reference\s*#\s*[0-9a-f]{1,4}\.[0-9a-f]{1,32}\.[0-9a-f]{1,32}", re.IGNORECASE
)
# Cloudflare Access / Zero Trust identity gates are authentication state
# machines, not phishing pages. Detected via the Access gate path only; IdP
# host strings are deliberately not used (phishing pages reference them too).
AUTH_GATEWAY_HTML_MARKERS = ("/cdn-cgi/access/",)

app = FastAPI(title="PhishNet fetcher (Phase 6)")


class FetchRequest(BaseModel):
    url: HttpUrl


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one fetch attempt, target-side only."""

    ok: bool
    final_url: str | None = None
    html: str | None = None
    error: str = ""
    status_code: int | None = None
    detail: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockHit:
    """A WAF/bot-wall interstitial match and where it was found."""

    marker: str
    trigger_type: str  # "text_token" | "html_token" | "header"


def block_error(
    title: str,
    visible_text: str = "",
    html: str = "",
    headers: dict[str, str] | None = None,
) -> BlockHit | None:
    """Return the matching interstitial, or None for a normal page.

    Header signals (``cf-mitigated: challenge``) are checked first, then the
    title/visible text for human-facing challenge phrases, then the raw HTML
    for challenge-specific script/iframe tokens (a bare
    ``datadome``/``perimeterx`` script reference is deliberately not used: a
    protected page that passed the check still carries it).
    """
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    if hdrs.get("cf-mitigated", "").lower() == "challenge":
        return BlockHit("cf-mitigated", "header")
    haystack = f"{title}\n{visible_text}"
    low = haystack.lower()
    for marker in CHALLENGE_TEXT_MARKERS:
        if marker in low:
            return BlockHit(marker, "text_token")
    if AKAMAI_REFERENCE_RE.search(haystack):
        return BlockHit("akamai_reference", "text_token")
    low_html = (html or "").lower()
    for marker in CHALLENGE_HTML_MARKERS:
        if marker in low_html:
            return BlockHit(marker, "html_token")
    return None


def auth_gateway_error(html: str) -> str | None:
    """Return the marker if the page is a Cloudflare Access identity gate."""
    low_html = (html or "").lower()
    for marker in AUTH_GATEWAY_HTML_MARKERS:
        if marker in low_html:
            return marker
    return None


def trigger_for(error: str, status_code: int | None) -> tuple[str, str]:
    """Classify a non-block failure as (trigger_type, trigger_match)."""
    if error in ("http_403", "http_404", "http_5xx"):
        return "status", str(status_code) if status_code is not None else error
    if error in ("dns", "refused", "tls", "origin_timeout", "other"):
        return "network", error
    return "internal", error


def classify_connection_error(message: str) -> str:
    """Map a requests `ConnectionError` message to `dns` or `refused`."""
    low = message.lower()
    if any(
        key in low
        for key in (
            "failed to resolve",
            "name resolution",
            "getaddrinfo",
            "name or service not known",
            "nodename nor servname",
        )
    ):
        return "dns"
    return "refused"


def classify_playwright_error(message: str) -> str | None:
    """Map a Playwright navigation error to a mechanism, or None (unknown)."""
    low = message.lower()
    if "err_name_not_resolved" in low:
        return "dns"
    if any(
        key in low
        for key in (
            "err_connection_refused",
            "err_connection_reset",
            "err_connection_closed",
        )
    ):
        return "refused"
    if "err_connection_timed_out" in low or "err_timed_out" in low or "timeout" in low:
        return "origin_timeout"
    if "err_cert" in low or "ssl" in low or "certificate" in low:
        return "tls"
    return None


def _status_error(status: int) -> FetchResult | None:
    if status in (401, 403):
        return FetchResult(
            False, error="http_403", status_code=status, detail=f"http {status}"
        )
    if status == 404:
        return FetchResult(
            False, error="http_404", status_code=status, detail="http 404"
        )
    if status >= 500:
        return FetchResult(
            False, error="http_5xx", status_code=status, detail=f"http {status}"
        )
    return None


def _fetch_with_playwright(url: str, timeout: float) -> FetchResult | None:
    """Render with Playwright, or None to fall back to plain HTTP."""
    try:
        from playwright.sync_api import Error as PWError
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            try:
                page = browser.new_page()
                response = page.goto(
                    url, wait_until="domcontentloaded", timeout=timeout * 1000
                )
                status = response.status if response is not None else None
                headers = dict(response.headers) if response is not None else {}
                final_url = page.url
                html = page.content()
            finally:
                browser.close()
    except PWTimeout:
        return FetchResult(
            False, error="origin_timeout", detail="playwright navigation timeout"
        )
    except PWError as exc:
        code = classify_playwright_error(str(exc))
        if code is None:
            return None
        return FetchResult(False, error=code, detail=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001 - never leak a 500
        return FetchResult(
            False, error="other", detail=f"{type(exc).__name__}: {exc}"[:200]
        )
    status_failure = _status_error(status) if status is not None else None
    if status_failure is not None:
        return FetchResult(
            False,
            final_url=final_url,
            error=status_failure.error,
            status_code=status_failure.status_code,
            detail=status_failure.detail,
            headers=headers,
        )
    return FetchResult(
        True, final_url=final_url, html=html, status_code=status, headers=headers
    )


def _fetch_with_requests(url: str, timeout: float) -> FetchResult:
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True
        )
    except requests.exceptions.ConnectTimeout:
        return FetchResult(False, error="origin_timeout", detail="connect timeout")
    except requests.exceptions.ReadTimeout:
        return FetchResult(False, error="origin_timeout", detail="read timeout")
    except requests.exceptions.SSLError as exc:
        return FetchResult(False, error="tls", detail=str(exc)[:200])
    except requests.exceptions.ConnectionError as exc:
        return FetchResult(
            False, error=classify_connection_error(str(exc)), detail=str(exc)[:200]
        )
    except requests.exceptions.RequestException as exc:
        return FetchResult(
            False, error="other", detail=f"{type(exc).__name__}: {exc}"[:200]
        )
    status_failure = _status_error(response.status_code)
    headers = {k.lower(): v for k, v in response.headers.items()}
    if status_failure is not None:
        return FetchResult(
            False,
            final_url=str(response.url),
            error=status_failure.error,
            status_code=status_failure.status_code,
            detail=status_failure.detail,
            headers=headers,
        )
    return FetchResult(
        True,
        final_url=str(response.url),
        html=response.text,
        status_code=response.status_code,
        headers=headers,
    )


def fetch_page(url: str, timeout: float = TIMEOUT_S) -> FetchResult:
    """Fetch one URL; Playwright when available, else plain HTTP."""
    if RENDER:
        result = _fetch_with_playwright(url, timeout)
        if result is not None:
            return result
    return _fetch_with_requests(url, timeout)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "render": RENDER, "timeout_s": TIMEOUT_S}


@app.post("/fetch")
def fetch(body: FetchRequest) -> dict[str, object]:
    url = str(body.url)
    result = fetch_page(url)
    if not result.ok or result.html is None:
        error = result.error or "other"
        trigger_type, trigger_match = trigger_for(error, result.status_code)
        return {
            "ok": False,
            "url": url,
            "stage": "origin_fetch",
            "error": error,
            "status_code": result.status_code,
            "trigger_type": trigger_type,
            "trigger_match": trigger_match,
            "detail": result.detail,
        }
    extract = canonical_extract(result.html, result.final_url or url)
    gateway = auth_gateway_error(result.html or "")
    if gateway is not None:
        return {
            "ok": False,
            "url": url,
            "stage": "origin_fetch",
            "error": "auth_gateway",
            "status_code": result.status_code,
            "trigger_type": "auth_gateway",
            "trigger_match": gateway,
            "detail": f"auth gateway: {gateway}",
        }
    hit = block_error(
        str(extract.get("title", "")),
        str(extract.get("visible_text", "")),
        result.html or "",
        result.headers,
    )
    if hit is not None:
        return {
            "ok": False,
            "url": url,
            "stage": "origin_fetch",
            "error": "blocked",
            "status_code": result.status_code,
            "trigger_type": hit.trigger_type,
            "trigger_match": hit.marker,
            "detail": f"block interstitial: {hit.marker}",
        }
    return {
        "ok": True,
        "url": url,
        "final_url": result.final_url,
        "extract": extract,
    }
