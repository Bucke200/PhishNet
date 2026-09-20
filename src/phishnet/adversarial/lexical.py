"""Phase 5 lexical-evasion transforms (prereg §7.1, frozen in commit 1).

Pure functions `url -> url`, one transform per URL, no stacking. Tier-1-only
arm: URL-string attacks never reach the LLM. The attack *set* (which test URLs
get transformed) is fixed in commit 2; this module holds only the mechanics.
"""

from __future__ import annotations

import ipaddress
import random
import string
import urllib.parse

import tldextract

HOMOGLYPH_SEED = 7
HOMOGLYPH_MAX_SUBS = 2

# §7.1 confusable table: ASCII -> Cyrillic lookalike.
CONFUSABLES: dict[str, str] = {
    "a": "а",  # U+0430
    "e": "е",  # U+0435
    "o": "о",  # U+043E
    "p": "р",  # U+0440
    "c": "с",  # U+0441
    "i": "і",  # U+0456
}

# §7.1 shortener hosts: 5 covered by tier-1's `is_shortened` list, 5 synthetic
# uncovered (RFC 2606 `.example`, so no real service is involved).
COVERED_SHORTENERS: tuple[str, ...] = (
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
)
UNCOVERED_SHORTENERS: tuple[str, ...] = (
    "short.example",
    "go.example",
    "s.example",
    "tiny.example",
    "link.example",
)

# §7.1 redirect hosts (fixed template below; scored offline, never fetched).
REDIRECT_HOSTS: tuple[str, ...] = (
    "portal.example",
    "login.example",
    "news.example",
)


def _hostname(url: str) -> str:
    parts = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    return parts.hostname or ""


def is_applicable(url: str, transform: str) -> bool:
    """False where a transform cannot apply (counted, never silently dropped)."""
    if transform == "homoglyph":
        host = _hostname(url)
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            pass
        domain = tldextract.extract(host).domain
        return any(ch.lower() in CONFUSABLES for ch in domain)
    return True


def homoglyph_domain(url: str, seed: int = HOMOGLYPH_SEED) -> str:
    """Substitute at most 2 confusables in the registrable domain label."""
    rng = random.Random(f"{seed}|{url}")
    parts = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    host = parts.hostname or ""
    ext = tldextract.extract(host)
    chars = list(ext.domain)
    candidates = [i for i, ch in enumerate(chars) if ch.lower() in CONFUSABLES]
    rng.shuffle(candidates)
    for i in candidates[:HOMOGLYPH_MAX_SUBS]:
        chars[i] = CONFUSABLES[chars[i].lower()]
    new_domain = "".join(chars)
    new_host = ".".join(p for p in (ext.subdomain, new_domain, ext.suffix) if p)
    netloc = new_host
    if parts.port:
        netloc += f":{parts.port}"
    rebuilt = urllib.parse.urlunsplit(
        (parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment)
    )
    return rebuilt if "://" in url else rebuilt.split("://", 1)[1]


def to_ascii_form(url: str) -> str | None:
    """IDNA-encode the host labels (`xn--` form — the serving-side primary)."""
    parts = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    try:
        ascii_host = ".".join(
            label.encode("idna").decode("ascii")
            for label in (parts.hostname or "").split(".")
        )
    except UnicodeError:
        return None
    netloc = ascii_host
    if parts.port:
        netloc += f":{parts.port}"
    return urllib.parse.urlunsplit(
        (parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment)
    )


def shortener_wrap(url: str, shortener_host: str, seed: int = HOMOGLYPH_SEED) -> str:
    """Replace the URL with a synthetic shortener URL (tier 1 sees only it)."""
    rng = random.Random(f"{seed}|{url}|{shortener_host}")
    token = "".join(rng.choices(string.ascii_letters + string.digits, k=7))
    return f"https://{shortener_host}/r/{token}"


def open_redirect_wrap(url: str, benign_host: str) -> str:
    """Wrap the pct-encoded URL in the fixed redirect template."""
    return f"https://{benign_host}/redirect?url={urllib.parse.quote(url, safe='')}"


def is_covered_shortener(host: str) -> bool:
    """True iff tier-1's `is_shortened` list covers the host."""
    return host.lower() in COVERED_SHORTENERS
