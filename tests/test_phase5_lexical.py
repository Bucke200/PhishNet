"""Phase 5 lexical mechanics tests (no attack set, no scores — commit 1).

Asserts the §7.1 transform contract: pure, deterministic, parseable, one
transform per URL, with the covered/uncovered shortener split and the
not-applicable accounting. Scoring the registered set happens after commit 2.
"""

import urllib.parse

from phishnet.adversarial.lexical import (
    COVERED_SHORTENERS,
    HOMOGLYPH_MAX_SUBS,
    REDIRECT_HOSTS,
    UNCOVERED_SHORTENERS,
    homoglyph_domain,
    is_applicable,
    is_covered_shortener,
    open_redirect_wrap,
    shortener_wrap,
    to_ascii_form,
)


def _parseable(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
        return bool(parts.scheme and parts.netloc)
    except Exception:
        return False


def test_homoglyph_is_deterministic_bounded_and_parseable() -> None:
    url = "http://paypal.example-login.com/path?q=1"
    first = homoglyph_domain(url)
    assert first == homoglyph_domain(url)
    assert _parseable(first)
    assert first != url
    assert len(first) == len(url)  # substitution preserves length
    changed = sum(a != b for a, b in zip(url.lower(), first.lower(), strict=True))
    assert changed <= HOMOGLYPH_MAX_SUBS
    # Path and query held fixed; only the host moves.
    assert urllib.parse.urlsplit(first).path == "/path"


def test_homoglyph_unicode_and_ascii_forms_separate() -> None:
    url = "http://paypal.example.com/login"
    uni = homoglyph_domain(url)
    ascii_form = to_ascii_form(uni)
    assert ascii_form is not None
    assert "xn--" in ascii_form
    assert ascii_form != uni
    assert _parseable(ascii_form)


def test_homoglyph_not_applicable_on_ip_host() -> None:
    assert not is_applicable("http://192.168.1.1/admin", "homoglyph")
    assert is_applicable("http://paypal.example.com/", "homoglyph")


def test_shortener_cover_split() -> None:
    assert len(COVERED_SHORTENERS) == 5
    assert len(UNCOVERED_SHORTENERS) == 5
    assert all(is_covered_shortener(h) for h in COVERED_SHORTENERS)
    assert not any(is_covered_shortener(h) for h in UNCOVERED_SHORTENERS)
    out = shortener_wrap("http://evil.example/x", "bit.ly")
    assert out.startswith("https://bit.ly/r/") and len(out.rsplit("/", 1)[1]) == 7
    assert _parseable(out)
    assert shortener_wrap("http://evil.example/x", "bit.ly") == out


def test_open_redirect_fixed_template() -> None:
    assert len(REDIRECT_HOSTS) == 3
    out = open_redirect_wrap("http://evil.example/x?a=b", REDIRECT_HOSTS[0])
    parts = urllib.parse.urlsplit(out)
    assert parts.hostname == REDIRECT_HOSTS[0]
    assert parts.path == "/redirect"
    assert "url=" in parts.query
    assert _parseable(out)
