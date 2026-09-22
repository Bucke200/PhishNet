"""Fetcher error-classification helpers (structured outcome contract).

Pure functions: no network, no Playwright. These decide the mechanism string
the serving failure policy keys on, so they must be exact.
"""

from __future__ import annotations

import pytest

from phishnet.fetcher.app import (
    auth_gateway_error,
    block_error,
    classify_connection_error,
    classify_playwright_error,
    trigger_for,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("HTTPSConnectionPool: Failed to resolve 'x.example' ([Errno 11001])", "dns"),
        ("getaddrinfo failed", "dns"),
        ("Name or service not known", "dns"),
        (
            "[WinError 10061] No connection could be made because the target "
            "machine actively refused it",
            "refused",
        ),
        ("Connection reset by peer", "refused"),
    ],
)
def test_classify_connection_error(message: str, expected: str) -> None:
    assert classify_connection_error(message) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("net::ERR_NAME_NOT_RESOLVED", "dns"),
        ("net::ERR_CONNECTION_REFUSED", "refused"),
        ("net::ERR_CONNECTION_RESET", "refused"),
        ("net::ERR_CONNECTION_TIMED_OUT", "origin_timeout"),
        ("Timeout 8000ms exceeded", "origin_timeout"),
        ("net::ERR_CERT_DATE_INVALID", "tls"),
        ("some unrelated failure", None),
    ],
)
def test_classify_playwright_error(message: str, expected: str | None) -> None:
    assert classify_playwright_error(message) == expected


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Suspected Phishing | Cloudflare", "suspected phishing"),
        ("Just a moment...", "just a moment"),
        ("Attention Required! | Cloudflare", "attention required"),
        ("Access Denied", None),  # loose text removed; Akamai uses the reference
        ("Sign in to your account", None),
        ("", None),
    ],
)
def test_block_error(title: str, expected: str | None) -> None:
    hit = block_error(title)
    assert (hit.marker if hit else None) == expected
    if hit is not None:
        assert hit.trigger_type == "text_token"


def test_block_error_detects_akamai_reference() -> None:
    hit = block_error(
        "Access Denied",
        "You don't have permission to access this resource. "
        "Reference #18.abc1234.1234567890.abcdef",
    )
    assert hit is not None
    assert hit.marker == "akamai_reference"
    assert hit.trigger_type == "text_token"


def test_block_error_detects_cf_mitigated_header() -> None:
    hit = block_error(
        "example.com", "hello", "<html>hi</html>", {"cf-mitigated": "challenge"}
    )
    assert hit is not None
    assert hit.marker == "cf-mitigated"
    assert hit.trigger_type == "header"


def test_block_error_ignores_a_generic_access_denied_page() -> None:
    # A plain application 403 without an Akamai reference is not an interstitial.
    assert block_error("Access Denied", "You are not authorized.", "") is None


def test_auth_gateway_error_detects_cloudflare_access() -> None:
    html = "<html><body><a href='/cdn-cgi/access/login'>Sign in</a></body></html>"
    assert auth_gateway_error(html) == "/cdn-cgi/access/"
    assert auth_gateway_error("<html><body>hello</body></html>") is None


def test_block_error_detects_under_attack_body() -> None:
    # Title is the origin domain; the signal is in the body text.
    hit = block_error("example.com", "Checking if the site connection is secure")
    assert hit is not None
    assert hit.marker == "checking if the site connection is secure"
    assert hit.trigger_type == "text_token"


@pytest.mark.parametrize(
    "html,expected",
    [
        ("<script>window.__cf_chl_opt={};</script>", "__cf_chl"),
        (
            "<script src='/cdn-cgi/challenge-platform/h/b/orchestrate'></script>",
            "challenge-platform",
        ),
        (
            "<iframe src='https://geo.captcha-delivery.com/captcha/'></iframe>",
            "captcha-delivery.com",
        ),
        ("<div id='px-captcha'></div>", "px-captcha"),
    ],
)
def test_block_error_detects_challenge_html_tokens(html: str, expected: str) -> None:
    # Empty/origin title: only the raw HTML carries the challenge token.
    hit = block_error("example.com", "", html)
    assert hit is not None
    assert hit.marker == expected
    assert hit.trigger_type == "html_token"


def test_block_error_ignores_a_normal_phishing_page() -> None:
    html = "<form action='https://evil.example/post'><input type=password></form>"
    assert block_error("Sign in to your account", "Enter your password", html) is None


@pytest.mark.parametrize(
    "error,status,expected",
    [
        ("http_403", 403, ("status", "403")),
        ("http_404", 404, ("status", "404")),
        ("dns", None, ("network", "dns")),
        ("origin_timeout", None, ("network", "origin_timeout")),
    ],
)
def test_trigger_for(error: str, status: int | None, expected: tuple[str, str]) -> None:
    assert trigger_for(error, status) == expected
