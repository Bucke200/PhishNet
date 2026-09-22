"""Fetcher service contract (Phase 6 decision 2; structured outcome 2026-09-22).

The network/Playwright boundary is monkeypatched: the endpoint must return the
canonical extract and never the raw HTML, and a target-side failure must be a
200 with a structured `ok: false` body — never a 502 (which would conflate a
dead origin with a broken fetcher).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishnet.fetcher import app as fetcher
from phishnet.fetcher.app import FetchResult

HTML = """
<html><head><title>Sign in</title>
<meta name="description" content="Account login"></head>
<body><form action="https://evil.example/post">
<input name="user" type="email">
<input name="pass" type="password">
</form></body></html>
"""


def _client(monkeypatch: pytest.MonkeyPatch, result: FetchResult) -> TestClient:
    monkeypatch.setattr(fetcher, "fetch_page", lambda url: result)
    return TestClient(fetcher.app)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(monkeypatch, FetchResult(True, "https://evil.example/login", HTML))


def test_fetch_returns_extract_not_html(client: TestClient) -> None:
    body = client.post("/fetch", json={"url": "https://evil.example/login"}).json()
    assert body["ok"] is True
    assert body["final_url"] == "https://evil.example/login"
    extract = body["extract"]
    assert "raw_html" not in extract
    assert extract["title"] == "Sign in"
    assert extract["meta_description"] == "Account login"
    assert extract["forms"][0]["action_host"] == "evil.example"
    assert any(i["type"] == "password" for i in extract["forms"][0]["inputs"])


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "render" in body


def test_fetch_rejects_invalid_url(client: TestClient) -> None:
    assert client.post("/fetch", json={"url": "nope"}).status_code == 422


@pytest.mark.parametrize(
    "error,status",
    [("http_403", 403), ("http_404", 404), ("dns", None), ("origin_timeout", None)],
)
def test_target_failure_is_200_with_mechanism(
    monkeypatch: pytest.MonkeyPatch, error: str, status: int | None
) -> None:
    client = _client(
        monkeypatch,
        FetchResult(False, error=error, status_code=status, detail="x"),
    )
    response = client.post("/fetch", json={"url": "https://dead.example/"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["stage"] == "origin_fetch"
    assert body["error"] == error
    assert body["status_code"] == status
    assert body["trigger_type"] in ("status", "network")
    assert body["trigger_match"]


def test_block_interstitial_maps_to_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<html><head><title>Suspected Phishing | Cloudflare</title></head></html>"
    client = _client(
        monkeypatch, FetchResult(True, "https://x.example/", html, status_code=200)
    )
    body = client.post("/fetch", json={"url": "https://x.example/"}).json()
    assert body["ok"] is False
    assert body["error"] == "blocked"
    assert "suspected phishing" in body["detail"]
    assert body["trigger_type"] == "text_token"
    assert body["trigger_match"] == "suspected phishing"


def test_under_attack_200_challenge_maps_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Title is the origin domain; the challenge is only visible in the body.
    html = (
        "<html><head><title>bank.example</title></head><body>"
        "<h1>Checking if the site connection is secure</h1>"
        "<script>window.__cf_chl_opt={};</script></body></html>"
    )
    client = _client(
        monkeypatch, FetchResult(True, "https://bank.example/", html, status_code=200)
    )
    body = client.post("/fetch", json={"url": "https://bank.example/"}).json()
    assert body["ok"] is False
    assert body["error"] == "blocked"
    assert body["trigger_match"] == "checking if the site connection is secure"


def test_cf_mitigated_header_maps_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = "<html><head><title>example.com</title></head><body>hello</body></html>"
    client = _client(
        monkeypatch,
        FetchResult(
            True,
            "https://x.example/",
            html,
            status_code=200,
            headers={"cf-mitigated": "challenge"},
        ),
    )
    body = client.post("/fetch", json={"url": "https://x.example/"}).json()
    assert body["ok"] is False
    assert body["error"] == "blocked"
    assert body["trigger_type"] == "header"
    assert body["trigger_match"] == "cf-mitigated"


def test_auth_gateway_maps_to_auth_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<html><head><title>portal.example</title></head><body>"
        "<a href='/cdn-cgi/access/login'>Sign in</a></body></html>"
    )
    client = _client(
        monkeypatch, FetchResult(True, "https://portal.example/", html, status_code=200)
    )
    body = client.post("/fetch", json={"url": "https://portal.example/"}).json()
    assert body["ok"] is False
    assert body["error"] == "auth_gateway"
    assert body["trigger_type"] == "auth_gateway"
    assert body["trigger_match"] == "/cdn-cgi/access/"
