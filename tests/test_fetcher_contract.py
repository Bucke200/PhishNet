"""Fetcher service contract (Phase 6 decision 2).

The network/Playwright boundary is monkeypatched: the endpoint must return
the canonical extract and never the raw HTML.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishnet.fetcher import app as fetcher

HTML = """
<html><head><title>Sign in</title>
<meta name="description" content="Account login"></head>
<body><form action="https://evil.example/post">
<input name="user" type="email">
<input name="pass" type="password">
</form></body></html>
"""


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(fetcher, "fetch_html", lambda url, timeout=15.0: (HTML, url))
    return TestClient(fetcher.app)


def test_fetch_returns_extract_not_html(client: TestClient) -> None:
    body = client.post("/fetch", json={"url": "https://evil.example/login"}).json()
    assert body["final_url"] == "https://evil.example/login"
    extract = body["extract"]
    assert "raw_html" not in extract
    assert extract["title"] == "Sign in"
    assert extract["meta_description"] == "Account login"
    assert extract["forms"][0]["action_host"] == "evil.example"
    assert any(i["type"] == "password" for i in extract["forms"][0]["inputs"])


def test_fetch_rejects_invalid_url(client: TestClient) -> None:
    assert client.post("/fetch", json={"url": "nope"}).status_code == 422
