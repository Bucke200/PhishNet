"""Tests for the benign crawl behavior (depth, scheme, failures, provenance).

Network is fully stubbed: a fake ``requests.get`` answers homepage, page, and
sitemap URLs from an in-memory table, and ``_allowed`` is stubbed per test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

import collect


class _FakePage:
    def __init__(
        self,
        url: str,
        text: str = "",
        status: int = 200,
        content_type: str = "text/html",
    ) -> None:
        self.url = url
        self.text = text
        self.status_code = status
        self.headers = {"content-type": content_type}
        self.content = text.encode()


class _FakeNet:
    """Route exact URLs to pages (or exceptions); record every call."""

    def __init__(self) -> None:
        self.routes: dict[str, _FakePage | Exception] = {}
        self.calls: list[str] = []

    def add(self, url: str, page: _FakePage | Exception) -> None:
        self.routes[url] = page

    def get(self, url: str, **kwargs: object) -> _FakePage:
        self.calls.append(url)
        route = self.routes.get(url)
        if route is None:
            raise requests.ConnectionError(f"no route for {url}")
        if isinstance(route, Exception):
            raise route
        return route


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> _FakeNet:
    fake = _FakeNet()
    monkeypatch.setattr(collect.requests, "get", fake.get)
    monkeypatch.setattr(collect, "_allowed", lambda domain, *args, **kwargs: True)
    return fake


def _rows_for(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    domains: list[str],
    list_id: str = "TESTLIST",
) -> tuple[list[dict[str, object]], str]:
    rows = collect.collect_benign_from_domains(
        domains, list_id, per_domain=8, workers=1, today="2026-09-12"
    )
    return rows, list_id


def test_homepage_records_final_url_after_redirect(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://seed-probe.com/",
        _FakePage(
            "https://www.seed-probe.com/home",
            '<a href="/news">news</a>',
        ),
    )
    rows, _ = _rows_for(monkeypatch, tmp_path, ["seed-probe.com"])

    home = [r for r in rows if r["link_depth"] == 0]
    assert len(home) == 1
    assert home[0]["url"] == "https://www.seed-probe.com/home"
    assert home[0]["url"] != "https://seed-probe.com/"
    deep = [r for r in rows if r["link_depth"] != 0]
    assert deep and all(
        str(u).startswith("https://www.seed-probe.com/")
        for u in [r["url"] for r in deep]
    )


def test_https_falls_back_to_http(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add("https://plain-probe.com/", requests.ConnectionError("refused"))
    net.add(
        "http://plain-probe.com/",
        _FakePage("http://plain-probe.com/", '<a href="/about">a</a>'),
    )
    rows, _ = _rows_for(monkeypatch, tmp_path, ["plain-probe.com"])

    home = [r for r in rows if r["link_depth"] == 0]
    assert len(home) == 1
    assert str(home[0]["url"]).startswith("http://")


def test_fetch_failure_yields_no_fabricated_rows(
    net: _FakeNet,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    net.add("https://dead-probe.com/", requests.ConnectionError("dns"))
    net.add("http://dead-probe.com/", requests.ConnectionError("dns"))
    rows, _ = _rows_for(monkeypatch, tmp_path, ["dead-probe.com"])

    assert rows == []
    assert "failed=" in capsys.readouterr().err


def test_robots_denied_yields_no_rows(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(collect, "_allowed", lambda domain, **kwargs: False)
    rows, _ = _rows_for(monkeypatch, tmp_path, ["blocked-probe.com"])

    assert rows == []
    assert net.calls == []


def test_robots_fetch_uses_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> _FakePage:
        seen.update(kwargs)
        raise requests.ConnectionError("unreachable")

    monkeypatch.setattr(collect.requests, "get", fake_get)
    assert collect._allowed("anything-probe.com", timeout=7) is True
    assert seen.get("timeout") == 7


def test_robots_disallow_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "User-agent: *\nDisallow: /\n"

    def fake_get(url: str, **kwargs: object) -> _FakePage:
        return _FakePage(url, body, content_type="text/plain")

    monkeypatch.setattr(collect.requests, "get", fake_get)
    assert collect._allowed("strict-probe.com") is False


def test_non_html_homepage_yields_no_rows(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://api-probe.com/",
        _FakePage(
            "https://api-probe.com/", '{"a": 1}', content_type="application/json"
        ),
    )
    rows, _ = _rows_for(monkeypatch, tmp_path, ["api-probe.com"])

    assert rows == []


def test_subdomain_links_accepted_external_rejected(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://multi-probe.com/",
        _FakePage(
            "https://multi-probe.com/",
            '<a href="https://blog.multi-probe.com/post">b</a>'
            '<a href="https://external-probe.com/post">e</a>'
            '<a href="https://multi-probe.com/">bare</a>'
            '<a href="https://multi-probe.com/real">real</a>',
        ),
    )
    rows, _ = _rows_for(monkeypatch, tmp_path, ["multi-probe.com"])

    urls = sorted(str(r["url"]) for r in rows if r["link_depth"] != 0)
    assert "https://blog.multi-probe.com/post" in urls
    assert "https://multi-probe.com/real" in urls
    assert not any("external-probe.com" in u for u in urls)
    assert "https://multi-probe.com/" not in urls


def test_depth_two_follow_harvests_second_level(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://deep-probe.com/",
        _FakePage("https://deep-probe.com/", '<a href="/section">s</a>'),
    )
    net.add(
        "https://deep-probe.com/section",
        _FakePage(
            "https://deep-probe.com/section",
            '<a href="/section/article-1">a1</a>',
        ),
    )
    rows = collect.collect_benign_from_domains(
        ["deep-probe.com"], "T", per_domain=8, workers=1, today="t"
    )

    depths = {(str(r["url"]), r["link_depth"]) for r in rows if r["link_depth"] != 0}
    assert ("https://deep-probe.com/section", 1) in depths
    assert ("https://deep-probe.com/section/article-1", 2) in depths


def test_sitemap_fallback_covers_linkless_homepage(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://js-probe.com/", _FakePage("https://js-probe.com/", "<div>app</div>")
    )
    net.add(
        "https://js-probe.com/sitemap.xml",
        _FakePage(
            "https://js-probe.com/sitemap.xml",
            "<urlset><url><loc>https://js-probe.com/guide</loc></url>"
            "<url><loc>https://other-probe.com/x</loc></url></urlset>",
            content_type="application/xml",
        ),
    )
    rows, _ = _rows_for(monkeypatch, tmp_path, ["js-probe.com"])

    urls = [str(r["url"]) for r in rows if r["link_depth"] != 0]
    assert "https://js-probe.com/guide" in urls
    assert not any("other-probe.com" in u for u in urls)


def test_rows_carry_provenance(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://prov-probe.com/",
        _FakePage("https://prov-probe.com/", '<a href="/p">p</a>'),
    )
    rows, list_id = _rows_for(monkeypatch, tmp_path, ["prov-probe.com"])

    assert rows
    for row in rows:
        assert row["label"] == 0
        assert row["source"] == f"tranco:{list_id}"
        assert row["tranco_list_id"] == list_id
        assert row["time_basis"] == "crawled"
        assert row["seed_domain"] == "prov-probe.com"
        assert row["crawl_status"] == "ok"
        assert row["http_status"] == 200
        assert row["link_depth"] in (0, 1, 2)
        assert isinstance(row["tranco_rank"], int)


def test_crawl_is_deterministic(net: _FakeNet) -> None:
    net.add(
        "https://det-probe.com/",
        _FakePage(
            "https://det-probe.com/",
            "".join(f'<a href="/p{i}">x</a>' for i in range(10)),
        ),
    )
    first = collect.crawl_domain_outcome("det-probe.com", 5)
    second = collect.crawl_domain_outcome("det-probe.com", 5)

    assert first.links == second.links
    assert first.homepage_url == second.homepage_url


def test_wrapper_returns_deep_link_urls(net: _FakeNet) -> None:
    net.add(
        "https://wrap-probe.com/",
        _FakePage("https://wrap-probe.com/", '<a href="/w">w</a>'),
    )
    links = collect.crawl_domain("wrap-probe.com", 8)

    assert links == ["https://wrap-probe.com/w"]


def test_end_to_end_jsonl_schema(
    net: _FakeNet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    net.add(
        "https://e2e-probe.com/",
        _FakePage("https://e2e-probe.com/", '<a href="/w">w</a>'),
    )
    monkeypatch.setattr(collect, "RAW", tmp_path / "raw")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect.py",
            "--benign",
            "--tranco-id",
            "E2E",
            "--benign-domains",
            "1",
            "--per-domain",
            "2",
            "--workers",
            "1",
        ],
    )
    monkeypatch.setattr(collect, "fetch_tranco", lambda list_id, n: ["e2e-probe.com"])
    assert collect.main() == 0

    files = sorted((tmp_path / "raw").glob("benign-*.jsonl"))
    assert len(files) == 1
    rows = [
        json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2  # homepage + 1 deep link
    assert {str(r["url"]) for r in rows} == {
        "https://e2e-probe.com/",
        "https://e2e-probe.com/w",
    }
