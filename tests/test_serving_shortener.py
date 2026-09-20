"""C5 — shortener resolver: redirects, hop cap, timeout, loops, failures.

Uses a fake HTTP session (no network). The resolver must never read a body
(``stream=True``, response closed), follow at most ``max_hops``, stop inside
the budget, resolve a shortener-to-shortener chain to the first
non-shortener URL, and return ``unresolved_shortener`` on loops, dead hosts,
and non-redirects still on a shortener.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import requests

from phishnet.serving import shortener
from phishnet.serving.shortener import (
    HttpResponse,
    Resolution,
    is_shortener,
    resolve,
)


class FakeResponse:
    def __init__(self, status_code: int, location: str | None = None) -> None:
        self.status_code = status_code
        self.headers: Mapping[str, str] = {"Location": location} if location else {}
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        mapping: dict[str, FakeResponse],
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self.mapping = mapping
        self.raise_on = raise_on or {}
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        self.calls.append(url)
        self.kwargs.append(kwargs)
        if url in self.raise_on:
            raise self.raise_on[url]
        return self.mapping[url]


def test_is_shortener_matches_covered_hosts_and_subdomains() -> None:
    assert is_shortener("http://bit.ly/abc")
    assert is_shortener("https://www.tinyurl.com/abc")
    assert is_shortener("http://is.gd/xyz")
    assert not is_shortener("https://example.com/abc")
    assert not is_shortener("https://notbit.ly.example.com/abc")


def test_resolves_single_hop_to_final_url() -> None:
    session = FakeSession(
        {"http://bit.ly/abc": FakeResponse(301, "https://safe.example/landing")}
    )
    result = resolve("http://bit.ly/abc", session=session)
    assert result == Resolution("https://safe.example/landing", 1, "")
    assert session.kwargs[0]["stream"] is True
    assert session.kwargs[0]["allow_redirects"] is False


def test_resolves_shortener_to_shortener_chain() -> None:
    session = FakeSession(
        {
            "http://bit.ly/a": FakeResponse(302, "http://tinyurl.com/b"),
            "http://tinyurl.com/b": FakeResponse(301, "https://final.example/x"),
        }
    )
    result = resolve("http://bit.ly/a", session=session)
    assert result.final_url == "https://final.example/x"
    assert result.hops == 2


def test_hop_cap_yields_unresolved() -> None:
    # Six shortener hops with no exit, max_hops=5.
    mapping = {
        f"http://h{i}.bit.ly/x": FakeResponse(301, f"http://h{i + 1}.bit.ly/x")
        for i in range(7)
    }
    session = FakeSession(mapping)
    result = resolve("http://h0.bit.ly/x", max_hops=5, session=session)
    assert result.resolved is False
    assert result.reason == shortener.UNRESOLVED
    assert len(session.calls) == 5


def test_loop_yields_unresolved() -> None:
    session = FakeSession(
        {
            "http://bit.ly/a": FakeResponse(301, "http://tinyurl.com/b"),
            "http://tinyurl.com/b": FakeResponse(301, "http://bit.ly/a"),
        }
    )
    result = resolve("http://bit.ly/a", session=session)
    assert result.resolved is False


def test_non_redirect_on_shortener_is_unresolved() -> None:
    session = FakeSession({"http://bit.ly/a": FakeResponse(200)})
    result = resolve("http://bit.ly/a", session=session)
    assert result.resolved is False
    assert result.reason == shortener.UNRESOLVED


def test_connection_error_is_unresolved() -> None:
    session = FakeSession(
        {}, raise_on={"http://bit.ly/a": requests.ConnectionError("boom")}
    )
    result = resolve("http://bit.ly/a", session=session)
    assert result.resolved is False


def test_zero_budget_is_unresolved_without_calls() -> None:
    session = FakeSession({"http://bit.ly/a": FakeResponse(200)})
    result = resolve("http://bit.ly/a", budget=0.0, session=session)
    assert result.resolved is False
    assert session.calls == []


def test_non_shortener_input_resolves_to_itself() -> None:
    session = FakeSession({"https://example.com/a": FakeResponse(200)})
    result = resolve("https://example.com/a", session=session)
    assert result.final_url == "https://example.com/a"
