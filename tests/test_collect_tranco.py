"""Tests for Tranco API integration in the benign collector.

The Tranco API is fully stubbed -- no real network requests are made.
``PLACEHOLDER_KEY`` is a fake credential for local testing only; production
reads the real key from the TRANCO_API_KEY environment variable (GitHub
secret TRANCO_API_KEY), and the key never appears in logs or records.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests

import collect

PLACEHOLDER_KEY = "TRanco_API_KEY_TEMP_REPLACE_ME"
LIST_ID = "LATEST123"
LEGACY_ID = "ABC123"
META_URL = collect.TRANCO_API_LATEST
META_PAYLOAD = {
    "list_id": LIST_ID,
    "available": True,
    "download": f"https://tranco-list.eu/download/{LIST_ID}/1000000",
    "created_on": "2026-09-12T00:00:00.000000",
}


class _FakeResponse:
    def __init__(
        self,
        status: int = 200,
        json_data: object = None,
        json_error: Exception | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status
        self._json_data = json_data
        self._json_error = json_error
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


def _tranco_zip(domains: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "top-1m.csv",
            "".join(f"{i + 1},{d}\n" for i, d in enumerate(domains)),
        )
    return buf.getvalue()


class _TrancoServer:
    """Stub for requests.get: routes metadata/download URLs, records calls."""

    def __init__(self, domains: list[str]) -> None:
        self.calls: list[str] = []
        self.meta_status = 200
        self.meta_payload: object = dict(META_PAYLOAD)
        self.meta_raw: str | None = None
        self.zip_status = 200
        self.domains = domains
        # When set, the list download answers with these raw bytes instead
        # of a ZIP archive (i.e. the plain-CSV shape of the live endpoint).
        self.download_body: bytes | None = None

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append(url)
        if url == META_URL:
            if self.meta_raw is not None:
                return _FakeResponse(
                    status=self.meta_status,
                    json_error=ValueError("No JSON object could be decoded"),
                    content=self.meta_raw.encode(),
                )
            return _FakeResponse(status=self.meta_status, json_data=self.meta_payload)
        if self.download_body is not None:
            return _FakeResponse(status=self.zip_status, content=self.download_body)
        return _FakeResponse(status=self.zip_status, content=_tranco_zip(self.domains))


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _TrancoServer:
    srv = _TrancoServer(["alpha-probe.com", "bravo-probe.com", "charlie-probe.com"])
    monkeypatch.setattr(collect.requests, "get", srv.get)
    monkeypatch.setattr(
        collect,
        "crawl_domain_outcome",
        lambda domain, per_domain, *args: collect.CrawlOutcome(
            domain=domain,
            status="ok",
            homepage_url=f"https://{domain}/",
            http_status=200,
            links=[(f"https://{domain}/deep", 1)],
            error=None,
        ),
    )
    return srv


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> int:
    monkeypatch.setattr(collect, "RAW", tmp_path / "raw")
    monkeypatch.setattr(sys, "argv", ["collect.py", *argv])
    return collect.main()


def _benign_argv(*extra: str) -> list[str]:
    return [
        "--benign",
        "--benign-domains",
        "2",
        "--per-domain",
        "1",
        "--workers",
        "1",
        *extra,
    ]


def _read_rows(tmp_path: Path) -> list[dict[str, object]]:
    files = sorted((tmp_path / "raw").glob("benign-*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_explicit_id_uses_exact_list_and_records_provenance(
    server: _TrancoServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _run(monkeypatch, tmp_path, *_benign_argv("--tranco-id", LEGACY_ID))

    assert rc == 0
    assert server.calls == [collect.TRANCO.format(list_id=LEGACY_ID)]
    rows = _read_rows(tmp_path)
    assert len(rows) == 4  # 2 domains x (homepage + 1 deep link)
    for row in rows:
        assert row["label"] == 0
        assert row["source"] == f"tranco:{LEGACY_ID}"
        assert row["tranco_list_id"] == LEGACY_ID
    homepage_ranks: list[int] = []
    for row in rows:
        url = row.get("url")
        rank = row.get("tranco_rank")
        if isinstance(url, str) and url.endswith(".com/") and isinstance(rank, int):
            homepage_ranks.append(rank)
    assert sorted(homepage_ranks) == [1, 2]


def test_latest_resolves_once_and_records_pinned_id(
    server: _TrancoServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANCO_API_KEY", PLACEHOLDER_KEY)
    rc = _run(monkeypatch, tmp_path, *_benign_argv("--tranco-latest"))

    assert rc == 0
    assert server.calls.count(META_URL) == 1
    assert server.calls == [META_URL, META_PAYLOAD["download"]]
    rows = _read_rows(tmp_path)
    assert rows
    for row in rows:
        assert row["tranco_list_id"] == LIST_ID
        assert row["source"] == f"tranco:{LIST_ID}"
        assert isinstance(row["tranco_resolved_at"], str)
        assert row["tranco_resolved_at"]


def test_resolve_extracts_permanent_id(
    server: _TrancoServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANCO_API_KEY", PLACEHOLDER_KEY)
    ref = collect.resolve_tranco_latest(PLACEHOLDER_KEY)

    assert ref.list_id == LIST_ID
    assert LIST_ID in ref.download_url
    assert ref.created_on == "2026-09-12T00:00:00.000000"
    assert ref.resolved_at
    assert server.calls == [META_URL]


def test_latest_missing_key_fails_without_network(
    server: _TrancoServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("TRANCO_API_KEY", raising=False)
    rc = _run(monkeypatch, tmp_path, *_benign_argv("--tranco-latest"))

    assert rc == 2
    assert server.calls == []
    assert "TRANCO_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    "breakage", ["http500", "badjson", "noid", "unavailable", "baddownload"]
)
def test_latest_failures_are_loud_with_no_fallback(
    server: _TrancoServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    breakage: str,
) -> None:
    monkeypatch.setenv("TRANCO_API_KEY", PLACEHOLDER_KEY)
    if breakage == "http500":
        server.meta_status = 500
    elif breakage == "badjson":
        server.meta_raw = "<html>not json</html>"
    elif breakage == "noid":
        server.meta_payload = {"available": True, "download": "https://x.invalid/z"}
    elif breakage == "unavailable":
        server.meta_payload = {**META_PAYLOAD, "available": False}
    elif breakage == "baddownload":
        server.zip_status = 500

    rc = _run(monkeypatch, tmp_path, *_benign_argv("--tranco-latest"))

    assert rc == 1
    assert list((tmp_path / "raw").glob("*.jsonl")) == []
    assert capsys.readouterr().err
    if breakage != "baddownload":
        assert server.calls == [META_URL]


def test_tranco_modes_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collect, "RAW", tmp_path / "raw")
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect.py", "--benign", "--tranco-id", "X", "--tranco-latest"],
    )
    with pytest.raises(SystemExit) as exc:
        collect.main()
    assert exc.value.code == 2


def test_benign_without_mode_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _run(monkeypatch, tmp_path, "--benign")

    assert rc == 2


def test_api_key_never_leaks(
    server: _TrancoServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TRANCO_API_KEY", PLACEHOLDER_KEY)
    rc = _run(monkeypatch, tmp_path, *_benign_argv("--tranco-latest"))

    assert rc == 0
    out = capsys.readouterr()
    assert PLACEHOLDER_KEY not in out.out
    assert PLACEHOLDER_KEY not in out.err
    for path in (tmp_path / "raw").glob("*.jsonl"):
        assert PLACEHOLDER_KEY not in path.read_text(encoding="utf-8")


CSV_BODY = b"1,google.com\r\n2,example.com\r\n3,wikipedia.org\r\n"


def test_zip_and_csv_produce_same_logical_list() -> None:
    csv_domains = collect._parse_tranco_archive(CSV_BODY, 3)
    zip_domains = collect._parse_tranco_archive(
        _tranco_zip(["google.com", "example.com", "wikipedia.org"]), 3
    )

    assert csv_domains == ["google.com", "example.com", "wikipedia.org"]
    assert zip_domains == csv_domains


def test_latest_csv_download_records_pinned_id(
    server: _TrancoServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.download_body = CSV_BODY
    monkeypatch.setenv("TRANCO_API_KEY", PLACEHOLDER_KEY)
    monkeypatch.setattr(
        collect,
        "crawl_domain_outcome",
        lambda domain, per_domain, *args: collect.CrawlOutcome(
            domain=domain,
            status="ok",
            homepage_url=f"https://{domain}/",
            http_status=200,
            links=[],
            error=None,
        ),
    )
    rc = _run(
        monkeypatch,
        tmp_path,
        "--benign",
        "--tranco-latest",
        "--benign-domains",
        "3",
        "--per-domain",
        "1",
        "--workers",
        "1",
    )

    assert rc == 0
    assert server.calls == [META_URL, META_PAYLOAD["download"]]
    rows = _read_rows(tmp_path)
    assert sorted(str(row["url"]) for row in rows) == [
        "https://example.com/",
        "https://google.com/",
        "https://wikipedia.org/",
    ]
    for row in rows:
        assert row["tranco_list_id"] == LIST_ID
        assert row["source"] == f"tranco:{LIST_ID}"


def test_malformed_csv_rows_are_skipped() -> None:
    body = (
        b"\r\n"
        b"not-a-row\r\n"
        b"oops,example.com\r\n"
        b"1,google.com\r\n"
        b"2,\r\n"
        b",lonely\r\n"
        b"3,wikipedia.org\r\n"
    )

    assert collect._parse_tranco_archive(body, 5) == [
        "google.com",
        "wikipedia.org",
    ]


def test_garbage_csv_raises_tranco_error() -> None:
    with pytest.raises(collect.TrancoError):
        collect._parse_tranco_archive(b"no commas here\nnor here\n", 3)
    with pytest.raises(collect.TrancoError):
        collect._parse_tranco_archive(b"", 3)
