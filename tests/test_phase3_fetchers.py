"""Step 3 fetcher tests: offline (stubbed network, always run) + live smoke.

Live tests hit RDAP/crt.sh for real and run only with
``PHISHNET_LIVE_NETWORK=1`` (same convention as
test_cc_cdx_query_form.py), so CI stays green with no network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import requests

from phishnet.enrichment import batch, ct, rdap

pytestmark = pytest.mark.network


class _Resp:
    def __init__(
        self,
        payload: Any = None,
        status: int = 200,
        error: Exception | None = None,
    ):
        self._payload = payload
        self.status_code = status
        self._error = error

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error
        if self.status_code >= 400:
            raise requests.HTTPError(f"http {self.status_code}")


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)


# --- RDAP parsing (no network at all) ---


def test_parse_rdap_creation_picks_registration() -> None:
    payload = {
        "events": [
            {"eventAction": "last changed", "eventDate": "2024-01-01T00:00:00Z"},
            {"eventAction": "registration", "eventDate": "2010-05-05T00:00:00Z"},
        ]
    }
    assert rdap.parse_rdap_creation(payload) == "2010-05-05T00:00:00Z"
    assert rdap.parse_rdap_creation({"events": []}) is None
    assert rdap.parse_rdap_creation({}) is None


def test_rdap_servers_longest_suffix_first() -> None:
    bootstrap = {"uk": ["https://rdap.uk/"], "co.uk": ["https://rdap-couk/"]}
    assert rdap.rdap_servers_for("a.b.co.uk", bootstrap) == ["https://rdap-couk/"]
    assert rdap.rdap_servers_for("example.uk", bootstrap) == ["https://rdap.uk/"]
    assert rdap.rdap_servers_for("example.com", bootstrap) == []


def test_parse_whois_creation_phrasings() -> None:
    assert (
        rdap.parse_whois_creation(
            "Domain Name: X\nCreation Date: 2011-02-03T04:05:06Z\n"
        )
        == "2011-02-03T04:05:06Z"
    )
    assert (
        rdap.parse_whois_creation("Created: 2011-02-03\nRegistrar: Y\n") == "2011-02-03"
    )
    assert rdap.parse_whois_creation("Registered on: 03-Feb-2011\n") == "03-Feb-2011"
    assert rdap.parse_whois_creation("No match here\n") is None


# --- RDAP lookup (stubbed requests) ---


def test_rdap_lookup_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    payload = {
        "events": [{"eventAction": "registration", "eventDate": "2012-06-06T00:00:00Z"}]
    }
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(payload))
    out = rdap.rdap_lookup("example.com", {"com": ["https://rdap.verisign/"]})
    assert out == {
        "creation_date": "2012-06-06T00:00:00Z",
        "source": "rdap",
        "server": "https://rdap.verisign/",
        "error": None,
    }


def test_rdap_lookup_404_is_definitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(*a: Any, **k: Any) -> _Resp:
        calls.append(1)
        return _Resp(None, status=404)

    monkeypatch.setattr(requests, "get", fake_get)
    out = rdap.rdap_lookup("example.com", {"com": ["https://rdap.verisign/"]})
    assert out["error"] == "rdap-404" and out["creation_date"] is None
    assert len(calls) == 1  # no retry on a definitive answer


def test_rdap_lookup_no_service() -> None:
    out = rdap.rdap_lookup("example.com", {})
    assert out["error"] == "no-rdap-service"


def test_rdap_lookup_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    payload = {
        "events": [{"eventAction": "registration", "eventDate": "2012-06-06T00:00:00Z"}]
    }
    seq = [requests.ConnectionError("down"), _Resp(payload)]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _raise_or_resp(seq))
    out = rdap.rdap_lookup("example.com", {"com": ["https://s/"]}, retries=1)
    assert out["creation_date"] == "2012-06-06T00:00:00Z"


def _raise_or_resp(seq: list[Any]) -> _Resp:
    item = seq.pop(0)
    if isinstance(item, Exception):
        raise item
    resp: _Resp = item
    return resp


# --- WHOIS fallback (stubbed socket) ---


class _FakeSock:
    def __init__(self, text: str):
        self._data = text.encode()
        self.sent: list[bytes] = []

    def __enter__(self) -> _FakeSock:
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def settimeout(self, *a: Any) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, n: int) -> bytes:
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


def test_whois_lookup_referral_and_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_conn(addr: tuple[str, int], timeout: Any = None) -> _FakeSock:
        if addr[0] == "whois.iana.org":
            return _FakeSock("refer: x\nwhois: whois.example-tld\n")
        assert addr[0] == "whois.example-tld"
        return _FakeSock("Domain: example\nCreated: 2015-07-07\n")

    monkeypatch.setattr(rdap.socket, "create_connection", fake_conn)
    out = rdap.whois_lookup("example.example-tld")
    assert out["creation_date"] == "2015-07-07"
    assert out["source"] == "whois" and out["error"] is None


def test_fetch_age_ip_literal_skips() -> None:
    for ip in ("192.168.1.1", "2606:4700:4700::1111"):
        out = rdap.fetch_age(ip, {"com": ["https://s/"]})
        assert out["error"] == "ip-literal" and out["creation_date"] is None


def test_fetch_age_falls_back_to_whois(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rdap,
        "rdap_lookup",
        lambda *a, **k: {
            "creation_date": None,
            "source": None,
            "server": "s",
            "error": "rdap-failed",
        },
    )
    monkeypatch.setattr(
        rdap,
        "whois_lookup",
        lambda *a, **k: {
            "creation_date": "2015-01-01",
            "source": "whois",
            "server": "w",
            "error": None,
        },
    )
    out = rdap.fetch_age("example.com", {"com": ["s"]})
    assert out["source"] == "whois" and out["rdap_error"] == "rdap-failed"


# --- CT lookup (stubbed requests) ---


def test_crtsh_success_and_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "id": 1,
            "entry_timestamp": "2021-01-01T00:00:00",
            "not_before": "2021-01-01T00:00:00",
            "not_after": "2022-01-01T00:00:00",
            "common_name": "example.com",
            "extra": "dropped",
        },
    ]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(rows))
    out = ct.crtsh_lookup("example.com")
    assert out["provider"] == "crt.sh-json" and out["error"] is None
    assert out["certs"] == [
        {
            "id": 1,
            "entry_timestamp": "2021-01-01T00:00:00",
            "not_before": "2021-01-01T00:00:00",
            "not_after": "2022-01-01T00:00:00",
            "common_name": "example.com",
        }
    ]


def test_crtsh_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    seq: list[Any] = [_Resp(None, status=429), _Resp([])]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _raise_or_resp(seq))
    out = ct.crtsh_lookup("example.com", retries=1)
    assert out["certs"] == [] and out["error"] is None


def test_crtsh_failure_and_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )
    out = ct.crtsh_lookup("example.com", retries=0)
    assert out["certs"] is None and "ConnectionError" in str(out["error"])
    chained = ct.fetch_ct("example.com", allow_postgres=False)
    assert chained["certs"] is None and chained["provider"] is None


def test_postgres_needs_dsn_or_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRT_PG_DSN", raising=False)
    out = ct.postgres_lookup("example.com")
    assert out["error"] == "no-pg-dsn" and out["certs"] is None
    monkeypatch.setenv("CRT_PG_DSN", "dbname=crtsh host=example")
    out = ct.postgres_lookup("example.com")
    # Either the optional driver is missing (clean error) or the bogus DSN
    # fails to connect (driver/network error) — never a silent success.
    assert out["certs"] is None and out["error"] is not None


# --- Batch runner (injected fetchers, real store) ---


def test_enrich_key_hosted_never_queries() -> None:
    def boom(*a: Any, **k: Any) -> dict[str, Any]:
        raise AssertionError("hosted keys must not be queried")

    row = batch.enrich_key("tenant.core.windows.net", {}, ct_fetch=boom, age_fetch=boom)
    assert row == {
        "cache_key": "tenant.core.windows.net",
        "hosted": True,
        "rdap": None,
        "ct": None,
    }


def test_enrich_keys_resume_and_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phishnet.enrichment.batch as B
    from phishnet.enrichment.store import load_pinned_run

    snap = tmp_path / "batch.jsonl"
    calls: list[str] = []

    def age_fetch(key: str, *a: Any, **k: Any) -> dict[str, Any]:
        calls.append(key)
        return {"creation_date": "2020-01-01T00:00:00+00:00", "source": "rdap"}

    def ct_fetch(key: str, *a: Any, **k: Any) -> dict[str, Any]:
        return {"certs": [], "provider": "crt.sh-json"}

    monkeypatch.setattr(B.rdap, "fetch_age", age_fetch)
    monkeypatch.setattr(B.ct, "fetch_ct", ct_fetch)
    urls = [
        "https://a.example/x",
        "https://b.example/y",
        "https://t.core.windows.net/z",
    ]
    sidecar = B.enrich_keys(urls, snap, "run-1", bootstrap={}, progress_every=0)
    # a.example + b.example share one registrable-domain key; the hosted
    # tenant resolves na without querying.
    assert sidecar["n_records"] == 2
    assert calls == ["example"]  # hosted never fetched
    assert (tmp_path / "batch.jsonl.run-run-1.meta.json").exists()
    # Resume is a no-op for stored keys.
    before = load_pinned_run(snap, "run-1")
    calls.clear()
    B.enrich_keys(urls, snap, "run-1", bootstrap={}, progress_every=0)
    assert calls == []
    assert load_pinned_run(snap, "run-1") == before


# --- Feature table (Step 4, offline fixtures) ---


def _feature_snapshot(tmp_path: Path) -> Path:
    from phishnet.enrichment.store import append_records, seal_run

    snap = tmp_path / "features.jsonl"
    append_records(
        snap,
        "run-1",
        [
            {
                "cache_key": "example.com",
                "rdap": {
                    "creation_date": "2020-06-01T00:00:00+00:00",
                    "source": "rdap",
                },
                "ct": {
                    "certs": [
                        {"entry_timestamp": "2021-03-01T00:00:00+00:00"},
                        {"entry_timestamp": "2026-10-01T00:00:00+00:00"},
                    ],
                    "provider": "crt.sh-json",
                },
                "enriched_at": "2026-09-16T00:00:00+00:00",
            }
        ],
    )
    seal_run(snap, "run-1")
    return snap


def _lexical_columns() -> list[str]:
    import pickle

    cols: list[str] = pickle.loads(
        Path("src/phishnet/urlset_ml_assets/feature_columns.pkl").read_bytes()
    )
    return cols


def test_feature_table_values_and_vocab(tmp_path: Path) -> None:
    from phishnet.enrichment.features import ENRICHED_COLUMNS, build_feature_table

    snap = _feature_snapshot(tmp_path)
    rows = [
        {
            "url": "https://mail.example.com/inbox",
            "label": 1,
            "first_seen": "2026-09-15T00:00:00+00:00",
            "survival_stratum": "short",
        },
        {"url": "https://absent.example/x", "label": 0},  # unknown key
        {
            "url": "https://t.core.windows.net/app",  # hosted: na
            "label": 1,
            "first_seen": "2026-09-15T00:00:00+00:00",
        },
    ]
    frame, vocab, manifest = build_feature_table(
        rows,
        snap,
        {"rule": "pinned-run", "run_id": "run-1"},
        _lexical_columns(),
        canonicalize=True,
    )
    assert vocab == _lexical_columns() + ENRICHED_COLUMNS
    assert list(frame.columns) == vocab
    assert len(frame) == 3
    known = frame.iloc[0]
    assert known["age_known"] == 1.0 and known["ct_known"] == 1.0
    assert known["ct_cert_count_pre"] == 1.0  # post-cutoff cert excluded
    assert known["domain_age_days"] > 1900.0
    unknown = frame.iloc[1]
    assert [unknown[c] for c in ENRICHED_COLUMNS] == [0.0] * 5
    hosted = frame.iloc[2]
    assert [hosted[c] for c in ENRICHED_COLUMNS] == [0.0] * 5
    # na flags are analysis-only: they must not appear in X.
    assert "age_na" not in frame.columns and "ct_na" not in frame.columns
    assert manifest["canonicalize"] is True
    assert manifest["join"]["selection_rule"] == "pinned-run"


def test_feature_table_canonicalize_converges(tmp_path: Path) -> None:
    import numpy as np

    from phishnet.enrichment.features import build_feature_table

    snap = _feature_snapshot(tmp_path)
    mk = lambda scheme: [  # noqa: E731
        {
            "url": f"{scheme}://mail.example.com/inbox",
            "label": 1,
            "first_seen": "2026-09-15T00:00:00+00:00",
        }
    ]
    sel = {"rule": "pinned-run", "run_id": "run-1"}
    http, _, _ = build_feature_table(
        mk("http"), snap, sel, _lexical_columns(), canonicalize=True
    )
    https, _, _ = build_feature_table(
        mk("https"), snap, sel, _lexical_columns(), canonicalize=True
    )
    np.testing.assert_array_equal(
        http.to_numpy(dtype=float), https.to_numpy(dtype=float)
    )


# --- Live smoke (env-gated, never in CI) ---


@pytest.mark.skipif(
    os.environ.get("PHISHNET_LIVE_NETWORK") != "1",
    reason="live RDAP/CT queries; set PHISHNET_LIVE_NETWORK=1 to run",
)
def test_live_example_com_rdap_and_crtsh() -> None:
    bootstrap = rdap.fetch_bootstrap()
    assert "com" in bootstrap
    age = rdap.fetch_age("example.com", bootstrap)
    assert age["creation_date"] is not None and age["source"] == "rdap"
    hist = ct.fetch_ct("example.com", allow_postgres=False)
    assert hist["certs"] and hist["provider"] == "crt.sh-json"
