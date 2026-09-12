"""Tests for verified model-artifact downloads.

Uses tiny synthetic byte payloads only -- never the real production
``*.pkl`` files and never the network. ``requests.get`` is stubbed.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterator
from typing import Literal

import pytest

from phishnet import verified_download as vd


def _spec(payload: bytes, name: str = "scaler.pkl") -> tuple[vd.ArtifactSpec, bytes]:
    digest = hashlib.sha256(payload).hexdigest()
    return (
        vd.ArtifactSpec(
            name=name,
            url="https://example.invalid/models/" + name,
            sha256=digest,
            size=len(payload),
        ),
        payload,
    )


class _FakeResponse:
    def __init__(
        self,
        payload: bytes = b"",
        status: int = 200,
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(payload))}
        self._error = error

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        if self._error is not None:
            raise self._error
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vd.time, "sleep", lambda _s: None)


def _tmp_leftovers(directory: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in directory.iterdir() if ".tmp." in p.name]


def test_1_successful_download_and_sha256(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, payload = _spec(b"verified-model-bytes-1")
    monkeypatch.setattr(vd.requests, "get", lambda *a, **k: _FakeResponse(payload))
    dest = vd.download_artifact(spec, tmp_path, max_attempts=1)
    assert dest.read_bytes() == payload
    assert vd.is_valid_artifact(dest, spec)
    assert _tmp_leftovers(tmp_path) == []


def test_2_incorrect_sha256_rejected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _spec(b"expected-bytes")
    bad = b"expected-bytez"  # same length, different digest
    monkeypatch.setattr(
        vd.requests,
        "get",
        lambda *a, **k: _FakeResponse(bad, headers={"Content-Length": str(spec.size)}),
    )
    with pytest.raises(vd.VerificationError) as exc:
        vd.download_artifact(spec, tmp_path, max_attempts=1)
    assert spec.sha256 in str(exc.value)
    assert not (tmp_path / spec.name).exists()
    assert _tmp_leftovers(tmp_path) == []


def test_3_http_failure_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    spec, _ = _spec(b"model-bytes")
    monkeypatch.setattr(
        vd.requests, "get", lambda *a, **k: _FakeResponse(b"", status=500)
    )
    with pytest.raises(vd.DownloadError):
        vd.download_artifact(spec, tmp_path, max_attempts=2)
    assert not (tmp_path / spec.name).exists()
    assert _tmp_leftovers(tmp_path) == []


def test_4_interrupted_download_cleaned_up(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _spec(b"model-bytes")
    boom = ConnectionError("connection reset mid-stream")
    monkeypatch.setattr(vd.requests, "get", lambda *a, **k: _FakeResponse(error=boom))
    with pytest.raises(vd.DownloadError):
        vd.download_artifact(spec, tmp_path, max_attempts=1)
    assert not (tmp_path / spec.name).exists()
    assert _tmp_leftovers(tmp_path) == []


def test_5_existing_valid_artifact_reused_without_network(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, payload = _spec(b"already-good-bytes")
    (tmp_path / spec.name).write_bytes(payload)

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("network must not be used")

    monkeypatch.setattr(vd.requests, "get", _fail)
    dest = vd.download_artifact(spec, tmp_path, max_attempts=1)
    assert dest.read_bytes() == payload


def test_6_failed_replacement_preserves_existing_valid_artifact(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file valid for v1 must survive a failed upgrade to v2.
    old_spec, v1 = _spec(b"v1-bytes!!!!")
    new_spec, _ = _spec(b"v2-bytes!!!!", name=old_spec.name)
    assert vd.is_valid_artifact(tmp_path / old_spec.name, old_spec) is False
    (tmp_path / old_spec.name).write_bytes(v1)
    assert vd.is_valid_artifact(tmp_path / old_spec.name, old_spec)
    bad = b"corrupted!!!"
    monkeypatch.setattr(
        vd.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            bad, headers={"Content-Length": str(new_spec.size)}
        ),
    )
    with pytest.raises(vd.VerificationError):
        vd.download_artifact(new_spec, tmp_path, max_attempts=1)
    assert (tmp_path / old_spec.name).read_bytes() == v1
    assert vd.is_valid_artifact(tmp_path / old_spec.name, old_spec)
    assert _tmp_leftovers(tmp_path) == []


def test_7_no_temp_files_left_after_success_or_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_ok, payload_ok = _spec(b"ok-bytes", name="a.pkl")
    monkeypatch.setattr(vd.requests, "get", lambda *a, **k: _FakeResponse(payload_ok))
    vd.download_artifact(spec_ok, tmp_path, max_attempts=1)

    spec_bad, _ = _spec(b"wanted-bytes", name="b.pkl")
    bad_payload = b"wrong-bytes!"
    monkeypatch.setattr(
        vd.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            bad_payload, headers={"Content-Length": str(spec_bad.size)}
        ),
    )
    with pytest.raises(vd.VerificationError):
        vd.download_artifact(spec_bad, tmp_path, max_attempts=1)

    assert _tmp_leftovers(tmp_path) == []
    assert (tmp_path / "a.pkl").read_bytes() == payload_ok
    assert not (tmp_path / "b.pkl").exists()


def test_manifest_lists_all_required_artifacts() -> None:
    version, specs = vd.load_manifest()
    assert version
    assert set(specs) == {
        "urlset_ensemble_model.pkl",
        "scaler.pkl",
        "feature_columns.pkl",
    }
    for spec in specs.values():
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://")
