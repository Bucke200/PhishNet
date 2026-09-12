"""Verified download + atomic install for ML model artifacts.

Flow per artifact::

    trusted source (manifest URL or ``PHISHNET_MODELS_BASE_URL`` override)
        -> download to temp file in the destination directory
        -> HTTP/download validation
        -> SHA256 over raw bytes (never unpickled)
        -> compare against trusted expected hash
        -> atomic ``os.replace`` into the final path

Only a file with ``actual_sha256 == expected_sha256`` becomes active.
This module replaced the former unverified ``backend/download_models.py``
(Google Drive fetch without verification); it resolves to the same
artifact filenames so existing load paths (``phishnet.api`` via
``$PHISHNET_ML_ASSETS_DIR``) keep working.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass

import requests  # type: ignore[import-untyped]

MANIFEST_PATH = pathlib.Path(__file__).with_name("model_manifest.json")

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 60.0
CHUNK_SIZE = 8192
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class DownloadError(RuntimeError):
    """Network/HTTP failure while fetching an artifact."""


class VerificationError(ValueError):
    """Downloaded bytes did not match the trusted manifest digest/size."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    url: str
    sha256: str
    size: int | None = None


def load_manifest(
    path: pathlib.Path | str = MANIFEST_PATH,
) -> tuple[str, dict[str, ArtifactSpec]]:
    """Load ``(version, specs)`` from the committed manifest JSON."""
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    version = str(raw["version"])
    specs: dict[str, ArtifactSpec] = {}
    artifacts = raw["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise VerificationError(f"Manifest {path} contains no artifacts")
    for name, entry in artifacts.items():
        digest = str(entry["sha256"]).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VerificationError(
                f"Manifest entry {name!r} has an invalid sha256 digest"
            )
        size = entry.get("size")
        specs[name] = ArtifactSpec(
            name=name,
            url=str(entry["url"]),
            sha256=digest,
            size=int(size) if size is not None else None,
        )
    return version, specs


def resolve_url(spec: ArtifactSpec) -> str:
    """Resolve the trusted download URL for an artifact.

    ``PHISHNET_MODELS_BASE_URL`` (release/object-storage migration) takes
    precedence and maps to ``<base>/<name>``; otherwise the manifest URL is
    used. Only manifest-controlled locations are ever fetched -- callers
    cannot supply arbitrary URLs.
    """
    base = os.getenv("PHISHNET_MODELS_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/{spec.name}"
    return spec.url


def resolve_dest_dir(dest_dir: pathlib.Path | str | None = None) -> pathlib.Path:
    """Resolve the artifact directory (same precedence as ``phishnet.api``)."""
    override = dest_dir if dest_dir is not None else os.getenv("PHISHNET_ML_ASSETS_DIR")
    if override:
        return pathlib.Path(override)
    return pathlib.Path(__file__).parent / "urlset_ml_assets"


def sha256_of_file(path: pathlib.Path | str) -> str:
    """SHA256 of raw file bytes. No deserialization is performed."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_artifact(path: pathlib.Path | str, spec: ArtifactSpec) -> bool:
    """Return True iff an existing file matches size and trusted digest."""
    p = pathlib.Path(path)
    if not p.is_file():
        return False
    if spec.size is not None:
        try:
            if p.stat().st_size != spec.size:
                return False
        except OSError:
            return False
    try:
        actual = sha256_of_file(p)
    except OSError:
        return False
    return hmac.compare_digest(actual.lower(), spec.sha256.lower())


def _is_transient_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _fetch_once(
    spec: ArtifactSpec,
    tmp_path: pathlib.Path,
    timeout: tuple[float, float],
) -> None:
    """Stream one attempt into ``tmp_path``; raises on HTTP/download errors."""
    url = resolve_url(spec)
    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except requests.Timeout as e:
        raise DownloadError(f"{spec.name}: timed out fetching {url}: {e}") from e
    except requests.ConnectionError as e:
        raise DownloadError(f"{spec.name}: connection error fetching {url}: {e}") from e

    with response:
        status = response.status_code
        if status != 200:
            if _is_transient_status(status):
                raise DownloadError(f"{spec.name}: transient HTTP {status} from {url}")
            raise DownloadError(f"{spec.name}: HTTP {status} from {url}")

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            raise DownloadError(
                f"{spec.name}: refused HTML response "
                f"(Content-Type: {content_type!r}); probable error page, "
                "not a model artifact"
            )
        length = response.headers.get("Content-Length")
        if length is not None and spec.size is not None:
            try:
                if int(length) != spec.size:
                    raise DownloadError(
                        f"{spec.name}: Content-Length {length} != expected {spec.size}"
                    )
            except ValueError:
                pass

        digest = hashlib.sha256()
        written = 0
        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    # Early HTML/error-page guard: Drive quota/consent pages
                    # are served with HTTP 200 but contain HTML, not pickle.
                    if written == 0 and chunk.lstrip()[:5].lower() in (
                        b"<html",
                        b"<!doc",
                    ):
                        raise DownloadError(
                            f"{spec.name}: response body looks like HTML, "
                            "not a model artifact"
                        )
                    f.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if spec.size is not None and written > spec.size:
                        raise DownloadError(
                            f"{spec.name}: downloaded more than expected "
                            f"{spec.size} bytes"
                        )
                f.flush()
                os.fsync(f.fileno())
        except DownloadError:
            raise
        except Exception as e:
            raise DownloadError(
                f"{spec.name}: interrupted download from {url}: {e}"
            ) from e

    if spec.size is not None and written != spec.size:
        raise VerificationError(
            f"{spec.name}: size mismatch: expected {spec.size} bytes, "
            f"got {written} bytes"
        )
    if written == 0:
        raise VerificationError(f"{spec.name}: refusing empty download")
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual.lower(), spec.sha256.lower()):
        raise VerificationError(
            f"{spec.name}: SHA256 mismatch: expected {spec.sha256}, got {actual}"
        )


def download_artifact(
    spec: ArtifactSpec,
    dest_dir: pathlib.Path | str | None = None,
    timeout: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT),
    max_attempts: int = MAX_ATTEMPTS,
) -> pathlib.Path:
    """Ensure ``spec.name`` exists verified in ``dest_dir``; return its path.

    Reuses an already-valid file without network I/O. Otherwise downloads
    to a temp file in the same directory, verifies size + SHA256 on raw
    bytes, then atomically replaces the destination. Temp files are always
    cleaned up; a failed replacement never destroys an existing valid file.
    """
    directory = resolve_dest_dir(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / spec.name

    if is_valid_artifact(dest, spec):
        print(f"{dest} already present and verified, reusing.")
        return dest

    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f"{spec.name}.tmp.")
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                _fetch_once(spec, tmp_path, timeout)
                break
            except VerificationError:
                # Wrong bytes from source: retrying is pointless; fail fast.
                raise
            except DownloadError as e:
                last_error = e
                if attempt >= max_attempts:
                    raise
                delay = RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                ]
                print(
                    f"{spec.name}: attempt {attempt} failed ({e}); "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
        else:  # pragma: no cover - loop always breaks or raises
            assert last_error is not None
            raise last_error
        os.replace(tmp_path, dest)
        print(f"{dest} verified (sha256 {spec.sha256[:12]}...) and installed.")
        return dest
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def ensure_artifacts(
    dest_dir: pathlib.Path | str | None = None,
    specs: dict[str, ArtifactSpec] | None = None,
) -> tuple[str, dict[str, pathlib.Path]]:
    """Ensure every manifest artifact is present and verified.

    Returns ``(manifest_version, {name: path})`` so production can log
    exactly which artifact version it is running.
    """
    version, manifest_specs = load_manifest()
    active = specs if specs is not None else manifest_specs
    installed: dict[str, pathlib.Path] = {}
    for name, spec in active.items():
        installed[name] = download_artifact(spec, dest_dir)
    print(f"Artifact set {version} ready: {sorted(installed)}")
    return version, installed


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(
        description="Download + verify PhishNet model artifacts."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Destination directory (default: $PHISHNET_ML_ASSETS_DIR "
        "or packaged urlset_ml_assets/)",
    )
    args = parser.parse_args()
    manifest_version, paths = ensure_artifacts(args.dir)
    print(f"Installed artifact version: {manifest_version}")
    for artifact_name, artifact_path in sorted(paths.items()):
        print(f"  {artifact_name}: {artifact_path}")
