"""CORS contract: localhost + Chromium extensions in, web pages out.

Regression backing for docs/chrome-extension-id-cors.md: a published Web
Store ID must work without pinning (regex mode), pinning must stay exact,
and https:// origins must never match.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from phishnet.serving.app import create_app

DEV_ID = "cphacgebncakdmjbpoibajnihhbbcjec"
STORE_ID = "abcdefghabcdefghabcdefghabcdefgh"
OTHER_ID = "qrstuvwxqrstuvwxqrstuvwxqrstuvwx"

LOCALHOST = "http://localhost:8000"
EVIL = "https://evil-website.com"


class _StubTier1:
    model_hash = "d" * 64
    columns_hash = "e" * 64
    columns = ["url_len"]
    thresholds = {
        "t_alert": 0.9269363298832987,
        "lower_edge": 0.6493076453312958,
        "t_1pct": 0.8780843789420926,
    }
    thresholds_source = "test:deadbeef"


def _allow_origin(extension_id: str | None, origin: str) -> str | None:
    app = create_app(servable=_StubTier1(), extension_id=extension_id)  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    value = response.headers.get("access-control-allow-origin")
    return str(value) if value is not None else None


def _ext(origin_id: str) -> str:
    return f"chrome-extension://{origin_id}"


def test_pinned_dev_id_allowed() -> None:
    assert _allow_origin(DEV_ID, _ext(DEV_ID)) == _ext(DEV_ID)


def test_unpinned_store_id_blocked_in_pin_mode() -> None:
    assert _allow_origin(DEV_ID, _ext(STORE_ID)) is None


def test_regex_mode_allows_any_extension_id() -> None:
    assert _allow_origin("*", _ext(DEV_ID)) == _ext(DEV_ID)
    assert _allow_origin("*", _ext(STORE_ID)) == _ext(STORE_ID)


def test_regex_mode_blocks_web_origins() -> None:
    assert _allow_origin("*", EVIL) is None
    assert _allow_origin("*", "http://localhost:9999") is None


def test_regex_mode_blocks_malformed_extension_origins() -> None:
    assert _allow_origin("*", "chrome-extension://short") is None
    assert _allow_origin("*", "chrome-extension://evil.com") is None
    assert (
        _allow_origin("*", "chrome-extension://ABCDEFGHABCDEFGHABCDEFGHABCDEFGH")
        is None
    )


def test_comma_list_pins_multiple_ids() -> None:
    both = f"{DEV_ID},{STORE_ID}"
    assert _allow_origin(both, _ext(DEV_ID)) == _ext(DEV_ID)
    assert _allow_origin(both, _ext(STORE_ID)) == _ext(STORE_ID)
    assert _allow_origin(both, _ext(OTHER_ID)) is None


def test_localhost_allowed_in_pin_mode() -> None:
    assert _allow_origin(DEV_ID, LOCALHOST) == LOCALHOST
