"""Smoke test guarding the Common Crawl CDX query form.

* Offline (always runs): ``query_index`` must build a ``matchType=domain``
  query and the superseded bare-prefix ``url=<domain>/*`` SURT form must
  appear nowhere in ``build_cc_benign.py``.
* Live (network-gated): a ``matchType=domain`` query for a domain
  guaranteed to be in the index must return a non-empty result. Skipped
  unless ``PHISHNET_LIVE_NETWORK=1`` is set, so the offline suite
  (including CI) stays green with no network.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

import build_cc_benign as B

pytestmark = pytest.mark.network

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_query_form_constant_is_matchtype_domain() -> None:
    assert B.QUERY_FORM == "matchType=domain"


def test_query_index_source_uses_matchtype_not_surt_prefix() -> None:
    src = inspect.getsource(B.query_index)
    assert "matchType=domain" in src
    assert "/*" not in src


def test_no_surt_prefix_form_anywhere_in_module() -> None:
    text = (REPO_ROOT / "build_cc_benign.py").read_text(encoding="utf-8")
    assert "matchType=domain" in text
    # The v1 defect: url=<domain>/* misses subdomain captures by
    # SURT-prefix construction (404 on facebook.com). It must not return.
    # (The docstring may describe the old form in words with <angle>
    # brackets; only a {brace} interpolation back into a query counts.)
    assert "url={domain}/*" not in text
    assert "}/*" not in text


@pytest.mark.skipif(
    os.environ.get("PHISHNET_LIVE_NETWORK") != "1",
    reason="live CDX query; set PHISHNET_LIVE_NETWORK=1 to run",
)
def test_matchtype_domain_query_returns_records_live() -> None:
    status, records, note = B.query_index("example.com", B.CC_INDEX_PRIMARY)
    assert note == "ok", f"CDX query failed: status={status} note={note}"
    assert len(records) > 0
    assert any(
        str(r.get("url", "")).startswith(("http://", "https://")) for r in records
    )
