"""Tier-2 page fetcher, a separate image from the Tier-1 server.

Serving decision 2 keeps Playwright out of the Tier-1 image so the latency
number is not measured beside a browser. This service fetches one URL and
returns the frozen canonical extract (raw HTML is never returned). It
renders with Playwright when available and falls back to a plain HTTP fetch;
the extract is the same pure function either way, so the model sees the
same shape.

Live mode is opt-in: `LiveTier2Provider` posts here only when
`PHISHNET_FETCHER_URL` (and a key) are set.
"""

from __future__ import annotations

import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from phishnet.snapshot.extract import canonical_extract

UA = "PhishNet-phase6-fetcher/1.0 (+research; single fetch)"
TIMEOUT_S = float(os.getenv("PHISHNET_FETCH_TIMEOUT", "15"))
RENDER = os.getenv("PHISHNET_FETCHER_RENDER", "1") == "1"

app = FastAPI(title="PhishNet fetcher (Phase 6)")


class FetchRequest(BaseModel):
    url: HttpUrl


def fetch_html(url: str, timeout: float = TIMEOUT_S) -> tuple[str, str]:
    """Return ``(html, final_url)``; Playwright when available, else requests."""
    if RENDER:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox"])
                try:
                    page = browser.new_page()
                    page.goto(
                        url, wait_until="domcontentloaded", timeout=timeout * 1000
                    )
                    return page.content(), page.url
                finally:
                    browser.close()
        except ImportError:
            pass
        except Exception:
            pass
    response = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    response.raise_for_status()
    return response.text, str(response.url)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "render": RENDER, "timeout_s": TIMEOUT_S}


@app.post("/fetch")
def fetch(body: FetchRequest) -> dict[str, object]:
    url = str(body.url)
    try:
        html, final_url = fetch_html(url)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"unfetchable:{type(e).__name__}"
        ) from e
    return {
        "url": url,
        "final_url": final_url,
        "extract": canonical_extract(html, final_url),
    }
