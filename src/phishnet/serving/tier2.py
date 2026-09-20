"""Tier-2 providers (C6/phase6-A).

The demo runs Tier 2 from the **sealed Phase 5 response cache** by default
(`SealedTier2Provider`): the scripted pages are the frozen adversarial
pages, and their `p5-h1` verdicts are already sealed, so the demo is offline
and reproducible. Live Groq calls are opt-in (`LiveTier2Provider`), gated
behind a key and a separate Playwright fetcher service; the demo labels
which mode produced each verdict.

Both providers return a :class:`~phishnet.serving.cascade.Tier2Outcome`, or
``None`` when the URL is not in the registered demo set. The cascade treats
``None`` as ``can't assess`` and turns failures into ``alert`` (fail
closed).
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

from phishnet.llm.client import judge
from phishnet.serving.cascade import Tier2Outcome
from phishnet.snapshot.extract import to_model_text

MANIFEST = Path("reports/adversarial-manifest-p5.json")
PHASE5_RUNS = Path("runs/phase5")
DEFAULT_ARM = "p5-h1"


class SealedTier2Provider:
    """Replay sealed Phase 5 verdicts by URL (offline demo default)."""

    mode = "sealed"

    def __init__(
        self,
        manifest: Path = MANIFEST,
        runs_dir: Path = PHASE5_RUNS,
        arm: str = DEFAULT_ARM,
    ):
        self._by_url: dict[str, tuple[str, bool]] = {}
        self._by_hash: dict[str, dict[str, Any]] = {}
        if not manifest.is_file():
            raise FileNotFoundError(f"sealed Tier-2 manifest missing: {manifest}")
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            self._by_url[row["url"]] = (
                row["sha256_canonical_extract"],
                bool(row.get("detector_hit")),
            )
        for path in sorted(runs_dir.glob("p5-eval-*/calls.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("prompt_version") != arm:
                    continue
                self._by_hash.setdefault(record["snapshot_hash"], record)

    def judge(self, url: str) -> Tier2Outcome | None:
        entry = self._by_url.get(url)
        if entry is None:
            return None
        extract_hash, detector_hit = entry
        # phase6-A: the frozen detector escalates a hit before any LLM call.
        if detector_hit:
            return Tier2Outcome("phishing", "detector")
        record = self._by_hash.get(extract_hash)
        if record is None:
            return None
        verdict = record.get("verdict")
        if verdict is None:
            return Tier2Outcome("failure", record.get("error", "unparsed"))
        return Tier2Outcome(str(verdict))


class LiveTier2Provider:
    """Live Tier-2: fetch an extract, then one governed Groq call."""

    mode = "live"

    def __init__(self, fetcher_url: str, api_key: str, prompt_version: str = "p6-v1"):
        self.fetcher_url = fetcher_url
        self.api_key = api_key
        self.prompt_version = prompt_version

    def judge(self, url: str) -> Tier2Outcome | None:
        import requests

        page_host = urllib.parse.urlsplit(url).netloc.split(":")[0].lower()
        try:
            response = requests.post(self.fetcher_url, json={"url": url}, timeout=20)
            response.raise_for_status()
            extract = response.json()["extract"]
        except Exception as e:
            return Tier2Outcome("failure", f"unfetchable:{type(e).__name__}")
        from phishnet.adversarial.detect import detect

        if detect(extract)["hit"]:
            return Tier2Outcome("phishing", "detector")
        _, judgment = judge(
            self.api_key,
            page_host,
            to_model_text(extract),
            prompt_version=self.prompt_version,
        )
        if judgment.ok and judgment.parsed is not None:
            return Tier2Outcome(str(judgment.parsed.get("verdict", "suspicious")))
        return Tier2Outcome("failure", f"http={judgment.status}")


def provider_from_env() -> SealedTier2Provider | LiveTier2Provider | None:
    """Build the Tier-2 provider from the environment, or None (disabled)."""
    mode = os.getenv("PHISHNET_TIER2_MODE", "sealed").lower()
    if mode == "disabled":
        return None
    if mode == "live":
        key = os.getenv("GROQ_API_KEY", "")
        fetcher = os.getenv("PHISHNET_FETCHER_URL", "")
        if key and fetcher:
            return LiveTier2Provider(fetcher, key)
        return None
    try:
        return SealedTier2Provider()
    except FileNotFoundError:
        return None
