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
import time
import urllib.parse
from pathlib import Path
from typing import Any

from phishnet.features.extraction import canonicalize_scheme
from phishnet.llm.client import Judgment, judge
from phishnet.serving.cascade import Tier2Outcome
from phishnet.snapshot.extract import to_model_text

MANIFEST = Path("reports/adversarial-manifest-p5.json")
PHASE5_RUNS = Path("runs/phase5")
DEFAULT_ARM = "p5-h1"
# Provider -> fetcher RPC budget: a trigger means the fetcher is saturated,
# distinct from an origin that never answered (the fetcher's own 8 s budget).
RPC_TIMEOUT_S = 25.0


def _canonical_url(url: str) -> str:
    """Scheme-insensitive URL key.

    The manifest records the Phase 5 spelling, but a browser may upgrade
    `http://` to `https://` (or the site may redirect), so an exact-string
    lookup misses the same page under the other scheme. Row (a) already
    treats the two as identical (`canonicalize_scheme`), so the sealed index
    uses the same rule.
    """
    return canonicalize_scheme(url)


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
            self._by_url[_canonical_url(row["url"])] = (
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
        entry = self._by_url.get(_canonical_url(url))
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


def _judge_with_retry(
    api_key: str,
    page_host: str,
    extract_text: str,
    prompt_version: str,
    *,
    attempts: int = 2,
    wait_s: float = 2.0,
) -> Judgment:
    """One governed call with a bounded transient retry (T2-7).

    A transport failure (`status -1`) or a provider 5xx is retried once after
    `wait_s`, so a single blip cannot turn an in-band page into a
    failure-alert. A 429 (quota) is returned immediately: the registered
    backoff is 60 s, longer than any serving request should wait.
    """
    attempts_left = attempts
    while True:
        _, judgment = judge(
            api_key,
            page_host,
            extract_text,
            prompt_version=prompt_version,
        )
        attempts_left -= 1
        if judgment.status == 429:
            return judgment
        if 0 <= judgment.status < 500:
            return judgment
        if attempts_left <= 0:
            return judgment
        time.sleep(wait_s)


class LiveTier2Provider:
    """Live Tier-2: fetch an extract, then one governed Groq call.

    The fetcher answers 200 with a structured body; a target-side failure
    arrives as ``{"ok": false, "error": <mechanism>, ...}`` and is passed
    through verbatim so the serving failure policy can key on the mechanism
    (http_403, dns, origin_timeout, …). A provider->fetcher RPC timeout is a
    distinct condition (internal saturation) and maps to ``fetcher_timeout``.
    """

    mode = "live"

    def __init__(
        self,
        fetcher_url: str,
        api_key: str,
        prompt_version: str = "p6-v1",
        rpc_timeout: float = RPC_TIMEOUT_S,
    ):
        self.fetcher_url = fetcher_url
        self.api_key = api_key
        self.prompt_version = prompt_version
        self.rpc_timeout = rpc_timeout

    def judge(self, url: str) -> Tier2Outcome | None:
        import requests

        page_host = urllib.parse.urlsplit(url).netloc.split(":")[0].lower()
        try:
            response = requests.post(
                self.fetcher_url, json={"url": url}, timeout=self.rpc_timeout
            )
        except requests.exceptions.Timeout:
            return Tier2Outcome(
                "failure",
                "fetcher_timeout",
                trigger_type="internal",
                trigger_match="rpc_timeout",
            )
        except requests.exceptions.RequestException:
            return Tier2Outcome(
                "failure",
                "fetcher_error",
                trigger_type="internal",
                trigger_match="rpc_error",
            )
        try:
            payload = response.json()
        except ValueError:
            return Tier2Outcome(
                "failure",
                "fetcher_http",
                trigger_type="internal",
                trigger_match="bad_payload",
            )
        if not isinstance(payload, dict):
            return Tier2Outcome(
                "failure",
                "fetcher_http",
                trigger_type="internal",
                trigger_match="bad_payload",
            )
        if not payload.get("ok", False):
            return Tier2Outcome(
                "failure",
                str(payload.get("error", "other")),
                trigger_type=str(payload.get("trigger_type", "")),
                trigger_match=str(payload.get("trigger_match", "")),
            )
        extract = payload.get("extract")
        if not isinstance(extract, dict):
            return Tier2Outcome("failure", "fetcher_http")

        from phishnet.adversarial.detect import detect_serving
        from phishnet.llm.budget import BudgetError

        if detect_serving(extract)["hit"]:
            return Tier2Outcome("phishing", "detector")
        try:
            judgment = _judge_with_retry(
                self.api_key,
                page_host,
                to_model_text(extract),
                self.prompt_version,
            )
        except BudgetError as exc:
            # Spend guard tripped (cap exceeded or STOP sentinel): nothing was
            # sent. A failure outcome lets the cascade decide, not a 500.
            return Tier2Outcome("failure", f"budget:{type(exc).__name__}")
        if judgment.ok and judgment.parsed is not None:
            return Tier2Outcome(str(judgment.parsed.get("verdict", "suspicious")))
        return Tier2Outcome("failure", f"http={judgment.status}")


def provider_from_env() -> SealedTier2Provider | LiveTier2Provider | None:
    """Build the Tier-2 provider from the environment, or None (disabled).

    `live` is fail-loud: if the key or fetcher URL is missing, startup
    refuses rather than silently degrading to a disabled LLM layer (which
    would show every in-band URL as "can't assess"). `sealed` still degrades
    to None when the demo data is absent, because that is a packaging
    condition, not a misconfiguration.
    """
    mode = os.getenv("PHISHNET_TIER2_MODE", "sealed").lower()
    if mode == "disabled":
        return None
    if mode == "live":
        key = os.getenv("GROQ_API_KEY", "")
        fetcher = os.getenv("PHISHNET_FETCHER_URL", "")
        missing = [
            name
            for name, value in (
                ("GROQ_API_KEY", key),
                ("PHISHNET_FETCHER_URL", fetcher),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "PHISHNET_TIER2_MODE=live requires "
                + ", ".join(missing)
                + " (refusing to start with a silently disabled LLM layer)"
            )
        return LiveTier2Provider(fetcher, key)
    try:
        return SealedTier2Provider()
    except FileNotFoundError:
        return None
