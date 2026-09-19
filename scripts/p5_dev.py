"""Phase 5 dev sweep driver: baseline and hardened versions on dev pages.

Baseline (`p5-dev-baseline`, `p4-v1`, 1 repeat): ALL dev pages — clean,
injected ordinary, probes, framing, dev aware. Hardened versions
(`p5-dev-hN`, prompt `p5-hN`): dev ORDINARY INJECTED pages only, 1 repeat
each (cap arithmetic: 72 + 3x30 = 162 <= 200; clean cost per version is not
measured during iteration — it is judged on held-out clean at the end per
§5.3, and `phase5-H` forbids tuning to clean anyway).

Structural guards, fail loudly: every judged page must be `split == dev`
(held-out never sent before the hardened freeze — not here, not anywhere);
bodies and extracts hash-verified against the committed manifest (judging
pinned bytes only). Every call through `llm/cache.py` (run id + repeat
index); sealed successes count toward the dev cap of 200 (`phase5-E`),
failures logged with their own disposition, attempts bounded. Pacing interval
read from the committed gate measurement (`pacing.json`), not re-derived.
Resumes existing cached calls for the exact run id if interrupted.

`--prompt-version` is accepted but currently pinned to `p4-v1`: hardened
prompts need a client prompt parameter that does not exist yet (deferred to
the freeze step, with its own review — the pinned adapter is not refactored
casually for a version that does not exist).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, ".")

from phishnet.llm import cache as Q  # noqa: E402
from phishnet.llm.client import judge  # noqa: E402
from phishnet.llm.schema import MODEL_ID, RESPONSE_SCHEMA  # noqa: E402
from phishnet.snapshot.extract import (  # noqa: E402
    canonical_extract,
    extract_hash,
    to_model_text,
)

MANIFEST = Path("reports/adversarial-manifest-p5.json")
BODIES_CLEAN = Path("data/adversarial-p5/clean")
BODIES_INJECTED = Path("data/adversarial-p5/injected")
PACING = Path("runs/phase5/p5-gate/pacing.json")
RUNS = Path("runs/phase5")
CACHE_DIR = Path("runs/phase5/cache")
MAX_ATTEMPTS_MULTIPLE = 2


def load_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("GROQ_API_KEY="):
                key = stripped.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise SystemExit("GROQ_API_KEY not found")
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-version", default="p4-v1")
    parser.add_argument("--ordinary-injected-only", action="store_true")
    args = parser.parse_args(argv)
    assert args.prompt_version == "p4-v1", (
        "hardened prompts need the client parameter (deferred to freeze)"
    )

    key = load_key()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = [r for r in manifest if r["split"] == "dev"]
    assert targets, "no dev pages"
    assert all(r["split"] == "dev" for r in targets), "held-out leak into dev run"
    if args.ordinary_injected_only:
        targets = [
            r
            for r in targets
            if r["kind"] == "injected" and r["payload_family"] == "ordinary"
        ]
    pacing = json.loads(PACING.read_text(encoding="utf-8"))
    interval = float(pacing["pacing_interval_s"])
    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    max_attempts = MAX_ATTEMPTS_MULTIPLE * len(targets)

    sealed: list[dict] = []
    successes = 0
    cached_hits = 0
    fresh_calls = 0
    attempts = 0
    fingerprints: list[str | None] = []

    for row in targets:
        if successes >= len(targets):
            break
        body_dir = BODIES_CLEAN if row["kind"] == "clean" else BODIES_INJECTED
        raw = (body_dir / f"{row['page_id']}.html").read_bytes()
        assert raw, f"missing body for {row['page_id']}"
        assert hashlib.sha256(raw).hexdigest() == row["sha256_raw_html"], row["page_id"]

        html = raw.decode("utf-8")
        url = str(row["url"])
        page_host = (urllib.parse.urlsplit(url).hostname or "").lower()
        extract = canonical_extract(html, url)
        assert extract_hash(extract) == row["sha256_canonical_extract"], row["page_id"]
        ckey = Q.cache_key(
            extract_hash(extract), args.prompt_version, MODEL_ID, args.run_id, 0
        )

        prior = Q.lookup(CACHE_DIR, ckey)
        if prior is not None:
            # Resuming an already sealed call for this exact run id
            assert set(prior.get("parsed", {})) == set(RESPONSE_SCHEMA["required"]), (
                f"cached entry violates schema: {ckey}"
            )
            sealed.append(prior)
            successes += 1
            cached_hits += 1
            fingerprints.append(prior.get("system_fingerprint"))
            verdict = prior.get("verdict")
            print(f"[{successes}/{len(targets)}] {row['page_id']}: CACHED {verdict}")
            continue

        record: dict | None = None
        while record is None and attempts < max_attempts:
            attempts += 1
            fresh_calls += 1
            req, judgment = judge(key, page_host, to_model_text(extract))
            usage = judgment.usage if isinstance(judgment.usage, dict) else {}
            if judgment.ok and judgment.parsed is not None:
                assert set(judgment.parsed) == set(RESPONSE_SCHEMA["required"]), (
                    f"strict schema not honored: {sorted(judgment.parsed)}"
                )
                record = {
                    "page_id": row["page_id"],
                    "url": url,
                    "snapshot_hash": extract_hash(extract),
                    "cache_key": ckey,
                    "cold_cache": True,
                    "prompt_version": args.prompt_version,
                    "verdict": judgment.parsed.get("verdict"),
                    "parsed": judgment.parsed,
                    "system_fingerprint": judgment.fingerprint,
                    "usage": usage,
                    "latency_ms": judgment.latency_ms,
                    "status": judgment.status,
                    "error": judgment.error,
                }
                Q.store(CACHE_DIR, ckey, record)
                successes += 1
                fingerprints.append(judgment.fingerprint)
                v = record["verdict"]
                print(f"[{successes}/{len(targets)}] {row['page_id']}: {v}")
            else:
                sealed.append(
                    {
                        "page_id": row["page_id"],
                        "url": url,
                        "snapshot_hash": extract_hash(extract),
                        "cache_key": ckey,
                        "cold_cache": True,
                        "prompt_version": args.prompt_version,
                        "verdict": None,
                        "status": judgment.status,
                        "error": judgment.error or "unparsed",
                        "usage": usage,
                        "latency_ms": judgment.latency_ms,
                    }
                )
                err = judgment.error or "unparsed"
                print(
                    f"[{successes}/{len(targets)}] {row['page_id']}: "
                    f"FAILED {judgment.status} ({err})"
                )
            if successes < len(targets):
                time.sleep(interval)
        if record is not None:
            sealed.append(record)

    distinct_fp = sorted({f for f in fingerprints if f})
    run = {
        "run_id": args.run_id,
        "run_class": "provisional",
        "model": MODEL_ID,
        "prompt_version": args.prompt_version,
        "seed": 0,
        "tier": "free",
        "n_targeted": len(targets),
        "n_success": successes,
        "n_failures": len(sealed) - successes,
        "n_attempts": attempts,
        "n_cached": cached_hits,
        "n_fresh": fresh_calls,
        "pacing_interval_s": interval,
        "fingerprints": distinct_fp,
        "n_fingerprints": len(distinct_fp),
        "identity_note": "phase4-C: distribution, never a predicate",
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=1) + "\n", encoding="utf-8"
    )
    (run_dir / "calls.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in sealed), encoding="utf-8"
    )
    summary_msg = (
        f"done: {successes}/{len(targets)} successes "
        f"({cached_hits} cached, {fresh_calls} fresh), {attempts} attempts"
    )
    print(summary_msg)
    assert successes == len(targets), f"short: {successes}/{len(targets)}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
