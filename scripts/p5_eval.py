"""Phase 5 held-out evaluation sweep driver (§4, §5.3, §8).

Executes the registered held-out evaluation across 3 repeated cold judgments
(repeat_idx in {0, 1, 2}) under baseline prompt `p4-v1` and frozen hardened
prompt `p5-h1`.

Targets per repeat (prereg §4.1, §8):
- 36 clean held-out base pages (kind == "clean")
- 58 reaching held-out injected pages (kind == "injected", reached == True)
Total: 94 targets per run.

Budget (prereg §8):
(36 clean + 58 injected) x 2 prompts x 3 repeats = 564 calls (cap 600).

Structural guards:
- Every judged page must have `split == "held_out"`;
- Bodies and extracts hash-verified against the committed manifest;
- Every call cached with `(snapshot_hash, prompt_version, model, run_id, repeat_idx)`
  guaranteeing independent cold keys across repeats;
- Deterministic failures (e.g. 400 schema violation, refusal) sealed per Phase 4 §2
  (retain tier-1 score) with their own disposition;
- Pacing interval loaded directly from `runs/phase5/p5-gate/pacing.json`.
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

ALLOWED_PROMPT_VERSIONS = ("p4-v1", "p5-h1")


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
    parser.add_argument(
        "--prompt-version",
        required=True,
        choices=ALLOWED_PROMPT_VERSIONS,
        help="Prompt version: p4-v1 (unhardened baseline) or p5-h1 (frozen hardened)",
    )
    parser.add_argument(
        "--repeat-idx",
        required=True,
        type=int,
        choices=(0, 1, 2),
        help="Repeat index for cold judgment: 0, 1, or 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify manifest, files, and hashes without invoking LLM",
    )
    args = parser.parse_args(argv)

    key = "" if args.dry_run else load_key()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # Select all held-out targets: clean bases + reaching injected pages
    targets = [
        r
        for r in manifest
        if r["split"] == "held_out"
        and (r["kind"] == "clean" or (r["kind"] == "injected" and r["reached"]))
    ]
    assert len(targets) == 94, f"unexpected held-out target count: {len(targets)}"
    assert all(r["split"] == "held_out" for r in targets), (
        "dev page leaked into held-out run"
    )

    pacing = json.loads(PACING.read_text(encoding="utf-8"))
    interval = float(pacing["pacing_interval_s"])
    run_dir = RUNS / args.run_id
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    max_attempts = MAX_ATTEMPTS_MULTIPLE * len(targets)

    sealed: list[dict] = []
    successes = 0
    failures = 0
    cached_hits = 0
    fresh_calls = 0
    attempts = 0
    fingerprints: list[str | None] = []

    for row in targets:
        if len(sealed) >= len(targets):
            break
        body_dir = BODIES_CLEAN if row["kind"] == "clean" else BODIES_INJECTED
        raw = (body_dir / f"{row['page_id']}.html").read_bytes()
        assert raw, f"missing body for {row['page_id']}"
        assert hashlib.sha256(raw).hexdigest() == row["sha256_raw_html"], row["page_id"]

        url = row["url"]
        page_host = urllib.parse.urlsplit(url).netloc.split(":")[0].lower()
        html = raw.decode("utf-8")
        extract = canonical_extract(html, url)
        assert extract_hash(extract) == row["sha256_canonical_extract"], row["page_id"]

        if args.dry_run:
            sealed.append({"page_id": row["page_id"], "dry_run": True})
            continue

        ckey = Q.cache_key(
            extract_hash(extract),
            args.prompt_version,
            MODEL_ID,
            args.run_id,
            args.repeat_idx,
        )

        prior = Q.lookup(CACHE_DIR, ckey)
        if prior is not None:
            if prior.get("verdict") is not None:
                req = set(RESPONSE_SCHEMA["required"])
                assert set(prior.get("parsed", {})) == req, (
                    f"cached entry violates schema: {ckey}"
                )
                successes += 1
            else:
                failures += 1
            sealed.append(prior)
            cached_hits += 1
            if prior.get("system_fingerprint"):
                fingerprints.append(prior.get("system_fingerprint"))
            verdict = prior.get("verdict")
            print(
                f"[{len(sealed)}/{len(targets)}] {row['page_id']}: CACHED {verdict}",
                flush=True,
            )
            continue

        record: dict | None = None
        while record is None and attempts < max_attempts:
            attempts += 1
            fresh_calls += 1
            req, judgment = judge(
                key,
                page_host,
                to_model_text(extract),
                prompt_version=args.prompt_version,
            )
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
                    "repeat_idx": args.repeat_idx,
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
                print(
                    f"[{len(sealed) + 1}/{len(targets)}] {row['page_id']}: {v}",
                    flush=True,
                )
                sealed.append(record)
            elif judgment.status == 429:
                backoff = max(interval, getattr(judgment, "retry_after", 30.0), 30.0)
                print(
                    f"Rate limited (429). Backing off {backoff:.1f}s...",
                    flush=True,
                )
                time.sleep(backoff)
            else:
                # Deterministic failure (e.g. 400 schema violation, refusal).
                # Sealed per Phase 4 §2 (retain tier-1 score).
                failures += 1
                record = {
                    "page_id": row["page_id"],
                    "url": url,
                    "snapshot_hash": extract_hash(extract),
                    "cache_key": ckey,
                    "cold_cache": True,
                    "prompt_version": args.prompt_version,
                    "repeat_idx": args.repeat_idx,
                    "verdict": None,
                    "status": judgment.status,
                    "error": judgment.error or "unparsed",
                    "usage": usage,
                    "latency_ms": judgment.latency_ms,
                }
                Q.store(CACHE_DIR, ckey, record)
                err = judgment.error or "unparsed"
                print(
                    f"[{len(sealed) + 1}/{len(targets)}] {row['page_id']}: "
                    f"SEALED FAILURE {judgment.status} ({err})",
                    flush=True,
                )
                sealed.append(record)
            if len(sealed) < len(targets) and judgment.status != 429:
                time.sleep(interval)

    if args.dry_run:
        print(f"Dry run complete: verified {len(sealed)} held-out targets.")
        return 0

    distinct_fp = sorted({f for f in fingerprints if f})
    run = {
        "run_id": args.run_id,
        "run_class": "registered",
        "model": MODEL_ID,
        "prompt_version": args.prompt_version,
        "repeat_idx": args.repeat_idx,
        "seed": args.repeat_idx,
        "tier": "free",
        "n_targeted": len(targets),
        "n_success": successes,
        "n_failures": failures,
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
        f"done: {len(sealed)}/{len(targets)} sealed "
        f"({successes} successes, {failures} failures, "
        f"{cached_hits} cached, {fresh_calls} fresh)"
    )
    print(summary_msg, flush=True)
    assert len(sealed) == len(targets), f"short: {len(sealed)}/{len(targets)}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
