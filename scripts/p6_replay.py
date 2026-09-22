"""C4 replay: re-send the 50 Phase 5 schema failures under `p6-v1`.

The 50 sealed Phase 5 calls that failed provider strict-schema validation
(`status: 400`) are re-sent one-for-one against their frozen extracts under
the widened `p6-v1` schema (phase6-B). The replay is paired before/after on
the same page set: each sealed failure already carries its `prompt_version`
and `repeat_idx`. No evasion or detection claim is made from it; Phase 5's
arm verdicts stay attached to `p4-v1`/`p5-h1`.

Bodies and extracts are hash-verified against `adversarial-manifest-p5.json`
before any call. Sealed under `runs/phase6/p6-replay/` with a cache key that
cannot collide with Phase 5 (`run_id=p6-replay`, `prompt_version=p6-v1`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, ".")

from phishnet.llm import cache as Q  # noqa: E402
from phishnet.llm.client import judge  # noqa: E402
from phishnet.llm.schema import MODEL_ID, response_schema  # noqa: E402
from phishnet.snapshot.extract import (  # noqa: E402
    canonical_extract,
    extract_hash,
    to_model_text,
)

MANIFEST = Path("reports/adversarial-manifest-p5.json")
BODIES = {
    "clean": Path("data/adversarial-p5/clean"),
    "injected": Path("data/adversarial-p5/injected"),
}
PHASE5_RUNS = Path("runs/phase5")
CACHE_DIR = Path("runs/phase6/cache")
P6_PROMPT = "p6-v1"
RUN_ID = "p6-replay"
OUT_DIR = Path("runs/phase6") / RUN_ID
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


def load_failures() -> list[dict]:
    """The 50 sealed Phase 5 calls with status 400."""
    rows: list[dict] = []
    for path in sorted(PHASE5_RUNS.glob("p5-eval-*/calls.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") == 400:
                record["_source_run"] = path.parent.name
                rows.append(record)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = {
        r["page_id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))
    }
    failures = load_failures()
    print(f"loaded {len(failures)} sealed schema failures", flush=True)
    if not failures:
        raise SystemExit("no status-400 calls found in runs/phase5")

    key = "" if args.dry_run else load_key()
    schema = response_schema(P6_PROMPT)
    required = set(schema["required"])
    interval = 2.0
    max_attempts = MAX_ATTEMPTS_MULTIPLE * len(failures)

    sealed: list[dict] = []
    successes = 0
    failures_after = 0
    cached_hits = 0
    fresh_calls = 0
    attempts = 0
    rate_limited = 0

    for record in failures:
        page_id = record["page_id"]
        row = manifest.get(page_id)
        if row is None:
            raise SystemExit(f"{page_id} missing from manifest")
        extract = canonical_extract(
            (BODIES[row["kind"]] / f"{page_id}.html").read_text(encoding="utf-8"),
            row["url"],
        )
        if extract_hash(extract) != row["sha256_canonical_extract"]:
            raise SystemExit(f"{page_id}: canonical extract hash mismatch")
        repeat_idx = int(record["repeat_idx"])
        ckey = Q.cache_key(
            extract_hash(extract), P6_PROMPT, MODEL_ID, RUN_ID, repeat_idx
        )
        prior = Q.lookup(CACHE_DIR, ckey)
        if prior is not None:
            sealed.append(prior)
            cached_hits += 1
            if prior.get("verdict") is not None:
                successes += 1
            else:
                failures_after += 1
            continue
        if args.dry_run:
            sealed.append({"page_id": page_id, "dry_run": True})
            continue

        page_host = urllib.parse.urlsplit(row["url"]).netloc.split(":")[0].lower()
        result: dict | None = None
        while result is None and attempts < max_attempts:
            attempts += 1
            fresh_calls += 1
            _, judgment = judge(
                key, page_host, to_model_text(extract), prompt_version=P6_PROMPT
            )
            if judgment.ok and judgment.parsed is not None:
                assert set(judgment.parsed) == required, sorted(judgment.parsed)
                result = {
                    "page_id": page_id,
                    "url": row["url"],
                    "snapshot_hash": extract_hash(extract),
                    "cache_key": ckey,
                    "prompt_version": P6_PROMPT,
                    "repeat_idx": repeat_idx,
                    "replay_of_run": record["_source_run"],
                    "verdict": judgment.parsed.get("verdict"),
                    "parsed": judgment.parsed,
                    "system_fingerprint": judgment.fingerprint,
                    "status": judgment.status,
                    "error": judgment.error,
                }
                successes += 1
            elif judgment.status == 429:
                rate_limited += 1
                backoff = max(getattr(judgment, "retry_after", 30.0), 30.0)
                print(f"rate limited (429); backing off {backoff:.0f}s", flush=True)
                time.sleep(backoff)
            else:
                result = {
                    "page_id": page_id,
                    "url": row["url"],
                    "snapshot_hash": extract_hash(extract),
                    "cache_key": ckey,
                    "prompt_version": P6_PROMPT,
                    "repeat_idx": repeat_idx,
                    "replay_of_run": record["_source_run"],
                    "verdict": None,
                    "status": judgment.status,
                    "error": judgment.error or "unparsed",
                }
                failures_after += 1
        if result is not None:
            Q.store(CACHE_DIR, ckey, result)
            sealed.append(result)
            print(
                f"[{len(sealed)}/{len(failures)}] {page_id} "
                f"({record['prompt_version']}/r{repeat_idx}): "
                f"{result.get('verdict') or 'FAIL ' + str(result.get('status'))}",
                flush=True,
            )
            time.sleep(interval)

    if args.dry_run:
        print(f"dry run: verified {len(sealed)} replays")
        return 0

    if len(sealed) < len(failures):
        print(
            f"incomplete: {len(sealed)}/{len(failures)} (rate limited "
            f"{rate_limited}x). Re-run to resume from the cache.",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calls.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in sealed), encoding="utf-8"
    )
    summary = {
        "run_id": RUN_ID,
        "run_class": "registered",
        "model": MODEL_ID,
        "prompt_version": P6_PROMPT,
        "schema": "p6-v1",
        "n_replayed": len(sealed),
        "n_success": successes,
        "n_failed": failures_after,
        "n_cached": cached_hits,
        "n_fresh": fresh_calls,
        "n_rate_limited": rate_limited,
        "before_error_rate": 1.0,
        "after_error_rate": failures_after / len(sealed) if sealed else None,
        "source_failures": len(failures),
    }
    (OUT_DIR / "run.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"done: {successes}/{len(sealed)} parsed under p6-v1; "
        f"after error rate {summary['after_error_rate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
