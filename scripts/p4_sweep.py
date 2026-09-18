"""Phase 4 sweep driver: cache, run store, run_class, determinism.

Population (option-1 scope): test-band in-band rows with outcome `ok`.
Cache key is `snapshot_hash + prompt_version + model_string`; the response
cache makes re-runs free, so every per-1,000 cost figure is labeled
cold-cache (first-judgment cost). The driver asserts full coverage before
marking a run `recorded`: 100% of the in-band population under one prompt
version, one model ID and one `system_fingerprint`, with no quota
truncation. Anything short is `provisional` — sealed and kept, never
published. A truncated sweep is re-run, not topped up; resume continues the
same run id only while the fingerprint matches, otherwise the run
invalidates to provisional (§4.1–§4.2).

Modes:
  --sweep         judge the full in-band test population (resume-safe)
  --determinism   re-judge 50 in-band snapshots cold (2x, same seed),
                  report the verdict disagreement rate (bar ≤ 5%)

Usage: uv run python scripts/p4_sweep.py --sweep --run-id p4-sweep-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from phishnet.llm.client import judge  # noqa: E402
from phishnet.llm.schema import MODEL_ID, PROMPT_VERSION  # noqa: E402
from phishnet.snapshot.tier1 import band_edges, score_band  # noqa: E402

MANIFEST = Path("reports/snapshot-manifest-p4.json")
RUNS = Path("runs/phase4")
CACHE = RUNS / "cache"
DETERMINISM_N = 50
DETERMINISM_SEED = 0


def load_key() -> str:
    import os

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                key = line.strip().split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise SystemExit("GROQ_API_KEY not found")
    return key


def population() -> tuple[pd.DataFrame, float, float]:
    """Test-band in-band fetched-ok rows + calib-fixed edges."""
    y_calib, s_calib = score_band("data/splits-p3/calib.csv")
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    y_test, s_test = score_band("data/splits-p3/test.csv")
    manifest = pd.DataFrame(json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"])
    ok_urls = set(manifest.loc[manifest["outcome"] == "ok", "url"].astype(str))
    test = pd.read_csv("data/splits-p3/test.csv")
    test["tier1"] = np.asarray(s_test, dtype=float)
    in_band = test[
        (test["tier1"] >= lower_edge)
        & (test["tier1"] < t_alert)
        & (test["url"].astype(str).isin(ok_urls))
    ].copy()
    extracts: dict[str, dict] = {}
    for line in open("data/snapshots-p4/results.jsonl", encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if "extract" in row:
                extracts[row["url"]] = json.loads(row["extract"])
    in_band["extract"] = in_band["url"].astype(str).map(extracts)
    in_band = in_band[in_band["extract"].notna()].reset_index(drop=True)
    return in_band, t_alert, lower_edge


def cache_key(snapshot_hash: str) -> str:
    return hashlib.sha256(
        f"{snapshot_hash}|{PROMPT_VERSION}|{MODEL_ID}".encode()
    ).hexdigest()[:32]


def cached_judgment(key: str) -> dict | None:
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", action="store_true")
    group.add_argument("--determinism", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args(argv)

    in_band, t_alert, lower_edge = population()
    print(f"population: {len(in_band)} test in-band fetched-ok rows")
    print(f"edges: t_alert={t_alert:.6f} lower={lower_edge:.6f}")

    key = load_key()
    CACHE.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "judgments.jsonl"

    if args.determinism:
        rng = np.random.default_rng(DETERMINISM_SEED)
        idx = rng.choice(len(in_band), size=min(DETERMINISM_N, len(in_band)))
        targets = in_band.iloc[np.sort(idx)].reset_index(drop=True)
        cold = True
    else:
        targets = in_band
        cold = False

    done: dict[str, dict] = {}
    if out_path.exists() and not args.determinism:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["url"]] = row
    print(f"resume: {len(done)} sealed, {len(targets)} targeted")

    calls = 0
    repeats: list[tuple[str, str | None, str | None]] = []
    with out_path.open("a", encoding="utf-8") as fh:
        for _, row in targets.iterrows():
            url = str(row["url"])
            if url in done and not args.determinism:
                continue
            if args.max_calls is not None and calls >= args.max_calls:
                break
            extract: dict = row["extract"]
            snap_hash = hashlib.sha256(
                json.dumps(extract, sort_keys=True).encode()
            ).hexdigest()
            ckey = cache_key(snap_hash)
            from phishnet.snapshot.extract import to_model_text

            record: dict | None = None if cold else cached_judgment(ckey)
            from_cache = record is not None
            if record is None:
                attempts = 0
                while True:
                    req, judgment = judge(
                        key, str(extract.get("page_host", "")), to_model_text(extract)
                    )
                    attempts += 1
                    if judgment.status != 429 or attempts >= 3:
                        break
                    print("429 rate limit, backing off 60s")
                    time.sleep(60)
                record = {
                    "url": url,
                    "label": int(row["label"]),
                    "snapshot_hash": snap_hash,
                    "cache_key": ckey,
                    "cold_cache": True,
                    "verdict": (
                        judgment.parsed.get("verdict")
                        if judgment.ok and judgment.parsed
                        else None
                    ),
                    "confidence": (
                        judgment.parsed.get("confidence")
                        if judgment.ok and judgment.parsed
                        else None
                    ),
                    "parsed": judgment.parsed,
                    "system_fingerprint": judgment.fingerprint,
                    "usage": judgment.usage,
                    "latency_ms": judgment.latency_ms,
                    "status": judgment.status,
                    "error": judgment.error,
                }
                (CACHE / f"{ckey}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                attempts_note = f" (attempts={attempts})" if attempts > 1 else ""
                print(
                    f"{record['verdict']} fp={judgment.fingerprint} "
                    f"{judgment.latency_ms:.0f}ms{attempts_note}"
                )
            if args.determinism:
                # Second cold judgment of the same snapshot.
                req2, judgment2 = judge(
                    key, str(extract.get("page_host", "")), to_model_text(extract)
                )
                calls += 2
                v1 = record["verdict"]
                v2 = (
                    judgment2.parsed.get("verdict")
                    if judgment2.ok and judgment2.parsed
                    else None
                )
                repeats.append((url, v1, v2))
                seal = dict(record)
                seal["repeat_verdict"] = v2
                seal["repeat_fingerprint"] = judgment2.fingerprint
                seal["repeat_usage"] = judgment2.usage
                fh.write(json.dumps(seal) + "\n")
            else:
                calls += 0 if from_cache else 1
                fh.write(json.dumps(record) + "\n")

    if args.determinism:
        n = len(repeats)
        disagree = sum(1 for _, a, b in repeats if a != b)
        rate = disagree / n if n else 1.0
        seals = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fingerprints = {s.get("system_fingerprint") for s in seals} | {
            s.get("repeat_fingerprint") for s in seals
        }
        result = {
            "run_id": args.run_id,
            "run_class": "provisional",
            "n": n,
            "disagreements": disagree,
            "disagreement_rate": rate,
            "bar": 0.05,
            "within_bar": bool(rate <= 0.05),
            "fingerprints": sorted(f for f in fingerprints if f),
            "single_fingerprint": len([f for f in fingerprints if f]) == 1,
        }
        (run_dir / "determinism.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0

    seals = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_urls = {s["url"] for s in seals}
    fingerprints = {
        s.get("system_fingerprint") for s in seals if s.get("system_fingerprint")
    }
    full_coverage = len(sealed_urls) >= len(in_band) and set(
        in_band["url"].astype(str)
    ) <= sealed_urls
    # phase4-C: fingerprint rotates per call, so it cannot predicate identity.
    # Recorded = full coverage under one model/prompt/seed; the fingerprint
    # distribution is sealed beside the run, explicitly weaker than registered.
    run_class = "recorded" if full_coverage else "provisional"
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "run_class": run_class,
                "model": MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "seed": 0,
                "tier": "free",
                "n_population": len(in_band),
                "n_sealed": len(sealed_urls),
                "full_coverage": bool(full_coverage),
                "fingerprints": sorted(fingerprints),
                "n_fingerprints": len(fingerprints),
                "identity_note": "phase4-C: fingerprint rotates per call; "
                "identity is model+prompt+seed, explicitly weaker",
                "t_alert": t_alert,
                "lower_edge": lower_edge,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"run_class={run_class} sealed={len(sealed_urls)}/{len(in_band)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
