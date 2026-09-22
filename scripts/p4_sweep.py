"""Phase 4 sweep driver: cache, run store, run_class, repeats, budget.

Population (option-1 scope): test-band in-band rows with outcome `ok` (1,106).

Cache: keyed `snapshot_hash + prompt_version + model + run_id + repeat_idx`
via `phishnet.llm.cache` — the `phase4-D` defect fixed by construction, so
no provisional or determinism seal can seed a recorded run and repeats 2-3
are real calls rather than repeat 1's responses.

Run class: the driver asserts full coverage before marking a run `recorded`:
100% of the in-band population for every requested repeat, one prompt
version, one model ID, no quota truncation. Anything short is `provisional`
— sealed and kept, never published. A truncated sweep is re-run, not topped
up (phase4-preregistration.md 4.1-4.2).

Spend: every call is guarded by `phishnet.llm.budget` — a cumulative on-disk
ledger with an optional cap (`PHISHNET_LLM_BUDGET_USD`), a STOP sentinel and
an exclusive run lock. The guard refuses *before* a request is sent, so a
rate-limit day or an agent relaunch cannot boomerang the card.

Modes:
  --sweep         judge the full in-band test population (resume-safe)
  --determinism   re-judge 50 in-band snapshots cold (2x, same seed),
                  report the verdict disagreement rate (bar <= 5%)

Usage:
  uv run python scripts/p4_sweep.py --sweep --run-id p4-sweep-1
  uv run python scripts/p4_sweep.py --sweep --run-id p4-recorded --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from phishnet.llm import budget  # noqa: E402
from phishnet.llm.cache import cache_key, lookup, store  # noqa: E402
from phishnet.llm.client import Judgment, judge  # noqa: E402
from phishnet.llm.schema import MODEL_ID, PROMPT_VERSION  # noqa: E402
from phishnet.snapshot.tier1 import band_edges, score_band  # noqa: E402

MANIFEST = Path("reports/snapshot-manifest-p4.json")
RUNS = Path("runs/phase4")
CACHE = RUNS / "cache"
DETERMINISM_N = 50
DETERMINISM_SEED = 0
EXPECTED_POPULATION = 1106
VALID_SEAL_STATUSES = (200, 400)
# Estimation only (measured mean over the sealed cache): dry-run projects
# spend before anything is sent.
EST_PROMPT_TOKENS = 1432
EST_COMPLETION_TOKENS = 245
LOCK_NAME = "phase4-run"


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


def hash_extract(extract: object) -> str:
    return hashlib.sha256(json.dumps(extract, sort_keys=True).encode()).hexdigest()


def cache_key_for(url: str, snap_hash: str, run_id: str, repeat_idx: int) -> str:
    """URL-scoped cache key.

    The snapshot hash alone is not row identity: distinct URLs can resolve to
    the same landing page and therefore the same extract (measured: 83 of
    1,106 rows per repeat). Hashing `url|snapshot` keeps those rows distinct
    so neither can return the other's cached judgment.
    """
    return str(
        cache_key(f"{url}|{snap_hash}", PROMPT_VERSION, MODEL_ID, run_id, repeat_idx)
    )


def is_valid_seal(record: dict) -> bool:
    """A judged row. Only 200 (judged) and 400 (provider schema refusal,
    sealed per 4.2 and retaining Tier-1) count toward coverage. A 429 is a
    quota failure, a transport error is `status -1`, and a 5xx is provider
    trouble: none are judgments, so none count and all are re-called on
    resume (4.1: no truncation)."""
    return int(record.get("status", 0)) in VALID_SEAL_STATUSES


def sealed_index(path: Path) -> dict[int, set[str]]:
    """URLs already sealed per repeat index (missing index reads as 0)."""
    sealed: dict[int, set[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not is_valid_seal(row):
                    continue
                idx = int(row.get("repeat_idx", 0))
                sealed.setdefault(idx, set()).add(str(row["url"]))
    return sealed


def projection(
    in_band: pd.DataFrame, run_id: str, repeats: int
) -> tuple[int, int, float]:
    """(calls to make, total calls, estimated USD) with no API traffic."""
    total = 0
    pending = 0
    for repeat_idx in range(repeats):
        for _, row in in_band.iterrows():
            total += 1
            url = str(row["url"])
            snap = hash_extract(row["extract"])
            key = cache_key_for(url, snap, run_id, repeat_idx)
            cached = lookup(CACHE, key)
            if cached is None or not is_valid_seal(cached):
                pending += 1
    est_usd = pending * budget.estimate_usd(EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS)
    return pending, total, est_usd


def judge_with_backoff(
    key: str, extract: dict[str, object]
) -> tuple[dict[str, object], Judgment, int]:
    """One judged call with bounded retries.

    Retries: 429 waits 60 s (the registered quota backoff); a transport
    failure (`status -1`) or a 5xx waits 5 s. Both are capped at 3 attempts,
    so a single provider blip cannot spoil an otherwise-complete paid sweep.
    """
    from phishnet.snapshot.extract import to_model_text

    attempts = 0
    while True:
        req, judgment = judge(
            key, str(extract.get("page_host", "")), to_model_text(extract)
        )
        attempts += 1
        if attempts >= 3:
            return req, judgment, attempts
        if judgment.status == 429:
            print("429 rate limit, backing off 60s")
            time.sleep(60)
            continue
        if judgment.status < 0 or judgment.status >= 500:
            print(f"transient status={judgment.status}, retrying in 5s")
            time.sleep(5)
            continue
        return req, judgment, attempts


def run_determinism(
    in_band: pd.DataFrame, args: argparse.Namespace, run_dir: Path
) -> int:
    key = load_key()
    out_path = run_dir / "judgments.jsonl"
    rng = np.random.default_rng(DETERMINISM_SEED)
    idx = rng.choice(len(in_band), size=min(DETERMINISM_N, len(in_band)))
    targets = in_band.iloc[np.sort(idx)].reset_index(drop=True)
    repeats: list[tuple[str, str | None, str | None]] = []
    if not args.no_lock:
        budget.acquire_lock(LOCK_NAME)
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for _, row in targets.iterrows():
                url = str(row["url"])
                extract: dict = row["extract"]
                snap_hash = hash_extract(extract)
                ckey = cache_key_for(url, snap_hash, args.run_id, 0)
                _, judgment, attempts = judge_with_backoff(key, extract)
                record = {
                    "url": url,
                    "label": int(row["label"]),
                    "snapshot_hash": snap_hash,
                    "cache_key": ckey,
                    "repeat_idx": 0,
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
                store(CACHE, ckey, record)
                # Second cold judgment of the same snapshot.
                _, judgment2, _ = judge_with_backoff(key, extract)
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
                note = f" (attempts={attempts})" if attempts > 1 else ""
                print(f"{v1} -> {v2} fp={judgment.fingerprint}{note}")
    except budget.BudgetError as exc:
        print(f"budget stop: {exc}", file=sys.stderr)
        return 2
    finally:
        if not args.no_lock:
            budget.release_lock(LOCK_NAME)

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


def run_sweep(
    in_band: pd.DataFrame,
    args: argparse.Namespace,
    t_alert: float,
    lower_edge: float,
    run_dir: Path,
) -> int:
    repeats = max(1, args.repeats)
    out_path = run_dir / "judgments.jsonl"

    if args.dry_run:
        pending, total, est_usd = projection(in_band, args.run_id, repeats)
        print(f"dry-run: {pending} calls to make of {total} ({repeats} repeats)")
        print(
            f"dry-run: est {pending * EST_PROMPT_TOKENS:,} prompt + "
            f"{pending * EST_COMPLETION_TOKENS:,} completion tokens"
        )
        print(f"dry-run: est cost ${est_usd:.4f} (cap ${budget.cap_usd()})")
        print(json.dumps(budget.status(), indent=2, sort_keys=True))
        return 0

    key = load_key()
    print(json.dumps(budget.status(), indent=2, sort_keys=True))
    sealed = sealed_index(out_path)
    calls = 0
    stopped: str | None = None

    if not args.no_lock:
        budget.acquire_lock(LOCK_NAME)
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            for repeat_idx in range(repeats):
                done = sealed.get(repeat_idx, set())
                print(f"repeat {repeat_idx}: resume {len(done)} sealed")
                for _, row in in_band.iterrows():
                    url = str(row["url"])
                    if url in done:
                        continue
                    if args.max_calls is not None and calls >= args.max_calls:
                        print(f"max-calls {args.max_calls} reached; stopping")
                        break
                    extract: dict = row["extract"]
                    snap_hash = hash_extract(extract)
                    ckey = cache_key_for(url, snap_hash, args.run_id, repeat_idx)
                    record = lookup(CACHE, ckey)
                    if record is not None and not is_valid_seal(record):
                        record = None
                    if record is None:
                        _, judgment, attempts = judge_with_backoff(key, extract)
                        record = {
                            "url": url,
                            "label": int(row["label"]),
                            "snapshot_hash": snap_hash,
                            "cache_key": ckey,
                            "repeat_idx": repeat_idx,
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
                        store(CACHE, ckey, record)
                        calls += 1
                        note = f" (attempts={attempts})" if attempts > 1 else ""
                        print(
                            f"[r{repeat_idx}] {record['verdict']} "
                            f"fp={judgment.fingerprint} "
                            f"{judgment.latency_ms:.0f}ms{note}"
                        )
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    done.add(url)
                if args.max_calls is not None and calls >= args.max_calls:
                    break
    except budget.BudgetError as exc:
        stopped = str(exc)
        print(f"budget stop: {exc}", file=sys.stderr)
    finally:
        if not args.no_lock:
            budget.release_lock(LOCK_NAME)

    write_run_record(in_band, args, t_alert, lower_edge, run_dir, repeats, out_path)
    if stopped is not None:
        return 2
    return 0


def write_run_record(
    in_band: pd.DataFrame,
    args: argparse.Namespace,
    t_alert: float,
    lower_edge: float,
    run_dir: Path,
    repeats: int,
    out_path: Path,
) -> None:
    sealed = sealed_index(out_path)
    urls = set(in_band["url"].astype(str))
    coverage = {idx: len(sealed.get(idx, set())) for idx in range(repeats)}
    full_coverage = all(urls <= sealed.get(idx, set()) for idx in range(repeats))
    fingerprints: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fp = json.loads(line).get("system_fingerprint")
                if fp:
                    fingerprints.add(str(fp))
    run_class = "recorded" if full_coverage else "provisional"
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "run_class": run_class,
                "model": MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "seed": 0,
                "tier": args.tier,
                "n_repeats": repeats,
                "n_population": len(in_band),
                "n_sealed": coverage.get(0, 0),
                "repeat_coverage": coverage,
                "full_coverage": bool(full_coverage),
                "fingerprints": sorted(fingerprints),
                "n_fingerprints": len(fingerprints),
                "cache_key_scheme": "snapshot+prompt+model+run_id+repeat_idx",
                "identity_note": "phase4-C: fingerprint rotates per call; "
                "identity is model+prompt+seed, explicitly weaker",
                "budget_id": budget.budget_id(),
                "budget_cap_usd": budget.cap_usd(),
                "budget_est_usd": budget.status().get("est_usd"),
                "t_alert": t_alert,
                "lower_edge": lower_edge,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"run_class={run_class} coverage={coverage} population={len(in_band)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", action="store_true")
    group.add_argument("--determinism", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument(
        "--tier",
        default=os.environ.get("PHISHNET_LLM_TIER", "free"),
        choices=["free", "developer"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-population-drift", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)

    in_band, t_alert, lower_edge = population()
    print(f"population: {len(in_band)} test in-band fetched-ok rows")
    print(f"edges: t_alert={t_alert:.6f} lower={lower_edge:.6f}")
    if len(in_band) != EXPECTED_POPULATION and not args.allow_population_drift:
        print(
            f"refusing: population {len(in_band)} != registered "
            f"{EXPECTED_POPULATION}; pass --allow-population-drift to override",
            file=sys.stderr,
        )
        return 2

    CACHE.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.determinism:
        return run_determinism(in_band, args, run_dir)
    return run_sweep(in_band, args, t_alert, lower_edge, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
