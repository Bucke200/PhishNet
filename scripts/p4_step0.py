"""Step-0 driver: fetch_set → fetch once → manifest → table → trigger.

Reads the calib-fixed edges (recomputed from the pinned Tier-1 path, never
hard-coded), builds `fetch_set = step0_sample ∪ in_band(calib) ∪
in_band(test)`, fetches each URL exactly once (resume-safe: rows already in
`data/snapshots-p4/results.jsonl` are reused, never re-fetched), writes
bodies under `data/snapshots-p4/` (gitignored) and the manifest
`reports/snapshot-manifest-p4.json` (committed), then applies the pure
`trigger.py` verdict. Prints marginals; the verdict is recorded as `phase4-B`.

Usage: uv run python scripts/p4_step0.py [--workers 10] [--timeout 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from phishnet.snapshot.extract import (  # noqa: E402
    canonical_extract,
    canonical_json,
    extract_hash,
)
from phishnet.snapshot.fetch import fetch_once, to_record  # noqa: E402
from phishnet.snapshot.step0 import (  # noqa: E402
    build_fetch_set,
    manifest_hash,
    step0_table,
)
from phishnet.snapshot.trigger import trigger_verdict  # noqa: E402

SNAP_DIR = Path("data/snapshots-p4")
RESULTS = SNAP_DIR / "results.jsonl"
MANIFEST = Path("reports/snapshot-manifest-p4.json")
STEP0_OUT = Path("reports/phase4-step0.json")


def _body_path(raw_hash: str) -> Path:
    return SNAP_DIR / f"{raw_hash}.html"


def _load_done() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["url"]] = row
    return done


def _fetch_row(url: str, timeout: int) -> dict:
    fetched = fetch_once(url, timeout=timeout)
    raw_hash = fetched.raw_hash()
    if fetched.raw_html:
        _body_path(raw_hash).write_bytes(fetched.raw_html)
        try:
            text = fetched.raw_html.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        extract = canonical_extract(text, fetched.final_url or url)
    else:
        extract = canonical_extract("", fetched.final_url or url)
    record = to_record(fetched, extract_hash(extract))
    record["extract"] = canonical_json(extract)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args(argv)

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    fetch_set, meta = build_fetch_set()
    print(f"edges: t_alert={meta['t_alert']:.6f} lower={meta['lower_edge']:.6f}")
    print(
        f"fetch_set: {meta['n_fetch_set']} "
        f"(sample {meta['n_step0_sample']}, in-band {meta['n_in_band']})"
    )

    done = _load_done()
    todo = [u for u in fetch_set["url"].astype(str) if u not in done]
    print(f"resume: {len(done)} already fetched, {len(todo)} to fetch")
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_url = {
                pool.submit(_fetch_row, url, args.timeout): url for url in todo
            }
            with RESULTS.open("a", encoding="utf-8") as fh:
                for i, future in enumerate(as_completed(future_to_url)):
                    url = future_to_url[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "url": url,
                            "outcome": "timeout",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    done[url] = record
                    fh.write(json.dumps(record) + "\n")
                    if (i + 1) % 200 == 0:
                        print(f"  fetched {i + 1}/{len(todo)}")
        print(f"fetched {len(todo)} rows")

    manifest_rows = []
    for _, row in fetch_set.iterrows():
        rec = dict(done.get(str(row["url"]), {"url": row["url"], "outcome": "timeout"}))
        rec.update(
            {
                "label": int(row["label"]),
                "era": str(row["era"]),
                "survival_stratum": str(row.get("survival_stratum", "unknown")),
                "source": str(row.get("source", "")),
                "tier1_score": (
                    float(row["tier1_score"])
                    if pd.notna(row.get("tier1_score"))
                    else None
                ),
            }
        )
        manifest_rows.append(rec)

    manifest = {
        "edges": meta,
        "manifest_hash": manifest_hash(
            [{k: r.get(k) for k in sorted(r) if k != "extract"} for r in manifest_rows]
        ),
        "rows": [
            {k: v for k, v in r.items() if k != "extract"} for r in manifest_rows
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    table = step0_table(pd.DataFrame(manifest_rows))
    verdict = trigger_verdict(pd.DataFrame(manifest_rows))
    STEP0_OUT.write_text(
        json.dumps(
            {
                "edges": meta,
                "marginals": verdict,
                "cells": table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2))
    outcomes = pd.DataFrame(manifest_rows)["outcome"].value_counts()
    print(outcomes.to_string())
    print(f"manifest -> {MANIFEST} ({len(manifest_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
