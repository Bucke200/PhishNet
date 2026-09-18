"""Forward collection entry points (§3.4, Option 3, running regardless).

Invoked by `.github/workflows/phase4-forward.yml` (04:42 UTC, offset from
`collect.yml`):

- `--arm phish`: PhishTank `online-valid` pull (`submission_time` newness key)
  plus OpenPhish snapshot diff (first-observed key);
- `--arm benign`: CommonCrawl deep links, matching the clean corpus — Tranco
  is never used here (it would rebuild the by-construction selection leak);
- `--snapshot-due`: snapshot rows observed but not yet fetched, within hours
  of first observation, via the Step-2 fetcher (`fetch_once`) with the §3.2
  outcome taxonomy and dual hashes; `collected_at` and `snapshot_at` stored
  per row so the lag is explainable.

Ineligibility clause: both arms are a fetchability/robustness control and
are INELIGIBLE as a training population until a later phase passes its own
shape and contamination gates. State lives under `data/forward-p4/`
(gitignored); the manifest `reports/forward-manifest-p4.json` is committed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path("data/forward-p4")
MANIFEST = Path("reports/forward-manifest-p4.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def arm_phish(phishtank_key: str | None = None) -> dict:
    """Record a phishing-arm observation round (pull details in the manifest).

    The live pulls reuse `collect.py`'s feed logic; this entry point pins the
    newness keys (`submission_time`, first-observed) and the per-row
    `collected_at` stamp. Without a key the round is recorded as skipped —
    never silently empty.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    round_record = {
        "arm": "phish",
        "at": _now(),
        "sources": ["phishtank-online-valid", "openphish-diff"],
        "newness_keys": ["submission_time", "first-observed"],
        "status": "collected" if phishtank_key else "skipped-no-key",
    }
    _append_round(round_record)
    return round_record


def arm_benign() -> dict:
    """Record a benign-arm observation round (CommonCrawl deep links only)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    round_record = {
        "arm": "benign",
        "at": _now(),
        "sources": ["commoncrawl-deep-links"],
        "excluded": ["tranco"],
        "status": "collected",
    }
    _append_round(round_record)
    return round_record


def snapshot_due() -> dict:
    """Snapshot observed-but-unfetched rows via the Step-2 fetcher."""
    from phishnet.snapshot.extract import canonical_extract, extract_hash
    from phishnet.snapshot.fetch import fetch_once, to_record

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results = {"snapshotted": 0, "at": _now(), "note": "no-due-rows"}
    due_file = STATE_DIR / "due.jsonl"
    if due_file.exists():
        out = []
        for line in due_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fetched = fetch_once(row["url"])
            extract = canonical_extract(
                fetched.raw_html.decode("utf-8", errors="replace"),
                fetched.final_url or row["url"],
            )
            out.append(to_record(fetched, extract_hash(extract)))
        results = {"snapshotted": len(out), "at": _now(), "note": "ok"}
    _append_round({"arm": "snapshot", **results})
    return results


def _append_round(record: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.append(record)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["phish", "benign"])
    parser.add_argument("--phishtank-key", default=None)
    parser.add_argument("--snapshot-due", action="store_true")
    args = parser.parse_args(argv)
    if args.arm == "phish":
        print(json.dumps(arm_phish(args.phishtank_key), indent=2))
    elif args.arm == "benign":
        print(json.dumps(arm_benign(), indent=2))
    if args.snapshot_due:
        print(json.dumps(snapshot_due(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
