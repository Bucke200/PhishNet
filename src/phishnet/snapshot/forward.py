"""Forward collection entry points (§3.4, Option 3; `phase4-E` Minimal).

Invoked by `.github/workflows/phase4-forward.yml` (04:42 UTC, offset from
`collect.yml`), checking out a tag (never a branch) so the fetcher, outcome
taxonomy and hashes are always the Step-0 ones.

- `--arm phish`: OpenPhish-only pull (no key needed), diffed against the
  seen set with first-observed as the newness key. PhishTank is out:
  registration is closed and no key exists (`phase4-E`). New URLs queue in
  `due.jsonl` with `collected_at`.
- `--arm benign`: deferred — the CommonCrawl deep-link pull is not
  implemented (`phase4-E`). Records intent only, never a collection.
- `--snapshot-due`: snapshot queued rows via the Step-2 fetcher
  (`fetch_once`) with the §3.2 outcome taxonomy and dual hashes; bodies
  under `data/forward-p4/bodies/`, per-row records appended to
  `snapshots.jsonl`. `collected_at` and `snapshot_at` are stored per row so
  the lag is explainable.

Ineligibility clause: both arms are a fetchability/robustness control and
are INELIGIBLE as a training population until a later phase passes its own
shape and contamination gates. State lives under `data/forward-p4/`
(gitignored worktree, force-added on the data branch); the round manifest
`reports/forward-manifest-p4.json` is committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path("data/forward-p4")
SEEN_FILE = STATE_DIR / "openphish_seen.json"
DUE_FILE = STATE_DIR / "due.jsonl"
SNAPS_FILE = STATE_DIR / "snapshots.jsonl"
BODIES_DIR = STATE_DIR / "bodies"
MANIFEST = Path("reports/forward-manifest-p4.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_seen() -> set[str]:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    return set()


def _save_seen(seen: set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(seen), indent=1), encoding="utf-8")


def arm_phish() -> dict:
    """OpenPhish pull, first-observed diff, queue the new rows."""
    sys.path.insert(0, ".")
    from collect import fetch_openphish

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        feed = fetch_openphish(_now()[:10])
    except Exception as exc:
        round_record = {
            "arm": "phish",
            "at": _now(),
            "sources": ["openphish-diff"],
            "status": "fetch-failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _append_round(round_record)
        return round_record
    seen = _load_seen()
    new_rows = []
    for row in feed:
        url = str(row.get("url", ""))
        if url and url not in seen:
            seen.add(url)
            new_rows.append(
                {
                    "url": url,
                    "first_seen": row.get("first_seen"),
                    "source": "openphish",
                    "collected_at": _now(),
                }
            )
    _save_seen(seen)
    if new_rows:
        with DUE_FILE.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    round_record = {
        "arm": "phish",
        "at": _now(),
        "sources": ["openphish-diff"],
        "newness_keys": ["first-observed"],
        "n_feed": len(feed),
        "n_new": len(new_rows),
        "status": "collected",
    }
    _append_round(round_record)
    return round_record


def arm_benign() -> dict:
    """Deferred: the CC deep-link pull is not implemented (`phase4-E`)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    round_record = {
        "arm": "benign",
        "at": _now(),
        "sources": ["commoncrawl-deep-links"],
        "excluded": ["tranco"],
        "status": "deferred",
        "note": "CC deep-link pull not implemented; see phase4-E",
    }
    _append_round(round_record)
    return round_record


def snapshot_due() -> dict:
    """Snapshot queued rows; persist bodies + per-row records."""
    from phishnet.snapshot.extract import canonical_extract, extract_hash
    from phishnet.snapshot.fetch import fetch_once, to_record

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    BODIES_DIR.mkdir(parents=True, exist_ok=True)
    if not DUE_FILE.exists():
        results: dict = {"snapshotted": 0, "at": _now(), "note": "no-due-rows"}
        _append_round({"arm": "snapshot", **results})
        return results
    outcomes: dict[str, int] = {}
    n = 0
    with SNAPS_FILE.open("a", encoding="utf-8") as out:
        for line in DUE_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fetched = fetch_once(row["url"])
            if fetched.raw_html:
                (BODIES_DIR / f"{fetched.raw_hash()}.html").write_bytes(
                    fetched.raw_html
                )
            try:
                text = fetched.raw_html.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            extract = canonical_extract(text, fetched.final_url or row["url"])
            record = to_record(fetched, extract_hash(extract))
            record["collected_at"] = row.get("collected_at")
            record["snapshot_at"] = fetched.fetched_at
            out.write(json.dumps(record, sort_keys=True) + "\n")
            outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1
            n += 1
    DUE_FILE.write_text("", encoding="utf-8")
    results = {
        "snapshotted": n,
        "outcomes": outcomes,
        "at": _now(),
        "note": "ok",
    }
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
    parser.add_argument("--snapshot-due", action="store_true")
    args = parser.parse_args(argv)
    if args.arm == "phish":
        print(json.dumps(arm_phish(), indent=2))
    elif args.arm == "benign":
        print(json.dumps(arm_benign(), indent=2))
    if args.snapshot_due:
        print(json.dumps(snapshot_due(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
