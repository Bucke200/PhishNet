"""Phase 5 response cache (the `phase4-D` defect fixed by construction).

Key: `snapshot_hash + prompt_version + model + run_id + repeat_idx`, so
repeats 2–3 can never return repeat 1's cached responses. The directory
defaults to `runs/phase5/cache`, disjoint from Phase 4's, so no Phase 4 seal
— provisional, determinism, or otherwise — can seed a Phase 5 cache. Pure
storage: the driver decides what a sealed call is; this module only files it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PHASE5_CACHE_DIR = Path("runs/phase5/cache")


def cache_key(
    snapshot_hash: str,
    prompt_version: str,
    model: str,
    run_id: str,
    repeat_idx: int,
) -> str:
    """Full sha256 hex over the five key fields (no truncation)."""
    return hashlib.sha256(
        f"{snapshot_hash}|{prompt_version}|{model}|{run_id}|{repeat_idx}".encode()
    ).hexdigest()


def lookup(cache_dir: Path, key: str) -> dict | None:
    """Return the sealed record, or None on a cold cache."""
    path = cache_dir / f"{key}.json"
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    return None


def store(cache_dir: Path, key: str, record: dict) -> None:
    """Seal one record (overwrites nothing: keys are unique per repeat)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")
