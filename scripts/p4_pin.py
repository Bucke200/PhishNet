"""Pin sealed Phase 4 artifacts in `repro/hashes-p4.json`.

Hashes every sealed file (gate, determinism, provisional sweep, cache,
snapshot manifest, Step-0 table, baseline reference, close-out report) plus
a directory digest. Read-only over the run store; writes only the pin file.
Only sealed runs are pinnable (§5.2) — nothing here publishes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("repro/hashes-p4.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    pins: dict[str, str] = {}
    roots = [
        Path("runs/phase4/p4-gate"),
        Path("runs/phase4/p4-determinism-1"),
        Path("runs/phase4/p4-sweep-1"),
        Path("runs/phase4/cache"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    files.extend(
        [
            Path("reports/snapshot-manifest-p4.json"),
            Path("reports/phase4-step0.json"),
            Path("reports/phase4-baseline-ref.json"),
            Path("reports/phase4.json"),
            Path("reports/phase4.md"),
        ]
    )
    for path in files:
        pins[path.as_posix()] = sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode("utf-8")
    ).hexdigest()
    OUT.write_text(
        json.dumps({"files": pins, "digest": digest, "n": len(pins)}, indent=2),
        encoding="utf-8",
    )
    print(f"pinned {len(pins)} files -> {OUT} (digest {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
