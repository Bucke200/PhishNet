"""Verify a rebuilt evaluation population against the pinned hash table.

Compares every file listed in ``repro/hashes.json`` (CRLF-canonical bytes)
with the corresponding file in the rebuilt directory. ``run-meta.json`` is
intentionally unpinned: it carries the volatile run timestamp by design.

Usage:
    python repro/verify.py --hashes repro/hashes.json --dir $OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hashes", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    a = p.parse_args(argv)

    expected: dict[str, str] = json.loads(a.hashes.read_text(encoding="utf-8"))
    failures = 0
    for name in sorted(expected):
        target = a.dir / name
        if not target.is_file():
            print(f"MISSING  {name}")
            failures += 1
            continue
        got = sha256(target)
        ok = got == expected[name]
        print(f"{'OK       ' if ok else 'MISMATCH '} {name} {got[:12]}")
        failures += not ok
    unpinned = sorted(
        f.name for f in a.dir.iterdir() if f.is_file() and f.name not in expected
    )
    if unpinned:
        print(f"(unpinned, not checked: {', '.join(unpinned)})")
    print(f"{len(expected) - failures}/{len(expected)} pinned files match")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
