"""Collection guard: the golden marker set must be non-empty.

``pytest -m golden`` exits 5 when nothing matches, which is easy to miss;
this asserts explicitly so moving or renaming the golden test files fails
loudly instead of quietly running the base suite only.

Usage:
    python repro/check_golden.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "golden",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    collected = [ln for ln in proc.stdout.splitlines() if "::" in ln]
    print(f"golden tests collected: {len(collected)}")
    for ln in collected:
        print(f"  {ln}")
    if not collected:
        print("FAIL: golden marker set is empty", file=sys.stderr)
        return 1
    print("OK: golden set non-empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
