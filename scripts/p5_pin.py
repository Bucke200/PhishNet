"""Pin sealed Phase 5 artifacts in `repro/hashes-p5.json`.

Hashes every sealed Phase 5 run (gate, gate-probe, dev runs, eval runs, cache),
adversarial manifests, aware log, lexical evaluation, prompt, and close-out reports
plus a directory digest. Read-only over the run store; writes only the pin file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("repro/hashes-p5.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    pins: dict[str, str] = {}
    roots = [
        Path("runs/phase5"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    files.extend(
        [
            Path("reports/adversarial-manifest-p5.json"),
            Path("reports/adversarial-aware-log.json"),
            Path("reports/phase5-lexical.json"),
            Path("reports/phase5-lexical.md"),
            Path("reports/phase5-adversarial.json"),
            Path("reports/phase5-adversarial.md"),
            Path("src/phishnet/llm/prompts/p5-h1.txt"),
        ]
    )
    for path in sorted(files, key=lambda p: p.as_posix()):
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
