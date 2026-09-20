"""Phase 7 packaging invariants (P7-1, P7-2, P7-3).

Pins the README's fixed-threshold claims and bans the retired-system
phrasings, checks the model card carries every roadmap item, and verifies
every relative Markdown link in `README.md` and `docs/` resolves on disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MODEL_CARD = ROOT / "docs" / "model-card.md"
DOCS = ROOT / "docs"

# P7-1: values the README must carry, each traced in reports/phase7.md.
REQUIRED_README = (
    "50.4%",
    "0.40%",
    "60.7%",
    "0.98%",
    "53.4%",
    "0.45 ms",
    "0.0347",
    "webflow.io",
    "not a safety guarantee",
)

# P7-1: retired-system phrasings that must not come back.
BANNED_README = (
    "phishnet-pavv.onrender.com",
    "serves `phishnet.api`",
    '"explain": true',
    "prediction` (0 = legit",
)

# P7-2: every roadmap-listed model-card item.
REQUIRED_MODEL_CARD = (
    "Inverted depth prior",
    "canonicalize_scheme",
    "Point-in-time classification",
    "Survivorship",
    "RDAP 404",
    "Tranco selection leak",
    "Hosted coverage limits",
    "Age: gate failure and the conditional result",
    "Cold start",
    "Threshold transfer",
    "Why CT was dropped",
    "Fetchability is a label proxy",
    "system_fingerprint",
    "22%",
    "0.0347",
)

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

DOC_FILES = [README, *sorted(DOCS.glob("*.md"))]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_required_claims_present() -> None:
    text = _read(README)
    missing = [claim for claim in REQUIRED_README if claim not in text]
    assert not missing, missing


def test_readme_stale_claims_absent() -> None:
    text = _read(README)
    present = [claim for claim in BANNED_README if claim in text]
    assert not present, present


def test_model_card_is_complete() -> None:
    text = _read(MODEL_CARD)
    missing = [item for item in REQUIRED_MODEL_CARD if item not in text]
    assert not missing, missing


@pytest.mark.parametrize("path", DOC_FILES, ids=[p.name for p in DOC_FILES])
def test_relative_markdown_links_resolve(path: Path) -> None:
    broken = []
    for target in LINK.findall(_read(path)):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if relative and not (path.parent / relative).resolve().exists():
            broken.append(target)
    assert not broken, broken
