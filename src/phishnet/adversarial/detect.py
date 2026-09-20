"""Phase 5 injection detector (prereg §5.1, frozen in commit 1).

Pure function over the extract — no model call. Case-insensitive regex list;
a hit on any of `visible_text`, `title`, `meta_description`, `image_alt_text`
flags the page. The mapping of a hit (retain vs escalate) is NOT decided here:
both arms are reported per §5.1.
"""

from __future__ import annotations

import re

PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "ignore-instructions",
        r"ignore\s+((all|any|prior|previous|above|the|your)\s+)*instruc\w*",
    ),
    (
        "classify-benign",
        r"classify\s+this\s+page\s+as\s+(benign|safe|legitimate|trust\w*)",
    ),
    ("do-not-flag", r"do\s+not\s+(flag|report|block|mark)"),
    (
        "role-claim",
        r"you\s+are\s+(an?\s+)?(ai|assistant|classifier|reviewer|language\s+model)",
    ),
    ("system-marker", r"\[?\s*system\s*\]?[\s:]+"),
    ("developer-marker", r"developer\s*:"),
    ("fake-verdict", r'"verdict"\s*:\s*"?(phishing|benign|suspicious)"?'),
)

COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in PATTERNS
)

FIELDS: tuple[str, ...] = (
    "visible_text",
    "title",
    "meta_description",
    "image_alt_text",
)


def _field_text(extract: dict, field: str) -> str:
    value = extract.get(field, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def detect(extract: dict) -> dict:
    """Return `{"hit": bool, "hits": [{"field", "pattern", "match"}]}`."""
    hits = []
    for field in FIELDS:
        text = _field_text(extract, field)
        for name, rx in COMPILED:
            match = rx.search(text)
            if match:
                hits.append(
                    {"field": field, "pattern": name, "match": match.group(0)[:120]}
                )
    return {"hit": bool(hits), "hits": hits}
