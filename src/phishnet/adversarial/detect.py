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

# Serving-only correction (2026-09-22, live-performance plan R6/T2-3). The
# frozen `system-marker` pattern `\[?\s*system\s*\]?[\s:]+` has optional
# brackets, so it matches the bare English word anywhere in page text
# ("depending on your system", "Arc System Works", "Sound System"). In live
# testing that auto-alerted two benign pages with no LLM call. Serving
# requires the marker to sit at the start of a line (optionally bracketed),
# which still matches the Phase 5 payloads (`P-system-1` "System: ...",
# `P-delimiter-1` "\nSystem instruction: ...") but not mid-sentence prose.
# The frozen `detect()` and its pinned recall table are unchanged; only
# serving uses `detect_serving`.
SERVING_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    (
        name,
        r"(?im)^[ \t]*\[?[ \t]*system\b" if name == "system-marker" else pattern,
    )
    for name, pattern in PATTERNS
)
SERVING_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in SERVING_PATTERNS
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


def _run(extract: dict, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> dict:
    hits = []
    for field in FIELDS:
        text = _field_text(extract, field)
        for name, rx in patterns:
            match = rx.search(text)
            if match:
                hits.append(
                    {"field": field, "pattern": name, "match": match.group(0)[:120]}
                )
    return {"hit": bool(hits), "hits": hits}


def detect(extract: dict) -> dict:
    """Frozen Phase 5 detector: `{"hit", "hits"}` over the registered fields."""
    return _run(extract, COMPILED)


def detect_serving(extract: dict) -> dict:
    """Serving detector: frozen patterns with the `system-marker` defect fixed.

    Identical to `detect()` except the `system-marker` pattern requires a
    line-start marker, so ordinary prose containing the word "system" no
    longer escalates to an alert.
    """
    return _run(extract, SERVING_COMPILED)
