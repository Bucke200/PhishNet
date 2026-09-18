"""Password-field baseline (§7): pure function of the extract.

A password `<input>` is present AND the form-action host differs from the
page host → 1.0, else 0.0. Roughly twenty lines, `eval.py`-compatible
(`name` + `score(urls)`). Built and run BEFORE any LLM output is read, so it
cannot be retrofitted after the LLM looks good.

Extracts come from the frozen snapshot manifest (`url → extract` map); a URL
without a snapshot scores 0.0 (no page observed, no password form observed) —
a stated rule, applied identically to every unevaluated row. If the baseline
closes most of the tier-1 → tier-1+LLM gap, that is the phase's headline.
"""

from __future__ import annotations

from collections.abc import Sequence


def password_offdomain(extract: dict) -> float:
    """1.0 iff a password field posts to a foreign host."""
    page_host = str(extract.get("page_host", "") or "").lower()
    for form in extract.get("forms", []):
        action_host = str(form.get("action_host", "") or "").lower()
        has_password = any(
            "password" in str(inp.get("type", "")).lower()
            or "password" in str(inp.get("name", "")).lower()
            for inp in form.get("inputs", [])
        )
        if has_password and action_host and action_host != page_host:
            return 1.0
    return 0.0


class PasswordBaseline:
    """Eval-harness predictor over frozen extracts (missing snapshot → 0.0)."""

    name = "password_baseline(offdomain)"

    def __init__(self, extracts: dict[str, dict] | None = None):
        self.extracts = extracts or {}

    def score(self, urls: Sequence[str]) -> list[float]:
        out = []
        for url in urls:
            extract = self.extracts.get(url)
            out.append(password_offdomain(extract) if extract else 0.0)
        return out
