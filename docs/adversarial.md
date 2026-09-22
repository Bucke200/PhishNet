# Adversarial hardening (Phase 5)

Pointer page. The Phase 5 evidence lives in one place; this file does not
restate its numbers.

- **Protocol:** `docs/phase5-preregistration.md` (Amendments `phase5-A…I`).
- **Results:** `reports/phase5-adversarial.md`,
  `reports/phase5-adversarial.json`, `reports/phase5-lexical.md`,
  `reports/phase5-lexical.json`.
- **Sealed run store:** `runs/phase5/` — 564 calls, 3 cold repeats,
  `openai/gpt-oss-120b`, frozen `p5-h1` prompt and pure-function detector.
- **Model-card summary:** `docs/model-card.md` → "Tier-2 LLM Cascade
  Limitations".
- **Serving implications:** `reports/phase6.md` (C3 fail-closed, `phase6-A`)
  and `docs/production-gaps.md` §5–§6.

## What the verdicts are attached to

Phase 5's arm verdicts are attached to the frozen `p4-v1` / `p5-h1` prompts
and the frozen detector. `p6-v1` widens the response schema
(`credential_types`) for serving; it does **not** change any Phase 5 verdict
(`reports/phase6.md`, C4).

## Operational consequence

The Retain policy is prohibited (`production-gaps.md` §5); serving is
fail-closed (`reports/phase6.md`, C3). The detector is trivially evadable
(`production-gaps.md` §6), so it is not a security boundary — a hit
escalates, and instruction-hierarchy prompt hardening is the primary
defense.
