# Gate notes (`p5-gate`, committed after the run — additive, seals untouched)

- Two earlier attempts returned HTTP 200 with parsed verdicts but failed the
  original `cached_tokens`-present assertion (usage had no
  `prompt_tokens_details` object at all). The assert fires pre-store, so both
  attempts were discarded unsealed — no partial seals exist for them.
- The rule was revised (`phase5-G`, iff-cached) on probe evidence, and the
  committed 5/5 passed under the revised rule. Revision scope: API response
  shape only, never model quality.
- Run-class note: `provisional` (gate), per the Phase-4 pattern — a gate
  supplies no published number.
