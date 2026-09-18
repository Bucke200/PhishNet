# Phase 4 report — close-out (phase4-D, unpublished)

The recorded sweep was not run, so only a sealed provisional exists
and nothing here publishes. The phase question — whether the LLM
layer beats the password baseline — is unanswered, not negative.
Coverage is stated beside every number (criterion 14).

## Cascade [default] (provisional)
- fpr0.5: cascade recall=0.5049 fpr=0.00414 (tp=1918 fp=87) vs tier1 recall=0.5041 fpr=0.00404 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
- fpr1.0: cascade recall=0.6070 fpr=0.00990 (tp=2306 fp=208) vs tier1 recall=0.6065 fpr=0.00980 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
## Cascade [alternative] (provisional)
- fpr0.5: cascade recall=0.5049 fpr=0.00414 (tp=1918 fp=87) vs tier1 recall=0.5041 fpr=0.00404 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
- fpr1.0: cascade recall=0.6070 fpr=0.00990 (tp=2306 fp=208) vs tier1 recall=0.6065 fpr=0.00980 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.

## Lead findings (close-out, phase4-D)
- Fetchability is a label proxy: test phish fetch 0.135 vs test benign 0.886 (Step-0 marginals). The takedown filter already selected for live phish; the fetch then selects again.
- Structural ceiling (sealed Step-0 data, no LLM call): fetched in-band test phish 132/3799 = 0.0347 is the most recall the layer could ever add; FPR exposure is 969/21020 = 0.0461 benign. This explains 'indistinguishable' before a reader asks.
- system_fingerprint rotates per call (35 values over 101 calls); run identity is model+prompt+seed 0, explicitly weaker (phase4-C).
- Determinism 11/50 (22%, over the 5% bar): 10 free-tier quota failures plus 1 genuine phishing->suspicious wobble. The response cache, not the seed and not the temperature, is what makes the published numbers reproducible.
- Password baseline, exact: among fetched-ok test rows with extracts — the only rows the rule can see — 0 fires in 152 phish and 2 fires in 1,006 benign. (Over all 24,819 test rows that is still 2 fires, both benign; unfetched rows score 0.0 meaning 'no page', not 'rule didn't fire'.) Not threshold degeneracy (scores are 0/1 against t_alert=0.9269, so every 1.0 fires) — the rule itself barely fires, at n=152/1,006. A cascade-slot variant would be a new predictor after LLM reads and stays future work.

## Descriptive (provisional)
- agreement on sealed rows: 0.956 (n=68)
- verdict distribution: {'benign': 63, 'phishing': 5}
- rank metrics: artifactual per §2 (tie block at t_alert); fixed-threshold recall/FPR above is primary.
- cost (cold-cache, n=119): prompt 1475.82 / completion 243.22 (reasoning 89.58, visible 153.64) tokens per call (independently rounded means; exact: 1475.82 / 243.22 = 89.58 + 153.64); latency p50 1308ms p90 2069ms; provisional forecast at Groq listed rates ($0.15/$0.60 per 1M, 2026-09-18): $0.00037/call, $0.367/1k escalated, $0.0164/1k rows at escalation 0.0446; 3x1106 repeats ~= $1.22.
- coverage: verdicts sealed for 68/1106 test in-band fetched-ok rows with extracts. Provenance of the 1,106: 1,101 in-band by stored manifest scores plus 5 in-band step0-sample rows (stored tier1 NaN). The NaN is sample membership, not a scoring gap — 396 NaN rows in total equals the 396-row step0_sample, all 57 test-ok NaN rows are test-split members (0 in calib/train), and the sweep rescored every test row through the identical Tier-1 path (5 of the 57 in-band, verified). Criterion 4 is intact: bucketing is total over recomputed scores and the population is built from the test CSV only. Unjudged rows retain Tier-1 (§2 failure policy).

## Criteria (close-out)
- 1, 2, 4, 5, 7, 8, 9a, 11, 12: met.
- 3: unmet. Achieved benign band mass 406/8110 = 0.0501; §1.2's '0.05 by construction' was wrong — floor(0.055 x 8110) = 446 and floor(0.005 x 8110) = 40 admit at most 406 rows, above 405.5, knowable at registration (phase4-D).
- 6: met on provisional numbers only.
- 9: met under phase4-C (weaker, stated).
- 9b, 10b: unmet, per phase4-D (no recorded sweep; Developer tier unavailable, free tier ~100 calls/day).
- 10: reported (22%; 10 quota failures + 1 genuine wobble).
- 10a: met (unanswered, not negative; nothing beyond this model).
- 13: met on provisional counts; priced forecast from Groq's own rate page, dated 2026-09-18, labeled provisional.
- 14: met.
