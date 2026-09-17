# PhishNet execution plan — rev. 2 (2026-09-17)

Fixed text. This is the user-issued plan this phase executes against,
committed so the acceptance criteria are fixed text. Transcribed verbatim
from the two plan messages; the 12-criterion table in §2 is authoritative
(it supersedes any reconstructed draft).

## 1. Finish Phase 3

Amendment E, committed before any more lookups.
Drop CT, for three reasons: crt.sh's documented rate limit, no validated alternative source, and no way to serve it at request time.
Move forward DNS and TLS capture to future work.
Update the acceptance criteria that mention either.
Keep criterion 12 as the stub-latency measurement, which is nearly free.
Then freeze the protocol: bug fixes only. Anything else becomes a recorded deviation in the report.
RDAP pass. Age only, then seal and pin with the population manifest sha.
Join and gate. The contamination gate runs for age only, both bands, threshold 0.05.
Ablation.
Rows: (a) lexical + is_hosted_tenant, (b) + age, (e) the Tranco diagnostic.
Baseline: the platform-prior baseline on the hosted slice.
Reported alongside: the survival-stratum and hosted slices, the cold-start curve (0/50/100% missing age), the fixed-threshold verdicts at 0.5% and 1%, and the threshold-transfer verdict.
Write-up.
The Phase 3 report, which leads with the ablation, then refusals and amendments.
docs/point-in-time.md and the coverage doc.
The model card: survivorship, the Tranco leak, hosted coverage limits, cold start, threshold transfer, and why CT was dropped.
The README headline, the roadmap update, and the authorship lines.
Close. Check each acceptance criterion against its evidence, then open the PR.

The cold-start number stays. It's a research result about how much the feature depends on lookups succeeding, not a product metric.

## 2. Refinements (binding)

### 2.1. Register what happens if age fails its own gate

If age fails: the headline is row (a) only, and age is reported as ineligible, with its gap and interval.
Secondary analysis, labeled conditional: rows (a) and (b) retrained and evaluated on age-known rows only, in both bands, with paired lift on those rows. Report coverage per class and stratum beside it, and state that the result says nothing about rows where the lookup failed.
Cold-start curve: still published. It's the production-relevant number either way.

This keeps the gate binding and still answers whether age helps when it's available.

### 2.2. The real criteria

| # | Criterion (plan.md rev. 2) | Amendment E |
|---|---|---|
| 1 | collect.yml ran daily; fresh-stratum size reported | — |
| 2 | Population passes gates, or recorded refusal | — (met via D1, stratified gate) |
| 3 | repro/hashes.json namespaced; CC corpus pinned | — |
| 4 | is_https closed under the pre-committed rule | — |
| 5 | No unsafe or selection-leaked feature in the headline; classification documented; Tranco diagnostic only | + CT dropped, with reasons |
| 6 | Survival strata defined, measured, reported (incl. age distribution per stratum) | — |
| 7 | Hosted keyed correctly or flagged; reported separately | — (tenant grouping per A/B) |
| 8 | Per-class and per-stratum unknown rates for every enriched feature | age only |
| 9 | Fixed thresholds; FPR with interval; three-valued verdict | — (wider-interval rule per D) |
| 10 | Cold-start (100% miss) for all rows and the fresh stratum | miss = age |
| 11 | Threshold-transfer verdict | — |
| 12 | Tier-1 p50 single-digit ms with stub | — |

Forward DNS was never a criterion, so it simply moves to future work. The corpus-size and domain-floor items belong to D0.7, and are reported under criteria 2 and 9.

### 2.3. Commit every script that produces a result

The fixed-threshold eval driver as a script in Temp, and the Tranco diagnostic under scratch/, both produce numbers that go into the report. Put them in the repo (for example ml_training/eval_phase3.py and ml_training/tranco_diagnostic.py) and record their commit hashes in the report.

### 2.4. Take the RDAP limit per request, not upfront

Wrap each HTTP call in the limit for the server it's about to contact, one slot at a time. That can't deadlock, so no ordering rule is needed.

### 2.5. Leave CT out of the gate output entirely

Run the gate on the signals that were actually requested, and have the output list CT as excluded by Amendment E. Also add a check that no ct_* column appears in X for rows (a) or (b).

### 2.6. Define the transfer verdict so it can reach a conclusion

Report instead:

the calib-achieved FPR versus the test-achieved FPR at the same threshold, with the interval on the difference;
Phase 2's miss (0.60% at a 0.5% target) beside it.

The verdict then asks whether the calib-to-test drift is smaller than Phase 2's, at 0.5% and at 1%.

### Also

docs/point-in-time.md: mark the CT sections historical, and say that no retrospective TLS capture was attempted.
Model card: list the RDAP 404 rate by class as its own item, whichever way the gate goes. It's direct evidence of takedown during the collection window.
