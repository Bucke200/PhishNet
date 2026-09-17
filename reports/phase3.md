# Phase 3 report — domain age against a stratified-gate population

Headline first: on the pinned population, with thresholds fixed on the
calib band and judged by the wider-interval rule, the lexical baseline
with `is_hosted_tenant` reads **52.4% recall at 0.50% FPR
(indistinguishable, [0.38%, 0.64%])**. Domain age is **ineligible for
the headline**: its test-band unknown gap is 0.059 against the 0.05
budget. Where age resolves, it helps decisively (paired recall lift
+0.20–0.31 on all rows, +0.32–0.37 on age-known rows). Certificate
history was dropped before measurement (Amendment E).

## 1. Ablation (thresholds fixed on calib, applied to test)

Thresholds from row (b) calib scores: 0.917390 @0.5%, 0.865044 @1.0%.

| row | 0.5%: recall / FPR / verdict | 1%: recall / FPR / verdict |
|---|---|---|
| (a) lexical + hosted | 52.4% / 0.50% / **indistinguishable** [0.38, 0.64] | 62.2% / 1.25% / **unmet** [1.05, 1.47] |
| (b) + age (ineligible) | 78.0% / 0.56% / indistinguishable [0.43, 0.70] | 83.7% / 1.18% / indistinguishable [0.98, 1.41] |

Paired lift (b − a), domain-bootstrap: recall **+0.20–0.31**,
PR-AUC **+0.07–0.14** at the 0.5% threshold. Both intervals exclude
zero with margin — age carries real signal where lookups succeed.

Row (e), Tranco diagnostic (`reports/tranco-diagnostic-p3.json`):
long-tail tiers (10k–99,999 and 100k–1M) are NOT consistently closer
to phishing than the current benign population
(netloc_len AUC 0.84/0.80 vs 0.64 current; hyphen/digit AUCs flat).
No support for the popularity/head-sampling artifact hypothesis on
hostname shape. Rank stays out of every trained row.

Baseline, hosted slice: `platform_prior(train)` reaches PR-AUC 0.769
on the 1,601 hosted test rows beside the model's 81.6% hosted recall
at the 0.5% threshold — platform identity explains much of hosted
recall, as pre-registered (actor-disjointness is not claimed).

### Slices (row b, 0.5% threshold)

| slice | recall | note |
|---|---|---|
| fresh (579) | 77.9% | fresh claims read here, never the headline alone |
| short (2,236) | 83.4% | |
| unknown (984) | 65.9% | live-feed rows of unknown age |
| non-hosted (23,218) | 76.9% | lift is read here (Amendment B) |
| hosted-tenant (1,601) | 81.6% | FPR 1.67% on hosted benign (n=720) |
| novel-tenant (1,526) | 81.2% | seen-tenant n=75: 92.9%, thin, as measured |
| hosted roots | 99.8% | 646 phish vs **2 benign** — no FPR claim possible |
| hosted pathN benign | FPR 0.2% | n=509, the one hosted-benign cell with weight |
| hosted path1 benign | FPR 6.3% | n=126, coarse, disclosed |

### Conditional secondary (Amendment E.2, age-known rows only)

Gate failed, so this is labeled conditional — it says nothing about
rows where the lookup failed. Retrained and evaluated on age-known
rows (thresholds on calib age-known):

| row | 0.5%: recall / FPR / verdict | 1%: recall / FPR / verdict |
|---|---|---|
| (a) | 35.4% / 0.44% / indistinguishable | 48.1% / 0.82% / indistinguishable |
| (b) | 70.1% / 0.49% / indistinguishable | 78.1% / 1.03% / indistinguishable |

Paired lift (b − a): recall **+0.32–0.37**, PR-AUC +0.14–0.18.
Transfer fixed at both points (drift 0.01pp / 0.05pp).
Coverage beside it: train 27,323/44,285 age-known; calib
8,660/13,157; test 19,447/24,819.

### Cold-start curve (row b, 0.5% threshold)

| missing age | recall (all) | FPR | recall (fresh) |
|---|---|---|---|
| 0% | 78.0% | 0.56% | 77.9% |
| 50% | 65.7% | 1.10% | 63.7% |
| 100% | 53.4% | 1.59% | 54.4% |

Losing age costs ~25pp recall and triples FPR. This is a research
result about lookup dependence, not a product metric — it sizes the
Phase 4 escalation band (cold-start rate, calibrated-score space).

### Threshold transfer (criterion 11)

| target | calib FPR | test FPR | drift | Phase 2 miss | verdict |
|---|---|---|---|---|---|
| 0.5% | 0.49% | 0.56% | 0.06pp [-0.21, +0.33] | 0.10pp | **fixed** |
| 1% | 1.00% | 1.18% | 0.19pp [-0.20, +0.55] | 0.10pp | not-fixed |

The era-matched calib band fixes transfer at 0.5% (drift below Phase
2's 0.10pp with room); at 1% it does not by the pre-registered bar.
Read beside the "suspicious" calib shape audit (0.815) recorded at
population build.

### Latency (criterion 12)

Tier-1 serving shape (stub + lexical/hosted + model, per URL, n=300):
**p50 14.3 ms, p90 15.2 ms** — above single-digit, so criterion 12
reads **unmet**. Breakdown: ~7.8 ms fixed per-call extractor overhead
(0.3 ms batched), the stub itself negligible. Eval-mode join scorer:
p50 15.6 ms (conservative reference). Batch serving would be
sub-millisecond per URL; single-URL blocking is the honest number and
it misses the bar. Phase 6 serving work owns it.

## 2. Refusals (kept on record, unchanged)

1. The 40k enlargement failed the unstratified gate (root drift
   0.238, depth AUC 0.364).
2. The 12k corpus passes unstratified but fails the stratified gate
   on six metrics.
3. Age fails the test-band contamination gate (0.059 > 0.05) —
   headline is row (a); conditional secondary above.
4. CT dropped unmeasured (Amendment E), not failed.
5. Criterion 12 unmet (14.3 ms vs single-digit).

## 3. Amendments (in order)

A — `is_hosted_tenant` in X every row; na gap struck (would fail by
construction). B — no platform cap; lift read on the non-hosted
slice; row (a) ≠ Phase 2 champion. C — evasion rerated, tenant
bucketing, phishing-tenant exclusion, multi-crawl, 2,000-row
hosted-benign stratum (descriptive), novelty slice, platform-prior
baseline. D — stratified gate (D0.2), D0.1 pin, try order; D0.5–D0.8:
join-counted M3, bounded wave fetch, cap 4 with expected downgrade,
quartile bands, accepted N 23k–40k, errata, url_type re-derivation.
E — CT dropped (measured ~97% 429 failure; no validated source;
unservable); age-only finish; age-failure fallback; transfer drift
rule; gate on requested signals with CT listed excluded.

## 4. Recorded deviations

1. **Shared-10 provider pool.** The enrichment batch ran RDAP+CT
   through one pool of 10; crt.sh throttled ~97% of CT lookups
   (938/969 checkpointed, class-symmetric). Led to Amendment E;
   per-server RDAP gates (≤2) added for the age pass.
2. **Abandoned combined run.** Stopped at 3,500/41,739 keys,
   quarantined at `scratch/quarantine-enrichment-run1/`, sha256
   `47e0ee98258acbccea65046de5687f99e1bf120fd9475f95a9b673fe10e397e7`
   — never sealed or pinned.
3. **Wave intake suffix bug.** First D1 select consumed 0 wave rows
   (UNLOAD parts are extensionless); discarded ungated/unpinned,
   fixed with a regression test, select re-run once (Amendment D
   bug-fix allowance).
4. **CLST timestamps.** Chilean WHOIS dates parse with dropped-tz
   FutureWarnings; fail-closed handling covers a future hard error.
   Observed, not fixed (freeze).

## 5. Reproducibility

Populations: `repro/hashes.json` (successor eval) and
`repro/hashes-p3.json` (4/4 via `repro/verify.py`). Corpus:
`reports/d1-corpus-pin.json`. Enrichment:
`reports/phase3-enrichment-pin.json`. Gate:
`reports/phase3-age-gate.json`. Ablation:
`reports/phase3-ablation.json` (+ `-ageknown`). Diagnostic:
`reports/tranco-diagnostic-p3.json`. Result-producing scripts and
their commits: `ml_training/eval_phase3.py` @ `a9ad325b`,
`ml_training/tranco_diagnostic.py` @ `de30ec42` (this report's
per-URL-type block: driver @ `1a5f0d69`).
