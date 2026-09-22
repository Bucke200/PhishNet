# Phase 3 report — domain age against a stratified-gate population

Headline first: on the pinned population, with each row's thresholds
fixed on its own calib scores and judged by the wider-interval rule,
the lexical baseline with `is_hosted_tenant` reads **50.4% recall at
0.40% FPR (indistinguishable, [0.29%, 0.53%])**. Domain age is
**ineligible for the headline**: its test-band unknown gap is 0.059
against the 0.05 budget. Age helps decisively on paired rows
(+0.22–0.34 recall at each row's own operating point) — with the
caveats below on what that lift does and doesn't show. Certificate
history was dropped before measurement (Amendment E).

## 1. Ablation (thresholds fixed on calib, applied to test)

Each row fixed on its OWN calib scores (the models score on different
scales — an earlier draft applied row (b)'s numbers to row (a) and
overstated its FPR; corrected here). Row (a): 0.926936 @0.5%,
0.878084 @1.0%. Row (b): 0.917390 @0.5%, 0.865044 @1.0%.

| row | 0.5%: recall / FPR / verdict | 1%: recall / FPR / verdict |
|---|---|---|
| (a) lexical + hosted | 50.4% / 0.40% / **indistinguishable** [0.29, 0.53] | 60.7% / 0.98% / **indistinguishable** [0.81, 1.18] |
| (b) + age (ineligible) | 78.0% / 0.56% / indistinguishable [0.43, 0.70] | 83.7% / 1.18% / indistinguishable [0.98, 1.41] |

Paired lift (b − a), domain-bootstrap, each row at its own 0.5%
threshold: recall **+0.22–0.34**, PR-AUC **+0.07–0.14**. Both
intervals exclude zero with margin. What this does NOT show: it does
not show age helping "where lookups succeed" — the all-rows lift
includes the contaminated flag (benign rows read unknown more often,
so the model partly reads `age_known = 0` as benign), and benign
domains are old by Tranco construction (row (e) tested hostname
shape, not age). Only the conditional analysis below speaks to known
rows, and the benign-stratum table beside it checks the sampling
confound: in s4/s5, row (b)'s benign FPR (0.43%/0.42%) runs slightly
ABOVE row (a)'s (0.31%/0.35%) — no benign-side advantage in the low
strata; age's value is phishing recall, partly paid in benign-tail
FPs. s6 (n=131) is too thin to read.

Row (e), Tranco diagnostic (`reports/tranco-diagnostic-p3.json`):
long-tail tiers (10k–99,999 and 100k–1M) are NOT consistently closer
to phishing than the current benign population
(netloc_len AUC 0.84/0.80 vs 0.64 current; hyphen/digit AUCs flat).
No support for the popularity/head-sampling artifact hypothesis on
hostname shape. Rank stays out of every trained row.

Baseline, hosted slice, same metrics both sides: on the 1,601 hosted
test rows the model reads PR-AUC 0.979 with 81.6% recall at the 0.5%
threshold; the platform prior reads PR-AUC 0.769 with 53.2% recall at
that same threshold. Platform identity explains part of hosted
recall — but not most of it (actor-disjointness still not claimed).

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
| (a) | 39.5% / 0.57% / indistinguishable | 49.8% / 0.91% / indistinguishable |
| (b) | 70.1% / 0.49% / indistinguishable | 78.1% / 1.03% / indistinguishable |

Paired lift (b − a), each row at its own threshold: recall
**+0.28–0.33**, PR-AUC +0.14–0.18. Transfer indistinguishable at both
points (intervals straddle the bar / no comparator — corrected per
the same interval logic). Coverage beside it: train 27,323/44,285
age-known; calib 8,660/13,157; test 19,447/24,819. Note the mix
shift: row (a) falls from 50.4% to 39.5% recall on age-known rows, so
successful-lookup rows are a different, harder mix — the +0.28–0.33
is measured on that harder mix, not on the headline population.

### Cold-start curve (row b, 0.5% threshold)

| missing age | recall (all) | FPR | recall (fresh) |
|---|---|---|---|
| 0% | 78.0% | 0.56% | 77.9% |
| 50% | 65.7% | 1.10% | 63.7% |
| 100% | 53.4% | 1.59% | 54.4% |

Losing age costs ~25pp recall and triples FPR. This is a research
result about lookup dependence, not a product metric — it sizes the
Phase 4 escalation band (cold-start rate, calibrated-score space).
Consistency check: 53.4% recall at full miss against row (a)'s 50.4%
— the no-age model and the never-had-age row agree, as they should.

### Threshold transfer (criterion 11, on row (a))

An earlier draft computed this on row (b) — ineligible — with a
point-estimate rule that could not conclude: both drift intervals
contain Phase 2's 0.10pp. Corrected: row (a), wider-interval rule.

| target | calib FPR | test FPR | drift | bar | verdict |
|---|---|---|---|---|---|
| 0.5% | 0.49% | 0.40% | -0.09pp [-0.36, +0.18] | 0.10pp | **indistinguishable** (straddles) |
| 1% | 1.00% | 0.98% | -0.02pp [-0.41, +0.35] | none | **indistinguishable** (no Phase 2 1% comparator) |

Neither target concludes for or against era-matched transfer: the
0.5% interval straddles the bar, and at 1% there is nothing to compare
against. Read beside the calib shape audit below — the band the
thresholds were fixed on separates train from calib on shape alone at
0.815 (mixture), 0.700 main-stratum.

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

Caveat with refusal-level prominence: the calib band's shape audit
reads **suspicious** (shape-only train→calib ROC-AUC 0.815 — above the
0.753 that marked splits-eval suspicious), and every threshold in §1
was fixed on that band. The stratified rerun (main stratum only)
reads 0.700, verdict ok: the confound is the hosted mixture, not the
URL shapes the gate polices. Thresholds therefore separate partly on
platform mix — which is exactly what the transfer section measures
rather than assumes.

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

## 6. Close-out — acceptance criteria vs evidence

Criteria text formerly `docs/plan.md` §2.2 (committed in `33350729`,
file removed after the roadmap superseded it; full text preserved
below and in git history). Amendment column per Amendment E.

| # | Criterion | Amend. E | Verdict | Evidence |
|---|---|---|---|---|
| 1 | collect.yml ran daily; fresh-stratum size reported | — | met | snapshots 09-12..09-17 in `data/raw`; test fresh n=579 in split manifest |
| 2 | Population passes gates, or recorded refusal | — (met via D1, stratified gate) | met | D1 stratified main `[]`; p3-split exit 0 (audits ok / suspicious-advisory) |
| 3 | repro/hashes.json namespaced; CC corpus pinned | — | met | `repro/hashes.json` untouched; `repro/hashes-p3.json` 4/4; `reports/d1-corpus-pin.json` |
| 4 | is_https closed under the pre-committed rule | — | met | manifest `is_https_rule` drop, gap 0.069 train+calib |
| 5 | No unsafe or selection-leaked feature in the headline; classification documented; Tranco diagnostic only | + CT dropped, with reasons | met | headline (a) per Amendments A/C; `reports/tranco-diagnostic-p3.json`; Amendment E |
| 6 | Survival strata defined, measured, reported (incl. age distribution per stratum) | — | met | manifest `survival_strata`; coverage doc age table |
| 7 | Hosted keyed correctly or flagged; reported separately | — (tenant grouping per A/B) | met | tenant grouping; hosted/novelty slices; prior baseline |
| 8 | Per-class and per-stratum unknown rates for every enriched feature | age only | met | `reports/phase3-age-gate.json`; coverage doc |
| 9 | Fixed thresholds; FPR with interval; three-valued verdict | — (wider-interval rule per D) | met | `reports/phase3-ablation.json` (indistinguishable throughout, as measured) |
| 10 | Cold-start (100% miss) for all rows and the fresh stratum | miss = age | met | driver `cold_start` cells (all + fresh) |
| 11 | Threshold-transfer verdict | — | met | indistinguishable both points per E.3 interval rule (row (a)) |
| 12 | Tier-1 p50 single-digit ms with stub | — | **unmet** | 14.3 ms serving-shape p50 vs single-digit |

11 met, 1 unmet. Recorded alongside: age headline-ineligible
(conditional secondary published), CT unmeasured. Suite: 309 passed,
2 live-skips; ruff + mypy clean; CRLF rule holds (snapshot converted
post-seal, seal hash re-verified identical).

## 7. Reproducibility

Populations: `repro/hashes.json` (successor eval) and
`repro/hashes-p3.json` (4/4 via `repro/verify.py`). Corpus:
`reports/d1-corpus-pin.json`. Enrichment:
`reports/phase3-enrichment-pin.json`. Gate:
`reports/phase3-age-gate.json`. Ablation:
`reports/phase3-ablation.json` (+ `-ageknown`). Diagnostic:
`reports/tranco-diagnostic-p3.json`. Result-producing scripts and
their commits: `ml_training/eval_phase3.py` @ `41636096`
(per-URL-type block @ `1a5f0d69`; thresholds identical across both),
`ml_training/tranco_diagnostic.py` @ `de30ec42`.

Suite accounting: 311 collected, 309 passed, 2 live-skips. No test
was removed or merged at any point — `def test_` count went 313 →
340 across the phase (all additions; `git log -S "def test_"` shows
only additive commits, `--diff-filter=D` on `tests/` is empty). The
"330" figure from review matches no recorded run; the verified
numbers are above (the gap to 340 definitions is `test_eval.py`,
outside the `tests/` collection root).

Seal verification: the seal hashes records sorted by key and
canonically serialized — NOT the file bytes — which is why the
post-seal CRLF conversion left it unchanged, and also why `sha256sum`
on the file won't reproduce it. Both hashes on record: file sha256
`12d0acdb38b524f40564bf23ab4857509dcbafe448bb96d9d7196d01f3f6b801`,
seal `058ee583…60d29c426`. Reproduce the seal with:
`uv run python -c "import json,hashlib;recs=sorted((json.loads(l) for l in open('data/enrichment-p3-2026-09-17.jsonl',encoding='utf-8') if l.strip() if json.loads(l).get('run_id')=='run-1'),key=lambda r:str(r.get('cache_key')));h=hashlib.sha256();[h.update(json.dumps(r,sort_keys=True).encode()+b'\n') for r in recs];print(h.hexdigest())"`
