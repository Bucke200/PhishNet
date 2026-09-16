# Phase 3 pre-registration (committed before the bulk run)

Everything below was fixed before the full enrichment run and the
ablation numbers exist. The commit that adds this file is the
timestamp; changing any value afterwards is a protocol change, not a
tuning decision, and must be recorded as such. (`update.md` is
git-ignored and cannot serve this purpose.)

## 1. Population: band dates and power rule

* T1 = 2026-07-25T00:00:00+00:00, T2 = 2026-08-22T00:00:00+00:00
  (`make p3-split`: `P3_T1`, `P3_T2`). T2 is the splits-eval cutoff, so
  the threshold-transfer verdict differs from Phase 2 only in its
  calibration slice.
* Benign buckets 30/20/50 train/calib/test (`P3_TFRAC = 0.5`,
  `P3_CFRAC = 0.2` — fractions are test/calib shares). The recorded
  option-2 fallback is 20/20/60 (invoke with
  `--benign-test-fraction 0.6 --benign-calib-fraction 0.2`); the switch
  fires by rule on the post-cap benign test count
  (`PHASE3_BENIGN_TEST_FLOOR = 15_000`, recorded per rebuild in
  `phase3_power`), never by judgment after seeing results.
* Raw file list pinned per rebuild via `raw_file_hashes`. New daily
  snapshots grow a new population, never this one.

## 2. Contamination gate thresholds

* `max_unknown_gap = 0.05`, `max_na_gap = 0.02` (per-class |phish −
  benign| rate gaps, per signal, per band). The n=40 pilot showed an
  age unknown gap of 0.067 — age starts watched, not cleared.
* Unknown means a failed lookup and nothing else: na rows are excluded
  from the unknown denominator (`n_eligible` reported); the na share is
  gated on its own.
* Per-signal eligibility: a CT breach cannot knock age out (or vice
  versa). Missing class reads unmeasurable (fail-closed).
* The gate runs on the test band AND the train band (`gate_joined_rows`
  with domain-bootstrap gap intervals, clusters = cache keys). A point
  estimate inside budget whose interval straddles the threshold reads
  ineligible — unresolved at that scale, not passed.
* A breached signal stays out of the headline, however good its AUC.
* The na share is reported per class and band (`by_class` hosted
  composition in the split manifest; `na_gap` in the gate bundle) but
  never gated — see Amendment A.

## Amendment A — hosted flag in X, na gap demoted (2026-09-16, pre-rebuild)

Dry run on staged data (tenant grouping active) measured per-class na
gaps of 0.32 train / 0.45 calib / 0.20 test against the committed 0.02
budget: the na gate fails by construction on every band. The na gap is
not contamination — it is the hosted share, and hosted rows are
near-all phishing by nature of the phenomenon. Gating it would empty
the headline for a reason unrelated to lookup quality, while the model
(with na flags excluded from X) still learned "not known → phishing"
from hosted rows the unknown gate had removed.

Fix, committed before any numbers exist:

* `is_hosted_tenant` (0.0/1.0, URL-derived from the PSL-private/vendor
  list, serving-time known, no time dependence) rides X in EVERY
  ablation row including lexical-only (a). The enrichment lift cannot
  pick up the hosted indicator, and a known=0 on a non-hosted row means
  a failed lookup and nothing else.
* `max_na_gap` is struck — recorded here, not deleted. The na share
  stays reported per class and band beside the gate.

## Amendment B — no platform cap; lift read on the non-hosted slice (pre-rebuild)

Dry-run concentration check (staged CC-12k, tenant grouping, per band —
`hosted_concentration()` prints it at every phase-3 build):

| band | top platform | rows / tenants | share of band phish |
|---|---|---|---|
| train (31,956) | weebly.com | 3,652 / 3,649 | 11.4% |
| calib (5,047) | pages.dev | 1,224 / 618 | 24.3% |
| test (3,437) | vercel.app | 228 / 157 | 6.6% |

Rows ≈ tenants on every platform (e.g. 3,652/3,649): attackers already
use ~one URL per tenant, so the per-tenant cap binds nothing and a
platform cap would not fix actor concentration — it would delete a real
phenomenon. Leadership rotates by band (weebly → pages.dev →
vercel.app): no single persistent actor. Shared `(platform, "/")`
shapes (17 in train covering 10,089 rows) are generic kit defaults
across thousands of distinct tenants, not one campaign. Precedent:
`netloc_len` was deliberately left ungated for attacker behavior for
the same reason. **Decision: no platform-level secondary cap.** The
share is real; bit.ly's 50 rows on 1 tenant group are the designed
shortener behavior (opaque redirects key on the host itself).

Reporting rules, pre-registered with the same timestamp:

* Enrichment lift is interpreted on the **non-hosted slice** (hosted
  rows resolve na by construction, so age/CT cannot move them; up to
  ~45% of a band would otherwise dilute the headline into
  meaninglessness). The headline stays as committed in §3; the
  non-hosted numbers sit beside it with that label.
* **Wide recall/PR-AUC intervals are expected, not a surprise**: the
  eval bootstrap clusters on registrable domain, so all windows.net
  tenants form one cluster and a few clusters hold much of the phishing
  side. FPR is unaffected (benign rows are rarely hosted). The
  conservative clustering stays.
* **Row (a) is not the Phase 2 champion**: lexical-only now includes
  `is_hosted_tenant`, and at this hosted share that flag alone is
  likely a strong feature. The report states this outright so nobody
  reads the row-(a)-versus-Phase-2 gap as enrichment or as regression.

## Amendment C follow-up — artifact corrected; tenant bucketing; exclusion; descriptive target (pre-pin)

### Burstiness, corrected

The multi-platform 09-12/13/14 surges were a timestamp artifact:
OpenPhish rows carry the collection date as `first_seen`, so every
OpenPhish row lands on a snapshot date by construction. Re-run on
PhishTank-only rows (true submission times, 402 hosted test rows),
the synchronized surges disappear. What survives: platform-localized
bursts (blogspot 09-07 ×20 at 0.53 day-share, weebly 09-01 ×26,
netlify 08-22 ×38 on the T2 boundary) against a steady multi-week
drip (vercel 22 days, max 0.16; pages.dev 13 days, max 0.21). The
repeated-stem evidence is timestamp-independent and stands. The
evasion question stays open — which is why the platform-prior
baseline and the novelty slice below carry the weight, not the burst
table.

### Benign bucketing is tenant-level

The benign split hashed on registrable domain, so every vercel.app
benign tenant hashed as vercel.app into one bucket — a whole platform
per band, and the 2,000-row stratum could leave a platform with zero
test coverage. In `--phase3` mode benign rows now bucket by
`split_group` (non-hosted groups ARE registrable domains, so their
assignment is unchanged; legacy path verbatim). Proven by test: two
tenants, one platform, different bands.

### Hosted-benign hygiene at select time

* `--exclude-phishing-tenants-from RAWDIR`: drops candidates living on
  tenants behind phishing URLs (tenant-level, before quota counting so
  quotas still fill; counted per type in provenance as a label-noise
  lower bound — unobserved abusive tenants are not counted).
* `--require-multi-crawl`: keeps only candidates whose tenant appears
  in ≥2 distinct CC crawls (throwaways rarely last that long).
* Both default off (existing outputs byte-identical); both recorded in
  provenance when on.

### The 2,000 target is descriptive

At a 50% test share, ~1,000 hosted benign test rows read ±~0.44pp at
0.5%: the hosted slice will report "indistinguishable" at the budget,
plus counts and the 1% point. Sizing to resolvability (~19k test rows)
is not credible after tenant exclusions and would let one rare
subpopulation dominate benign composition. The 2,000 target stands
with three jobs, none requiring resolution: break the near-label
(hosted flag stops separating by itself), supply counts + 1% reads,
and put benign-hosted train rows in front of the model so
hosted≠phishing is learned rather than assumed. The exclusion
fallback (FPR claim excludes hosted benign, model card + headline
note) applies only if the stratum itself proves infeasible.

### Novel-tenant slice

`tenant_novelty` (phase-3 test CSVs): hosted rows whose digit-masked
tenant stem never appears in train read `novel-tenant`, the rest
`seen-tenant`, non-hosted `non-hosted`. A rough actor-separation
stand-in, reported beside the platform-prior baseline: together they
separate "detects hosted phishing" from "remembers this actor's
naming" without claiming actor identity. Stems come from the final
train frame (post-drop, post-cap).

## Amendment C — evasion evidence rerated; hosted slice discipline; hosted-benign stratum (pre-pin)

### C.1 The rows≈tenants reading was wrong, the decision stands

Amendment B cited rows≈tenants as evidence the hosted share is real.
It is consistent with throwaway-tenant-per-phish evasion and cannot
clear it; the path-shape check cannot separate the cases either
(static tenants serve from `/` either way). What the dry run actually
shows:

* Burstiness: steady background (vercel.app 2–5/day for weeks) plus
  multi-platform surge days (09-12/13/14 light up vercel, blogspot,
  pages.dev simultaneously) — campaign-structured, though feed-side
  batching versus attacker bursts cannot be separated from split
  output alone.
* Repeated stems (`fb-meta-verified-#` ×9/7 on vercel,
  `crypto-r#x` on netlify, `www` ×42/×35): kit morphology across
  tenants — one actor minting lookalikes, or one kit reused by many.
  Actor identity is unprovable from split output.
* Target mix: inconclusive — 97.5% of hosted raw rows carry
  `target: Other` (unlabeled); the labeled slice is brand-diverse per
  platform (Facebook/Comcast/Microsoft/IRS across platforms), i.e. no
  single-brand-per-platform pattern.

The no-cap decision stands, on narrowed grounds: a platform cap cannot
distinguish the cases either, and deleting a quarter of calib phishing
for living on one platform distorts the population to flatter a
metric. Instead, the memorization question is answered without actor
identity: every hosted-slice reading ships beside a
`platform_prior(train)` baseline (train-band platform phish rates,
Laplace-smoothed — `predictors.PlatformPriorBaseline`). If the prior
recovers most of a model's hosted recall, those numbers measure the
platform, not detection. Actor-disjointness for hosted rows is NOT
claimed: the same actor can sit in train and test on different
tenants, and the hosted slice's recall may partly reflect memorized
platform identity.

### C.2 Benign hosted coverage: measured gap, stratum fix

Dry-run benign hosted rows per band: train 61 / calib 58 / test 75
(1.3–3.7% of benign, on different platforms than the phishing side).
Phishing is 20–45% hosted. Consequences, both recorded before the pin:

* `is_hosted_tenant` is near-label in this corpus — the Tranco-selection
  problem recurring on a new axis. Stated, not hidden.
* FPR on hosted benign is currently unresolvable (75 test rows against
  a 0.5% budget), so a model learning "hosted → phishing" would never
  show it in the measured FPR — while production benign traffic on
  those platforms is everywhere.

Fix, in the corpus pin being prepared: a **hosted-benign stratum** in
the Athena enlargement — CC captures under the same platform suffixes
(`url_host_name LIKE '%.<platform>'`, same CC-MAIN-2026-34 primary,
same 200-only + per-tenant caps, same mechanism gates re-run with the
stratum included), **target 2,000 rows**, recorded here before the
query runs. Sizing: ±~0.3pp granularity on hosted FPR — coarse against
the 0.10pp full-test read, but non-vacuous where today there is
nothing. Platform mix follows the suffix list, never phishing
proportions (fitting benign sampling to the test set it will be
measured on). If the stratum proves infeasible (cost or gate
failure), the fallback applies, also pre-registered now: the FPR claim
**excludes** hosted benign pages, stated in the model card and beside
the headline — not discovered after the numbers.

## 3. Headline rule for the unknown stratum

The headline INCLUDES the ~1,063 OpenPhish (`unknown`) rows, with
per-stratum (fresh/short/long/unknown) and hosted slices beside it.
Excluding a quarter of test phishing by basis would be post-hoc
test-set filtering. Fresh claims read from the fresh slice, never the
headline alone. Test `long` is structurally empty (era stamps cap lag
at ~24d); long-lived analysis belongs to calib/train slices.

## 4. Scheme rule

DROP means `canonicalize_scheme()` before featurizing (training and
scoring, one switch), never drop-one-column — `https://` is a
character longer than `http://` and most lexical features read the raw
URL string. The decision is measured on train (+calib in three-band
mode) and persists via `train_config.json` into
`asset_fingerprint.canonicalize_scheme`.

## 5. Split grouping (tenant-level, decided)

Straddler-dropping, campaign caps, and domain floors group by
`split_group`: PSL private-section eTLD+1 where the snapshot resolves
one, else full host; non-hosted rows group by registrable domain
unchanged. Tenants are separate attackers — platform grouping merged
them, straddled every hosted tenant out of later bands (trial: 3.0%
hosted train, 0.0% test), and capped them as one domain. Eval
bootstrap still clusters on `registrable_domain` (conservative: fewer,
larger clusters); hosted rows additionally report as their own slice
via `is_hosted_tenant`.

## 6. Reporting rules

* Report lift (paired domain-bootstrap CIs on differences), not raw
  PR-AUC: the test base rate (~14%) differs from splits-eval's 32%.
* Never compare score distributions across bands: train was 90.2%
  phish and test 37.8% in the trial, so raw scores live on different
  scales. Fixed-threshold FPR (benign-only) is comparable; scores are
  not.
* Operating points use fixed thresholds from the calib band (never
  test sweeps), judged met/unmet/indistinguishable on the wider of
  Wilson vs domain-bootstrap. The pre-registered 1% point carries
  resolvability if 0.5% reads indistinguishable.

## Amendment D — stratified shape gate; corpus try order (pre-selection)

Branch point: master 1046962a. Recorded after the 40k enlargement was
refused (unstratified gate: root drift 0.238, depth AUC 0.3639); that
refusal stays on record unchanged.

### D0.1 Pinned phishing reference
Quotas, validator and p3-split use exactly the ten files
openphish-2026-09-12…16 and phishtank-2026-09-12…16. The refusal was
gated against 09-12…09-15; every rate in M2 differs by ≤ 0.001.

### D0.2 Stratified shape gate (replaces the unstratified gate for promotion)
Measurement (M2, phishing only, no benign outcome involved): hosted
tenants are 21.0% of phishing; P(root|hosted) = 80.1%,
P(root|non-hosted) = 25.0%. is_hosted_tenant is in X (Amendment A), so
the model conditions on this split, and shape alignment matters within
each stratum, not across the mixture.

- Main stratum (promotion-blocking): main benign vs non-hosted
  phishing — type drift ≤ 0.03 per type, scheme gap ≤ 0.04,
  |depth AUC − 0.5| ≤ 0.05, length inversion ≥ 0.
- Hosted stratum (recorded, not blocking): hosted benign vs hosted
  phishing, same metrics. The stratum is descriptive (Amendment C
  follow-up); its failure is reported beside hosted results.
- The unstratified gate is still computed and reported for every
  candidate.

Known cost, recorded before selection: the pinned 12k corpus passes
the unstratified gate (M1: drift ≤ 0.003, scheme 0.0026, depth 0.0096)
but is ~36.5% roots against 25.0% in the non-hosted stratum, so it is
expected to fail D0.2. This gate is adopted despite removing the
cheapest fallback.

### D0.3 Multi-crawl flag unchanged
require_multi_crawl stays as registered (keyed by tenant_group). Cache
audit: 17.9% of hosted tenants span both crawls, 9.0% of root tenants.
The flag disfavors roots by construction under per-crawl sampling.
Hosted-benign FPR is reported per URL type with counts; no hosted-root
FPR claim is made.

### D0.4 Try order (each gate run once, against D0.1 and D0.2)
0. M1-stratified: run D0.2 on the pinned 12k and record the result
   before M3.
1. D1: one second wave (s4–s6 to exhaustion) only if the read-only
   COUNT probe projects ≥ 5,900 selectable new apex roots. Select
   N = min(40k, pool bound under D0.2), gate once. Record the
   strata-imbalance caveat.
2. D2: otherwise, the pool-bound subset of the current pool,
   N fixed by pool counts (~4k roots / 0.25 + hosted stratum
   ≈ 17.5k), gate once.
3. D3: otherwise, the pinned 12k, only if M1-stratified passed.
4. D4: otherwise, recorded refusal per criterion 1. Proceed on the
   least-failing candidate, labeled suspicious, with main-stratum
   depth AUC beside every number, and interpretation through paired
   enrichment-over-lexical lift.

Under-floor results in D1–D3 follow the option-2 rules
(PHASE3_BENIGN_TEST_FLOOR).

### D0.5 M3 counts exact via join (pre-execution)

"To exhaustion" enumerates 994,693 fresh s4–s6 candidates (dry-run
enumeration: s4 7,963 / s5 88,947 / s6 898,706; 923 tenant-skipped), so
per-domain COUNTs (~2M queries) are infeasible, and a 1,500-domain
prefix projection would be stratum-biased (seeded order is per stratum)
and noisy at low per-domain root yield. M3 counts exact instead:

- The kept candidate domains are uploaded as a small table under
  s3://phishnet-athena/, and one join query per crawl partition counts
  DISTINCT apex(+www) root URLs per domain — 200-only, exact root-URL
  equalities, the same unit selection keeps, with the per-domain root
  cap applied client-side. Root synthesis is upside and uncounted.
- Both crawls are joined (partition scans, not per-domain queries);
  2026-30 counts apply only to domains with no 2026-34 capture
  (primary-first-fallback-on-miss, decided client-side per domain).
- Cost is two narrow-column partition scans (single-digit dollars at
  most); measured DataScannedInBytes replaces the estimate in the
  report, which also commits the counts and the query templates with
  the D1 decision either way.
- The probe stays dry by default; --execute uploads, queries, drops
  the temp table, and writes the report. The bar is unchanged: the
  exact selectable total ≥ 5,900 → D1, else D2. The prefix-projection
  alternative is not taken.

### D0.6 Bounded D1 design (probe evidence, pre-fetch)

M3 live (2026-09-16, reports/probe-m3-live.json): 422,610 of 994,693
fresh s4–s6 candidates (42.5%) hold root captures; capped pool 740,022,
takeable ≈ 527k vs the 5,900 bar → D1, at 53.8 GB scanned (~$0.27).
The pool is ~90× what D1 needs, so the constraint is no longer finding
roots but selecting from a huge pool without drifting the population.
"To exhaustion" (D0.4) is withdrawn for the fetch: exhausting s4–s6
would pull 898,706 s6 candidates out of 994,693 and maximise the
strata-imbalance caveat. What follows replaces it, recorded before any
fetch query runs.

### D0.6.1 Join-based bounded fetch

The D1 fetch uses the join, never per-domain queries (per-domain
enumeration priced ~2M queries / ~$493 upper bound in the v1 dry run).
One JOIN per crawl over an uploaded seeded domain sample (§D0.6.2),
bounded inside the SQL: url_type via CASE over the same regex family
as the probe (root `^https?://[^/]+/?$`, query `%?%`, path1
`^https?://[^/]+/[^/]+/?$`, else pathN), 200-only, then
`row_number() OVER (PARTITION BY domain, url_type ORDER BY
xxhash64(url, sample_seed))` keeping rn ≤ 6 per (domain, type) — the
pipeline's deterministic hash family, no Athena sort cost, no
earliest-first bias. Output goes to Parquet in S3
(s3://phishnet-athena/cc-fetch-<run-id>/, partitioned by stratum),
never appended to the JSON cache; a manifest (queries, row counts,
per-part sha256) is committed and select consumes
banked-JSON + wave-Parquet with both recorded in provenance.
Expected cost is single-digit dollars (narrow columns, two crawls);
measured bytes replace the estimate in the fetch manifest.

### D0.6.2 Stratum weights and bounded samples

Stratum weights stay as originally designed. s1–s3 are exhausted, so
their contributions stay fixed at whatever the banked pool yields
under the §D0.6.3 caps — no new fetch there. s4–s6 split the remainder
of the 40k in the 12k proportions (s4 44.8 / s5 31.9 / s6 23.3,
summing to 100.0), never in proportion to candidate counts.
The fetch takes a seeded bounded sample per stratum: the first D_s
fresh domains in fetch-identical replay order (seed 0, §probe replay),
with D_s = ceil(2 × rows_s / 4) against a remainder floor of
40,000 − 4,348 (s1–s3 banked rows under old caps, an upper bound since
tighter caps only shrink it): D_s4 = 7,963 (all fresh s4),
D_s5 = 5,687, D_s6 = 4,154 — ~17.8k domains, one wave, no refills.
If a stratum sample underfills, N = min(40k, pool bound) still
governs and the power rules below apply; there is no second sample.

### D0.6.3 Domain-count power

Power is set by domain count, not row count: the FPR verdict uses the
wider of Wilson and the domain bootstrap, and at ~11 rows/domain the
bootstrap interval dwarfs Wilson's ±0.10pp even at 40k rows. With a
90× pool this is fixable, so for the D1 select the main per-domain
row cap is 4 (replaces 16; the per-domain-type cap stays 6, inert
below a total of 4 — recorded, not removed; per-eTLD+1 stays 25;
hosted per-tenant caps unchanged). 40k rows then span ≥ 10k domains
in the ideal pack. Floors, on non-hosted registrable domains
(hosted rows cluster on platforms and report separately per D0.3):
overall ≥ 8,000 distinct benign domains, test band ≥ 3,500. A miss
downgrades the 0.5% claim to indistinguishable-expected with the
pre-registered 1% point carrying resolvability — recorded, no
re-selection. Effective-n statement: the 15k-row floor is necessary
but not sufficient; a met/unmet 0.5% verdict additionally requires
domain-bootstrap half-width ≤ 0.10pp at FPR 0.5% on the post-cap test
set, computed before any threshold verdict. Lower caps squeeze roots
(roots fill last): the root take under the new cap is judged inside
the single select+gate run, never in a separate tuning step.

### D0.6.4 Length bands (adopted)

Length alignment is a selection constraint because the gate registers
it: within each URL type, benign picks fill length-band quotas at the
per-type terciles of D0.1 non-hosted phishing URL length
(len(normalised URL) — the exact string the gate measures), 12 band
quotas total, edges pinned in provenance. This is adopted on
principle, before seeing the pool's length distribution. Depth needs
no new mechanism (stratified type quotas ARE the depth control). If
length still fails gate-once, D2 draws on the same kind of pool and
will likely fail the same way — so a length failure falls through to
D4, recorded now rather than discovered later.

### D0.6.5 M1 citation hygiene

The regenerated reports/m1-stratified.json (validator --mode
stratified: six main failures, 205 hosted rows partitioned out)
supersedes the TEMP-script figures from e714041e everywhere. Verified
by search: no doc, report, or test cites the superseded numbers; D0.2
cites only the unstratified M1 (drift ≤ 0.003, scheme 0.0026, depth
0.0096), which re-verified clean this session.

### D0.7 Review corrections (pre-merge, pre-fetch)

Each item supersedes the cited D0.6 lines; D0.6 stands otherwise.

### D0.7.1 SQL fix (supersedes the D0.6.1 regexes)

The root regex `^https?://[^/]+/?$` matches query-bearing roots
(`[^/]+` absorbs `?a=1`), and the path1 regex shares the flaw. Measured
impact on M3: zero — the banked cache holds no query-bearing
root-forms among 16,058 old-regex hits, so the D1 decision is
unaffected; fixed anyway because a million-domain wave is where rare
forms stop being rare. Fix: host/segment classes exclude `?`
(`[^/?]`), fragments tolerated (`(#.*)?` — normalise strips them, so a
fragment-bearing root still selects), and the CASE tests query first by
strpos (belt and braces; the classes already make the order
irrelevant). The full CASE is rendered from one source
(`probe render_type_case()`): query → root → path1 → pathN (valid
scheme+host) → malformed. Parity vs url_type over all 4.76M banked
records: 99.92%; the residual 0.08% is bare-trailing-`?` (SQL query vs
url_type path — drains the root pool, never feeds it), semicolon
params, double-slash paths, and uppercase schemes, all pinned in
tests/test_cc_probe.py. The 19 root-misses are conservative.
Hash order: the seed rides inside the hashed input
(`concat(url, '|', sample_seed)` — Athena's xxhash64 takes no seed
argument), ORDER BY the identical abs(signed-int64) integer the
sampling predicate thresholds on; Trino-stock-XXH64 pinned by vector
test; the fetch manifest records sample_seed. "Fetch-identical replay
order" is therefore checkable offline.

### D0.7.2 Sizing against 40k (supersedes the D0.6.2 numbers)

The D0.6 bound ran the wrong way: 4,348 is an upper bound on s1–s3, so
35,652 is a lower bound on the need — useless for sizing a sample.
Size against the full 40,000 in the 12k realized row mix (wording
corrected: "as originally designed" is struck — the original design
was equal domain quotas, the 44.8/31.9/23.3 mix is what is used,
chosen for continuity with a gate-passing corpus): s4 17,920 / s5
12,760 / s6 9,320 rows. Neither 2× nor the banked 64% is the basis
for the margin: the only fresh-domain measurement is M3's 42.5%
root-productivity (a lower bound on row-productivity), and fill under
12 length bands is lossy — so D_s = ceil(4 × rows_s / 4) with that
basis: D_s4 = 7,963 (all fresh s4), D_s5 = 12,760, D_s6 = 9,320
(~30k domains, one wave, no refills). Expected yield: 30,043 × 0.425
≈ 12.8k + 717 s1–s3 ≈ 13.5k productive domains against the 8,000
floor. Underfill rule, chosen now: N shrinks, weights hold — a
stratum shortfall never moves to s5/s6. Select invariant
(implemented + tested post-merge, stated now): consume domains in
seeded replay order and stop when full, so a larger pool never becomes
a best-fit search.

### D0.7.3 Cap 4 with expected downgrade (supersedes D0.6.3)

Decision: per-domain cap 4, and the 0.5% downgrade is the expected
outcome — not cap 2. A roots-last cap-4 simulation on the banked pool
(TEMP, existing artifacts only): 742 roots vs 3,735 at cap-16 — the
squeeze is real, but transferred to the fresh pool it floors fresh
takeable at ~99k vs the 10k root quota (10× margin; the live 527k
assumed cap-16-like fill and the D1 verdict survives the correction
either way). Cap 2 is rejected: at m≈2 it still needs ρ≤0.05 to clear
0.10pp, while risking N<25k via band-quota starvation on 2-slot
domains — asymmetric risk for no verdict gain. Expected half-width
under cap 4: test ≈20k rows over ~6.7k domains (m≈3) → ±0.11–0.14, so
the downgrade triggers and the pre-registered 1% point carries
resolvability. Caps enforce the floors arithmetically at full fill
(40k rows at ≤4/domain ⟹ ≥10k domains; test ≥5k); the floors bite
only on underfill. Verdict-order wording fixed: within one step,
compute the achieved FPR and its bootstrap half-width, then assign
met / unmet / indistinguishable by rule — no separate earlier
computation.

### D0.7.4 Quartiles, scheme string, D2 scope (supersedes D0.6.4)

Terciles are too coarse where it matters (roots have a narrow length
range; path1/query tails are widest past the tercile edge): quartile
bands, 16 quotas, fill checked in the same single select-and-gate
run. Quintiles rejected (fill pressure on top of the cap-4 squeeze).
Scheme handling: band edges and the validator gate both measure
len(normalised URL with scheme as emitted); the canonicalize decision
affects featurization only, and the 1-char http/https difference is
absorbed automatically since both sides use the identical string.
D2 designates the banked pool only (pre-wave) — there is no
combined-pool fallback; a D1 length failure would reproduce on the
combined pool. D4 fallthrough unchanged.
