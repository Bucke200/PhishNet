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
  `is_hosted_tenant`, and at this hosted share the flag alone is
  likely a strong feature. The report states this outright so nobody
  reads the row-(a)-versus-Phase-2 gap as enrichment or as regression.

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
