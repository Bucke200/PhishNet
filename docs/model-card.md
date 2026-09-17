# Model card — Phase 3 champion (row a; age conditional)

Scope: what the numbers support, and the failure modes that bound
them. Full evidence in `reports/phase3.md`.

## Survivorship

The phishing feed is takedown-filtered before collection
(PhishTank `online-valid` holds only live phish; OpenPhish is a live
snapshot). "Fresh" means caught within 2 days of submission, not all
phishing born in the window. RDAP 404 and re-registration counts are
takedown measured during collection (see below) — the population
understates short-lived phish by construction.

## Tranco selection leak

Every benign row was sampled from Tranco 46VQX, so rank encodes the
sampling frame. Row (e) finds long-tail tiers NOT consistently closer
to phishing on hostname shape — no popularity-artifact support — but
that does not clear the leak: rank never enters any trained row. And
row (e) tested shape, not age: benign domains are old by Tranco
construction, so part of the age lift may be sampling, not signal.
Checked per benign stratum — row (b)'s benign FPR runs slightly ABOVE
row (a)'s in s4/s5 (0.43%/0.42% vs 0.31%/0.35%): no benign-side
advantage in the low strata; age's value is phishing recall, partly
paid in benign-tail FPs.

## Hosted coverage limits

- No benign coverage for `pages.dev` (0/2,000-stratum rows arrived);
  hosted FPR Descoped to counted cells only.
- No hosted-root FPR claim: hosted roots are 646 phish vs 2 benign
  (model recalls 99.8% of them — platform-shaped, read beside the
  platform-prior PR-AUC 0.769).
- Hosted pathN benign FPR 0.2% (n=509); hosted path1 benign FPR 6.3%
  (n=126, coarse).
- Actor-disjointness not claimed for hosted rows: tenants are
  separate, actors may repeat across tenants.

## Cold start

Losing age costs ~25pp recall (78→53%) and triples FPR
(0.56→1.59%) at the fixed 0.5% threshold; fresh slice mirrors. Served
cold traffic (first visits) lives at the 100%-miss end. This sizes the
Phase 4 escalation band, in calibrated-score space.

## Threshold transfer

Era-matched calib fixes transfer at 0.5% (drift 0.06pp < Phase 2's
0.10pp) but not at 1% (0.19pp). Calibration shelf life is months at
best; thresholds must be refit on recent data, never swept on test.

## Why CT was dropped

crt.sh's documented limit failed ~97% of checkpointed lookups; no
validated alternative was provisioned; the feature is unservable at
request time. Dropped unmeasured (Amendment E), not failed. A future
population collected with CT from the start may revisit it.

## RDAP 404 rate by class (takedown evidence, either way)

Age-pass 404s: phishing-dominated (678 total; test-band
re-registrations 407 phish vs 6 benign, train 1,943 vs 5).
`age_known` carries the takedown signal — which is why the
contamination gate, not judgment, decides its eligibility (test gap
0.059 > 0.05: ineligible; train gap 0.016: pass).

## Latency

Tier-1 serving shape p50 14.3 ms (criterion 12 unmet): ~7.8 ms fixed
per-call extractor overhead, stub negligible. Sub-millisecond batched;
single-URL blocking is the honest number.
