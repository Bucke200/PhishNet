# Model card — Phase 3 champion (row a; age conditional)

Scope: what the numbers support, and the failure modes that bound them.
Full evidence in `reports/phase3.md`; serving in `reports/phase6.md`;
adversarial results in `reports/phase5-adversarial.md`.

## Intended use

A browser-extension URL scorer with a human-visible disposition, not an
autonomous block/allow gate. Tier 1 scores the URL string only (no page
content); Tier 2 reads a page extract for in-band rows. It is evaluated at
fixed operating points and must never be threshold-tuned on test. It is a
research/demo artifact, not a product: no monitoring, no feedback pipeline,
no hosted deployment, no retraining between phases.

## Inverted depth prior and the scheme decision

The corpus carries shape confounds that are recorded, not tuned away.
Benign paths average 1.75 deep against 1.03 for phishing, and benign URLs
are 97.6% https against 77% — the opposite of the usual "phishing is deep"
intuition (`docs/splits-eval-audit.md`). The scheme is a label proxy by
construction, so the champion strips it (`canonicalize_scheme`, the split
manifest's `manifest:drop` rule; benign https 0.9873 vs phish 0.8390, gap
0.1483) and `is_https` reads constant 0 and is dropped. The depth prior is
left in and reported; re-sampling benign by depth to "fix" it would
contradict the split rules.

## Point-in-time classification

Every enrichment filter is relative to a per-row timestamp, never to now
(`docs/point-in-time.md`). Phishing uses `first_seen` (submission time for
PhishTank, observation time for OpenPhish); benign uses the
capture/crawl stamp. The headline includes the ~1,063 OpenPhish
`unknown`-stratum rows — live phish of unknown, possibly long-lived age —
with per-stratum slices beside it, a choice pre-registered before the
numbers existed.

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

## Age: gate failure and the conditional result

Domain age is **ineligible for the headline**: its test-band unknown gap is
0.059 against the 0.05 budget, and the failure is benign-heavy (long-tail CC
domains fail lookups more often than phishing ones, mostly WHOIS records
without a parseable creation date). Reported alongside instead: the paired
recall lift of row (b) over row (a) is **+0.22–0.34** on all rows and
**+0.28–0.33** on age-known rows only, labeled conditional, with the mix
shift stated — row (a) itself falls from 50.4% to 39.5% on that subset.
There is no benign-side advantage in the s4/s5 strata, so age's value is
phishing recall, not benign protection.

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

Tier-1 serving shape p50 **0.45 ms** (criterion 12 met, Phase 6). The
earlier 14.3 ms figure attributed the cost to the extractor; it was per-call
pandas frame construction plus the sklearn wrapper. The serving fast path
(dict → preallocated row → `booster_.predict`) is bit-equal to the headline
scorer. The extractor alone is 0.29 ms; stub negligible.

## Host and TLD reputation behavior (Phase 5 lexical arm)

Tier 1 behaves largely as a host and TLD reputation model: it collapses on
unseen hosts (random `.example` hosts recall 0.03 at the fixed 0.5%
threshold; open redirects 0.00) and flags known-phishy host features
whatever the page is (covered-shortened benign links alert at 0.99 against
a 0.005 clean baseline). Same family as the takedown and
source-composition findings above: the label leaks through where the URL
was drawn from, not what it says. `.example` is the extreme case (not a
registrable TLD); the realistic version — rare real gTLDs — is future
work, not this phase. Full numbers in `reports/phase5-lexical.md`.

## Phase 4 close-out — the LLM layer is bounded and unanswered

The recorded three-repeat sweep was not run (Developer tier unavailable;
free tier ~100 calls/day at page-sized token volumes). The phase question —
whether the LLM layer beats the password baseline — is **unanswered, not
negative**, and is presented that way everywhere. Nothing generalizes
beyond `openai/gpt-oss-120b` on Groq (`phase4-A`). Close-out facts:

- **Fetchability is a label proxy:** test phish fetch 0.135 vs test benign
  0.886 (Step-0 marginals). The takedown filter already selected for live
  phish; the fetch selects again.
- **Structural ceiling** (sealed Step-0 data, no LLM call): fetched in-band
  test phish 132/3,799 = **0.0347** is the most recall the layer could ever
  add; benign FPR exposure is 969/21,020 = 0.0461. This is why the cascade
  reads "indistinguishable" before a reader asks.
- **Fingerprint rotation:** `system_fingerprint` rotated to 35 values over
  101 calls; run identity is model+prompt+seed 0, explicitly weaker
  (`phase4-C`).
- **Determinism 11/50 = 22%** (over the 5% bar): 10 free-tier quota failures
  plus one genuine phishing→suspicious wobble. The **response cache** — not
  the seed and not the temperature — is what makes the published numbers
  reproducible.
- **Cost:** p50 1,308 ms / p90 2,069 ms; provisional forecast $0.367 per
  1,000 escalated rows at Groq listed rates (2026-09-18).
- **Scope ends at `phase4-D`.** No number here publishes; the recorded sweep
  is future work under its own amendment.

## Tier-2 LLM Cascade Limitations (Phase 5 Adversarial Evaluation)

Offline evaluation of the two-tier cascade under adversarial injection
probes across 564 sealed calls (`openai/gpt-oss-120b`, 3 cold repeats)
identified five systemic operational limitations:

1. **~12% Schema Fail-Open Defect on Phishing:**
   Provider-side strict JSON schema validation rejected 8.9% of all calls
   (50/564) with HTTP 400. Crucially, the error rate is concentrated on
   phishing pages: **11.9% (15/126)** on clean phishing and **11.7% (35/300)**
   on injected phishing, against **0.0% (0/90)** on clean benign and **0.0% (0/48)**
   on framing. 47 of the 50 errors occurred because the frozen schema's
   `credential_types` enum omitted `login` and `credentials`. Sealed calls
   deterministically retain Tier-1 score (`verdict: None`), meaning the cascade
   fails open on credential-harvesting pages ~12% of the time, swamping prompt
   hardening effects at the cascade level.
2. **20.0% Samehost Login False-Alarm Rate:**
   Across all 6 sweeps, clean benign false-alert rate was exactly **20.0% (3/15)**,
   all on `samehost_login` base pages (`clean-benign-login-northvale`,
   `clean-benign-login-parcelyn`, `clean-benign-login-tesserapay`). Because
   fictitious company names do not resolve to known hosts, the model judged
   ordinary password authentication forms as phishing. Since ~5% of benign traffic
   falls into the escalation band (0.65 - 0.93), a 20% login false alarm rate
   would exhaust the entire 0.5% cascade FPR budget.
3. **Pure-Function Detector Trivially Evadable (14/14 First Attempt):**
   The frozen pure-function regex detector achieved 5/6 recall on isolated
   payloads but only 27/42 (64.3%) on held-out page extracts (0/5 on
   `form_placeholder`, 0/5 on `link_dilution`, 0/7 on `P-authority-1`). Under
   adaptive red-teaming, **14 of 14 aware candidate rewrites** (synonyms,
   paraphrases, split tokens) bypassed the detector on the author's very first
   attempt with 0 discards and 0 quality rejections.
4. **Retain Policy Unsafe (Must Not Ship):**
   Under the pre-registered Retain mapping (retaining Tier-1 score on detector hits),
   flagged evasion attacks escape detection, surging ordinary cascade evasion
   from 16.0% (17/106) to **62.3% (66/106)** across repeats (diff **-0.4623 [-0.6373, -0.2843]**).
   Retain is structurally unsafe and must never ship in production.
5. **Escalate Framing False-Alarm Cost (50.0% [4/8], Descriptive Lower Bound):**
   While the Escalate policy passes Criterion 1 at the cascade level
   (reducing evasion from 16.0% to 0.0%), it immediately lifts **50.0% (4/8)**
   of benign framing pages to alert because the detector triggers on injection
   keywords within benign text. With only 8 reaching framing pages (futility
   floor triggered), this 50% rate is a descriptive lower bound on operational
   framing vulnerability. Escalate cannot be deployed without resolving framing.
