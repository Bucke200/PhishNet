# PhishNet roadmap — revised (2026-09-17, Phase 3 closed)

The original plan, rewritten against what actually happened. Phases 0–3 are
records now. Phase 4 onward is the remaining work.

PhishNet is a **project, not a product**. The deliverable is a set of findings
that can be defended in detail, plus a working demo. It is not a service that
runs indefinitely. Phases 4–7 are cut to match: infrastructure that exists
only to keep a service alive moves to future work.

---

## Phase 0 — De-risk and strip ✅ done

All of it landed: credentials rotated, the installer and `read/` removed,
dependencies pinned exactly, `pyproject.toml` plus uv, the two
`feature_extraction.py` copies merged into `src/phishnet/`, and ruff, mypy
and pytest running in CI.

Beyond the original scope:
- `model_manifest.json`, with SHA256-pinned release artifacts;
- `repro/hashes.json`, with byte-exact rebuild verification;
- `docs/WAIVERS.md`, recording two populations that can never be regenerated
  byte-for-byte.

The waiver discipline mattered more than anything else in this phase.

---

## Phase 1 — Evaluation harness ✅ done

The split is temporal on the phishing side and hashed by registrable domain on
the benign side, with no per-URL random splitting and no rebalancing.
`eval.py` takes any predictor that implements `name` and `score`, and emits a
fixed report. A `url_shape_canary` predictor audits every corpus build for
shape leakage.

**The bare-domain leak was real, and it is quantified.** The legacy model's
ROC-AUC differs by 0.16 between populations: 0.8714 on the old
Tranco-bare-domain corpus, 0.7108 on CommonCrawl deep links. Retrained on the
clean corpus, the difference is 0.0003 (0.9100 vs 0.9097). That invariance is
the proof, and it is a stronger result than any accuracy gain.

**Recorded, not fixed:** `data/splits-eval` carries `verdict: suspicious`.
- A shape-only model reaches ROC-AUC 0.753.
- Benign path depth averages 1.75 against 1.03 for phishing.
- Benign URLs are 97.6% https against 77% for phishing.

Resampling the benign side by depth to fix this would mean per-URL
subsampling to hit a metric, which contradicts the split rules. The
deliverable is therefore a recorded refusal with the measurements attached,
in `docs/splits-eval-audit.md`.

---

## Phase 2 — Honest probabilistic baseline ✅ done, one criterion unmet

**The whitelist is gone.** It short-circuited about 30 domains before the
model ran, in serving only (`predictors.py` never had it). So the published
numbers described a pipeline the extension didn't run. An invariant test now
blocks re-adding it.

**Hard voting, not model quality, was the limiting factor.** Four models
voting hard produce scores in `{0, .25, .5, .75, 1}`. No threshold on that
scale can meet a 0.5% FPR budget, so every operating point read 0.00% recall.
Soft voting with the *same weights*, without refitting, produced the first
usable operating point:

| | hard vote | soft vote |
|---|---|---|
| v1 assets | PR 0.6485 · R 0.00% | PR 0.7795 · R 14.19% |
| CC assets | PR 0.7225 · R 0.00% | PR 0.9220 · R 56.80% |

**A single LightGBM ties soft voting on ranking** (PR-AUC 0.9236 vs 0.9220,
overlapping intervals). Its real advantages are calibration and speed: 3.2 ms
against about 31 ms.

**Calibration does not transfer across the temporal cut.**
- **Isotonic:** in-sample ECE improved from 0.095 to 0.003, but test ECE got
  worse (0.15 → 0.26).
- **Platt:** with only two parameters, it failed the same way on test. A
  low-capacity fit failing identically points to era drift, not overfitting.

The uncalibrated model stays champion.

**Threshold transfer was unresolved here:** a threshold fixed in advance gave
58.93% recall at 0.60% FPR, against a 0.5% budget recorded as unmet. Swept
numbers (58.01% at 0.49%) are labeled unattainable beside it. SHAP values ship
through LightGBM's native `pred_contrib`; the serving path returns 501 for
explanations until the model migration lands.

---

## Phase 3 — Signals the URL string can't give you ✅ done, 11 of 12 criteria met

Protocol: `docs/phase3-preregistration.md` (Amendments A–E). Results:
`reports/phase3.md`. Review unit: tag `phase-3-close`. Every change to the
plan is recorded as an amendment made before the numbers it affects existed.

### Results

**Headline (row (a), lexical + `is_hosted_tenant`, thresholds fixed on the
calib band):**

| target | recall | achieved FPR | verdict |
|---|---|---|---|
| 0.5% | 50.4% | 0.40% | indistinguishable |
| 1% | 60.7% | 0.98% | indistinguishable |

**Domain age is ineligible for the headline.** Its test-band unknown gap is
0.059 against the 0.05 budget, and the failure is benign-heavy: long-tail CC
domains fail lookups more often than phishing ones, mostly WHOIS records
without a parseable creation date. Reported alongside:
- paired recall lift of row (b) over row (a): **+0.22–0.34** on all rows;
- **+0.28–0.33** on age-known rows only, labeled conditional, with the mix
  shift stated (row (a) falls from 50.4% to 39.5% on that subset);
- no benign-side advantage in the s4/s5 strata, so age's value is phishing
  recall, not benign protection.

**Cold start:** losing age costs about 25 points of recall (78.0% → 53.4%) and
triples FPR. The 53.4% full-miss figure sits close to row (a)'s 50.4%, which
is the consistency check.

**Threshold transfer:** indistinguishable at both operating points under the
wider-interval rule. The point-estimate rule that would have read "fixed" is
recorded as superseded.

**Calibration band shape:** the mixture audit reads 0.815 (suspicious) while
the main stratum reads 0.700 (ok). The hosted mixture explains it, which is
direct evidence for the stratified gate adopted in Amendment D.

**Criterion 12 unmet:** tier-1 p50 is 14.3 ms in serving shape, against a
single-digit target. The cause is about 7.8 ms of fixed per-call extractor
overhead (0.3 ms batched); the stub provider is negligible. Phase 6 owns it.

### What was found along the way

- **The phishing feed is filtered by takedown before collection.** PhishTank
  `online-valid` holds only phish still live on the snapshot day.
  - **Train era:** 91.6% https, 20.5% on free hosting platforms.
  - **Test era:** 78.9% https, 13.3% on free hosting platforms.

  Every phishing row carries a survival-lag stratum: fresh, short, long, or
  unknown.
- **Tranco rank is a label leak by construction.** Every benign row was
  sampled from Tranco. Rank appears only as a labeled diagnostic row.
- **Phishing is a mixture of two populations:** hosted tenants (21% of
  phishing, 80.1% root URLs) and everything else (25.0% root URLs). The shape
  gate is therefore stratified on `is_hosted_tenant`, which is itself a model
  feature: the main stratum blocks promotion, the hosted stratum is
  descriptive.
- **Three refusals on record:** the 40k enlargement failed the unstratified
  gate; the 12k corpus passes unstratified but fails the stratified gate on
  six metrics; age fails its contamination gate. The stratified gate was
  adopted even though it ruled out the cheapest fallback.
- **Certificate history was dropped unmeasured (Amendment E),** for three
  reasons: crt.sh limits requests to about 5 per minute per IP, which made the
  first enrichment run fail 97% of CT lookups; no alternative source was
  validated; and the feature could not be served at request time anyway.
- **Hosted results are descriptive.** Hosted benign covers 18 platforms, and
  hosted roots have 2 benign rows, so no hosted-root FPR claim is made. The
  platform-prior baseline reaches PR-AUC 0.769 against the model's 0.979, so
  platform identity explains much but not all of hosted recall.

### Bugs caught by the protocol

The wave-intake suffix bug (a 0-row select, discarded ungated), the 97% crt.sh
throttling that forced Amendment E, and a threshold bug where row (a) was
scored with row (b)'s thresholds. The last one surfaced as an implausible
"unmet" verdict and changed the headline once fixed.

---

## Phase 4 — The LLM layer, evaluated offline (closed without the recorded sweep)

**Status (2026-09-18, amendments `phase4-A`–`phase4-D`): closed on sealed
provisional data. The recorded three-repeat sweep was not run — the
Developer tier is unavailable and the free tier (~100 calls/day at page-sized
token volumes) would take two to four weeks. The phase question, whether the
LLM layer beats the password baseline, is unanswered, not negative, and is
presented that way everywhere. Nothing generalizes beyond
`openai/gpt-oss-120b` on Groq (substituted for Gemini/NIM by `phase4-A`:
Production catalogue, open weights, strict `json_schema`, seed +
`system_fingerprint` instruments, same model string for dev and record).

The LLM reads what the URL string can't show: whether the page asks for
credentials, which brand it imitates, whether its text pressures the user, and
whether the page's apparent identity matches its domain. Playwright does the
fetching; the model does the judging.

**Hosted API, no local model.** Ollama is dropped (hardware constraint,
recorded as a deviation). Pin one provider and one model string, set
temperature 0, and record both in `asset_fingerprint`.
- **Primary:** Gemini via Google AI Studio — first-class response schemas, and
  rate limits that apply per project (check AI Studio for the live numbers).
- **Second opinion (optional):** NVIDIA NIM, OpenAI-compatible, free starting
  credits. Verify schema support per model first. An agreement check between
  two hosted models replaces the dropped local comparison.
- **Free tiers work at this volume** (hundreds of calls), but token-per-day
  ceilings bind before request ceilings, free capacity can disappear without
  notice, and failed attempts still count against quotas. Develop on a free
  model; make the recorded run on a pinned paid model.

**Step 0 — measure fetchability first, and decide on it.** Most train-band
phishing URLs are months old and gone. Fetch the sample, report the success
rate by class, era and survival stratum, then choose:
1. **Verdict-as-report (default):** the LLM's JSON is applied to escalated
   rows and reported for agreement with the label. No retraining, no joined
   feature.
2. **Joined feature:** retrain with the LLM fields as a feature group, only if
   enough train-band pages survive. Coverage and survivorship bias reported
   beside every number.
3. **Forward collection:** snapshot pages as new phishing URLs arrive, for a
   later phase.

Register the choice and its trigger before fetching.

**The rest of the design:**
- **Tier 1 is the uncalibrated Phase 3 champion, row (a).** Age is ineligible,
  so escalation cannot depend on it.
- **Band edges fixed on calib,** in the score space that ships, with achieved
  escalation rates reported on test. Never swept.
- **Budget against the cold-start escalation rate,** not the warm one.
- **Freeze the page snapshots.** Fetch once, store HTML plus a token-bounded
  extract (title, visible text, form fields, link hosts), hashed. Every
  evaluation runs against the snapshots.
- **Send the extract, not raw HTML.** It keeps token ceilings clear and makes
  the injection surface explicit for Phase 5.
- **Cache responses** on snapshot hash + prompt version + model string, and
  seal raw requests and responses in a run store, so results survive the
  model's retirement.
- **Constrain the output:** JSON through a response schema, never regex on
  prose. The verdict is an input, never an override.
- **Report cost and latency per 1,000 URLs,** measured, not estimated.
- **Close-out numbers (provisional 68/1,106 sweep, `reports/phase4.md`):**
  cascade vs Tier-1 indistinguishable at both thresholds under both
  unfetchable policies; structural ceiling 132/3,799 = 0.0347 recall add at
  most, FPR exposure 969/21,020 = 0.0461; determinism 11/50 (22%, over bar —
  10 quota failures + 1 wobble); fingerprint rotates per call (`phase4-C`);
  priced forecast $0.367/1k escalated at Groq listed rates, dated.
- **Forward workflow** (`.github/workflows/phase4-forward.yml` on master):
  checks out tag `phase-4-close` so the fetcher is always the Step-0 one;
  outputs commit to the `forward-p4-data` branch, never master. First run
  2026-09-18 succeeded (phish arm skipped, no key; benign recorded; snapshot
  no-due-rows); appendix in the Phase 4 prereg.

---

## Phase 5 — Adversarial hardening ✅ done

Protocol: `docs/phase5-preregistration.md` (Amendments `phase5-A`–`phase5-I`).
Results: `reports/phase5-adversarial.md`, `reports/phase5-adversarial.json`,
`reports/phase5-lexical.md`. Review unit: tag `phase-5-close`. 564 calls total
across 3 cold repeats on `openai/gpt-oss-120b` (pinned extractor, frozen `p5-h1`
prompt, frozen pure-function detector, sealed run store).

### Headline Results and Arm Verdicts

| Arm / Policy | Population ($N$) | Evasion Rate (Pooled) | Paired Diff vs Unhardened Baseline | 95% Bootstrap CI | Criterion 1 Verdict | Operational Shippability |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Unhardened Baseline** (`p4-v1`, no detector) | 106 | 16.0% (17/106) | Baseline reference | — | — | Vulnerable to delimiter injection + schema errors |
| **Hardened + Escalate** (`p5-h1` + detector + escalate) | 106 | 0.0% (0/106) | **+0.1604** | `[0.0714, 0.2679]` | **PASS** | **Do not ship:** 50% framing rate; detector trivially evadable |
| **Hardened + Retain** (`p5-h1` + detector + retain) | 106 | 62.3% (66/106) | **-0.4623** | `[-0.6373, -0.2843]` | **FAIL** | **Strictly unsafe:** attacker triggers detector to fail open |
| **Prompt-Only Ablation** (model level, errors excluded) | 94 | 0.0% (0/102) | **+0.1064** | `[0.0222, 0.2125]` | (Ablation PASS) | Neutralizes delimiter tags when valid JSON emitted |
| **Prompt-Only Ablation** (cascade level, errors = evade) | 106 | 6.6% (7/106) | **+0.0943** | `[-0.0192, 0.2115]` | (Ablation FAIL) | Crosses 0: swamped by ~12% schema fail-open defect |

- **Criteria 2 & 3 Met:** Clean catch fall $\le 0.05$ (max fall -0.0476, actually improved); clean false-alert rise $\le 0.05$ (rise == 0.0000 across all repeats).
- **Criterion 4 Met:** Ordinary eligible $N \in \{36, 32, 38\} \ge 20$.
- **Criterion 5 Futility Floor:** Framing sample $N=8 < 20$ floor; descriptive fallback applied (Escalate 50.0% [4/8], R1 prompt 12.5% [1/8]).

### Core Findings Uncovered

1. **The ~12% Schema Fail-Open Defect:** 50/564 calls (8.9%) failed strict schema validation (`status: 400`), identically on clean phish (11.9%) and injected phish (11.7%), with 0% on benign. 47/50 errors occurred because `credential_types` omitted `login` and `credentials`. Under Phase 4 §2, sealed runs fell back to retaining Tier-1 score, failing open on credential harvesters. Escalate achieved 0/106 because all 7 hardened errors were on detector-flagged pages, lifting them anyway.
2. **20% In-Band Login False Alarms:** Tier 2 produced a 20.0% false-alarm rate on clean benign authentication forms (`samehost_login`) because unrecognized corporate names were judged as phishing. In production, this would consume the entire 0.5% cascade FPR budget.
3. **Pure-Function Detector Trivially Evadable (14/14 First-Attempt Bypass):** The frozen regex achieved 64.3% recall on page extracts; adaptive rewrites bypassed it on the first draft with zero discards.
4. **Lexical Arm Findings:** Punycode (`xn--`) was robust (49.2% recall at $t_{0.5\%}$, paired diff `[-0.1117, +0.0447]`). However, shorteners exposed a massive feature leak (`is_shortened` flagged 99.2% of benign shortened links), and synthetic `.example` domains collapsed to 3.0% (redirect collapse was an artifact of unseen TLDs).

---

## Phase 6 — Serving, demo, and production hardening ✅ done

Protocol: `docs/phase6-preregistration.md` (Amendments `phase6-A…F`).
Results: `reports/phase6.md`, `reports/phase6.json`. Review unit: tag
`phase-6-close`.

Informed by the production gaps identified across Phases 1–5
(`docs/production-gaps.md`), Phase 6 built a reproducible, honest serving
path and resolved the architectural blockers:

- **Champion servable (C1):** the container now serves the Phase 3 row (a)
  LightGBM, not the legacy hard-vote urlset ensemble. `/predict` is
  bit-equal (max abs diff 0.0) to the headline scorer on all 13,157 calib
  and 24,819 test rows. `/health` + `/predict` + `/explain`; `/report` and
  MongoDB are removed.
- **Latency (C2):** criterion 12 is met at 0.45 ms p50 (from 14.3 ms). The
  fix is a pandas-free fast path, **not** the Cython/Rust extractor that the
  earlier roadmap named; `production-gaps.md` §7 is withdrawn.
- **Fail-closed Tier-2 (C3):** the `p5-h1` prompt plus the frozen detector,
  with every non-valid outcome alerting. The Retain policy is unreachable
  and blocked by an invariant test.
- **`p6-v1` schema (C4):** `credential_types` widened; the 50 sealed Phase 5
  schema failures replay to a 0.0 after-error rate.
- **Shortener handling (C5):** follow the redirect and score the final URL;
  stripping `is_shortened` is rejected as serve-time skew. No FPR claim.
- **Demo (C6):** extension shows disposition, score, and native SHAP;
  Tier 2 runs from the sealed cache by default, live Groq opt-in.

Not produced in this environment: the container build and the browser
recording (no Docker daemon / GUI). Both are delivered as runnable
artifacts and recorded as operator-run in `reports/phase6.md`.

**Cut, and tracked in `docs/production-gaps.md`:**
- Live external RDAP provider with Redis cache and timeouts;
- Prometheus monitoring stack and production alert dashboards;
- User feedback poisoning mitigation pipeline;
- Scheduled weekly calibration refresh jobs.

---

## Phase 7 — Package it so it reads correctly (2–3 days)

Protocol: `docs/phase7-preregistration.md` (Amendments `phase7-A…`).
Results: `reports/phase7.md`. Review unit: tag `phase-7-close`.

- **README.** Lead with the fixed-threshold numbers (50.4% at 0.40% FPR), with
  swept numbers labeled unattainable beside them, and the cold-start number
  next to the headline. Phase 4 is stated as bounded and unanswered, with
  the 0.0347 structural ceiling beside it — never as "the LLM didn't help."
- **Model card** (already started), carrying:
  - the inverted depth prior and the scheme decision;
  - point-in-time classification;
  - survivorship in the phishing feed, with the RDAP 404 rate by class;
  - the Tranco selection leak and the Tranco-age confound;
  - hosted coverage limits;
  - age's gate failure and the conditional result;
  - cold-start degradation;
  - calibration's shelf life and the threshold-transfer verdict;
  - why certificate history was dropped.
  - Phase 4 close-out additions: fetchability as a label proxy (test phish
    0.135 vs benign 0.886); fingerprint rotation and the weaker run
    predicate (`phase4-C`); 22% determinism, with the response cache — not
    the seed — as what makes numbers reproducible; scope ending at
    `phase4-D` (unanswered, ceiling 0.0347).
- **Docs to link in applications:**
  - `docs/adversarial.md` (Phase 5);
  - `docs/point-in-time.md`;
  - `docs/splits-eval-audit.md` and the corpus refusal records;
  - `docs/phase3-preregistration.md`, which is the clearest evidence of how
    the work was run.
- **Architecture diagram** with latency and cost annotations.
- **60-second demo GIF:** a safe site, a phishing site, an injection attempt
  failing.
- **Authorship.** Still open, at `README.md:3`, `README.md:491` (copyright)
  and `pyproject.toml:7`.

---

## Future work (not scheduled)

- **The Phase 4 recorded sweep.** Three full-population repeats on
  `openai/gpt-oss-120b` under frozen `p4-v1`, with `phase4-D`'s cache fix
  (repeat index in the key; no reuse of provisional seals), run if the
  Developer tier reopens (~$1.22 at current listed rates). Needs its own
  amendment and reports against the same sealed baseline.
- **Forward DNS and TLS capture in `collect.py`,** recorded when each URL is
  first seen, so a later corpus can use them without leakage.
- **Certificate history** via crt.sh's Postgres interface or an independent CT
  index, on a population collected with it from the start.
- **A better WHOIS creation-date parser for long-tail TLDs.** That is what
  failed age's gate, and fixing it now would be tuning after a failure; on a
  fresh population it is legitimate.
- **Hosted-platform coverage (`production-gaps.md` §8).** `webflow.io`
  phishing misses because the platform is absent from `HOSTED_PLATFORMS`:
  the frozen champion alerts 5/50 known webflow training rows and leaves 23
  below band; the `is_hosted_tenant=1` counterfactual alerts 48/50. The
  counterfactual is in-sample (0 webflow rows in calib/test) and the weights
  were fit with the flag reading 0, so the fix is a cited-source list
  extension plus a retrain on a population that contains those platforms —
  never a serving-only flag flip.
- **Live enrichment** behind the existing provider interface, plus everything
  in `docs/production-gaps.md`.

---

## Order and time

| Order | Phase | Estimate |
|---|---|---|
| ✅ | Phases 0–3 | done |
| ✅/⏸ | Phase 4 — LLM layer, closed provisional (phase4-D) | sealed; recorded sweep future work |
| 1 | Forward workflow onto master + first run | today |
| 2 | Phase 5 — adversarial | 1 week (lexical evasion can overlap close-out) |
| 3 | Phase 6 — minimal serving | 3–4 days |
| 4 | Phase 7 — packaging | 2–3 days |

Work on a branch from here; Phase 3 landed directly on master because
amendments had to be committed before the runs they governed.

---

## What the project is actually demonstrating

The original framing was "add AI to a phishing detector." The work has
produced something better: a measurement pipeline with recorded refusals,
negative results that held up, and numbers stated only at the level the
evidence supports.

The findings worth leading with:
- the corpus confound, quantified rather than tuned away;
- ranking that holds across a temporal cut while calibration doesn't;
- soft voting capturing nearly everything a GBM does;
- a phishing feed filtered by takedown before it was ever collected;
- a popularity feature that leaks the label through how the benign sample was
  drawn;
- a shape gate that had to be stratified because phishing is two populations;
- an enrichment signal that failed its own contamination gate and is reported
  as ineligible rather than quietly kept;
- two budgets reported as unmet or indistinguishable, never rounded toward.

Most portfolio projects that add WHOIS features have a takedown leak they
don't know about. This one names three such leaks with measurements, blocked
its own strongest new feature when the gate said so, and publishes the
cold-start number next to the warm one.
