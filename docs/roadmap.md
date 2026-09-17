# PhishNet roadmap — revised (2026-09-17)

The original plan, rewritten against what actually happened. Phases 0–2 are
records. Phase 3 is mostly built: its protocol is settled and its population
has passed the gate, but enrichment and the ablation remain.

This revision also changes the framing. PhishNet is a **project, not a
product**. The deliverable is a set of findings that can be defended in
detail, plus a working demo. It is not a service that runs indefinitely.
Phases 4–7 are cut to match: infrastructure that exists only to keep a
service alive moves to future work.

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

**Threshold transfer is unresolved.** The honest number is a threshold fixed
in advance and then applied to the untouched test set:

| | recall | FPR |
|---|---|---|
| swept on test (unattainable) | 58.01% | 0.49% |
| full-train, tuned on memorized slice (unattainable) | 80.58% | 4.61% |
| **champion, threshold fixed in advance** | **58.93%** | **0.60%** |

The 0.5% budget is recorded as unmet. SHAP values ship through LightGBM's
native `pred_contrib`, with no `shap` dependency. The serving path returns 501
for explanations until the model migration lands.

---

## Phase 3 — Signals the URL string can't give you ✅ done (record)

Full protocol: `docs/phase3-preregistration.md` (Amendments A–E). Report:
`reports/phase3.md` (ablation first, then refusals, amendments,
deviations). The plan changed a lot as it met the data; every change is
recorded as an amendment made before the numbers it affects existed.

### What was found

- **The phishing feed is filtered by takedown before collection.** PhishTank
  `online-valid` contains only phish still live on the snapshot day.
  - **Train era:** 91.6% https, 20.5% on free hosting platforms.
  - **Test era:** 78.9% https, 13.3% on free hosting platforms.

  Every phishing row now carries a survival-lag stratum: fresh, short, long,
  or unknown (OpenPhish rows have no submission time).
- **Tranco rank is a label leak by construction.** Every benign row was
  sampled from Tranco. Rank appears only as a labeled diagnostic row, never in
  the model.
- **Phishing is a mixture of two populations.**
  - **Hosted tenants:** 21% of phishing (`*.vercel.app`, `*.weebly.com`, and
    similar), 80.1% of them root URLs.
  - **Everything else:** 25.0% root URLs.

  The unstratified shape gate asked the benign side to reproduce that mixture.
  The gate is now stratified on `is_hosted_tenant`, which is itself a model
  feature: the main stratum blocks promotion, the hosted stratum is
  descriptive.
- **Two corpus refusals are on record.**
  - **The 40k enlargement** failed the unstratified gate (root drift 0.238,
    depth AUC 0.364).
  - **The existing 12k corpus** passes the unstratified gate but fails the
    stratified one on six metrics.

  The stratified gate was adopted even though it ruled out the cheapest
  fallback.
- **Roots were not scarce.** A join-based Athena probe found about 527k
  selectable roots against a bar of 5,900, for $0.27. The fetch was then
  bounded:
  - seeded domain samples, with fixed stratum weights;
  - at most 4 rows per domain, so the domain bootstrap has enough clusters;
  - quartile length bands.

  The resulting corpus **passed the stratified gate** with no main-stratum
  failures.
- **Hosted tenants are grouped by tenant, not platform.** Under platform
  grouping, the test set held zero hosted phishing. A benign hosted stratum
  was added. Hosted-benign FPR is descriptive: it covers 18 platforms, and it
  makes no claim about hosted root pages.

### What changed from the original plan

- **Offline ablation only.** Live lookups, caches and timeouts are future
  work.
- **Certificate history is dropped (Amendment E).** crt.sh limits requests
  to about 5 per minute per IP, which made the first enrichment run fail 97%
  of CT lookups. No validated alternative source exists, and the feature
  could not be served at request time anyway. Domain age carries most of the
  expected signal.
- **DNS is forward-collection only.** Resolving today against old
  `first_seen` dates is a takedown leak.
- **FPR verdicts are three-valued** (met / unmet / indistinguishable), using
  the wider of the Wilson and domain-bootstrap intervals, at thresholds fixed
  in advance at 0.5% and 1%. With about 3 rows per domain, the 0.5% verdict is
  expected to read "indistinguishable," and that is recorded in advance.

### What it earned

Amendment E committed, protocol frozen. RDAP age-only pass sealed and
pinned (41,739 keys). Age gate: train pass (gap 0.016), test fail
(0.059, benign-heavy) — headline is the lexical row; conditional
age-known secondary published with its limits stated. Ablation:
52.4% @ 0.50% indistinguishable; age lift +0.20–0.31 paired where
known; Tranco diagnostic negative; transfer fixed at 0.5%, not at 1%;
cold-start 78→53% recall as age goes missing; stub p50 14.3 ms
(criterion 12 unmet). Write-up: this report, `docs/point-in-time.md`,
`docs/enrichment-coverage.md`, `docs/model-card.md`, README headline.

**The claim this phase earned:** domain age measured without leaking
future information, on a population that passed its own stratified
gate — ineligible for the headline on a benign-heavy lookup gap, with
the cold-start number published next to the warm one and a clear
statement of what doesn't hold (CT unmeasured, latency over budget).

---

## Phase 4 — The LLM layer, evaluated offline (1–1.5 weeks)

The LLM layer is the project's AI angle. It is evaluated on a fixed, pinned
sample of pages, not run as a live service.

- **The first tier is the uncalibrated Phase 3 champion.** Calibration doesn't
  survive the temporal cut. So size the escalation band in the score space
  that ships, with band edges fixed on the calib band and achieved rates
  reported on test. Never sweep them.
- **Budget against the cold-start escalation rate.** First-visit URLs arrive
  without age, so the first tier is less certain about them and more of them
  land in the band. The cost-per-1,000 figure must use that rate, not the warm
  one.
- **Freeze the page snapshots.** Fetch pages with Playwright once, store them
  with hashes, and run every evaluation against the snapshots. Pages change
  and phishing pages disappear, so live fetching isn't reproducible.
- **Constrain the output.** The LLM returns JSON through a tool-use schema,
  never free text parsed with regex. Its verdict becomes an input feature
  alongside the others; it never overrules them.
- **Report cost and latency per 1,000 URLs.**
- **Local-model comparison via Ollama.** It shows the deployment trade-off
  cheaply.

---

## Phase 5 — Adversarial hardening (1 week) ← the differentiator

Prompt-injection testing needs Phase 4's LLM layer, so this phase comes after
it. The lexical-evasion tests don't, and can start earlier if time allows.

- **Injection.**
  - Build 50–100 adversarial pages across injection vectors.
  - Measure attack success on the unhardened pipeline first.
  - Harden: strip comments and invisible text, mark untrusted content with
    explicit delimiters, add a lightweight detection pass, and never let LLM
    output alone decide the verdict.
  - Measure again.
- **Warm versus cold.** The first tier's structural features resist
  persuasive page text, but on a first visit age is unknown. That is exactly
  when a fresh phishing domain is visited. Measure injection success
  separately for rows with and without age. If it's materially higher without
  age, that's a finding for the write-up.
- **Lexical evasion.** Test homoglyph and IDN tricks, URL shorteners, open
  redirects and punycode. Report accuracy under attack separately from clean
  accuracy.

The before-and-after table remains the most interesting artifact in the
project. If time runs short, cut from Phase 6, never from here.

---

## Phase 6 — Minimal serving for the demo (3–4 days)

Only what the demo and a reviewer running the repo need.

**Keep:**
- **Champion servable in Docker.** Make `predictors` importable in the image,
  with a verified download for the GBM assets. This also fixes the `explain`
  501 on the serving path.
- **Enrichment through the stub provider** (unknown on every request), which
  matches the published cold-start number.
- **`asset_fingerprint` in prediction responses,** so any result can be traced
  to its model.
- **The `lifespan` context manager** instead of `@app.on_event`, with model
  loading behind a health check.
- **An extension good enough to record the demo.**
  - Warn on positives only; show a badge otherwise.
  - A local cache by domain, and debounced navigation.
  - The popup shows top attributions in native units ("registered 3 days
    ago").

**Cut, and move to "What production would need":**
- a live RDAP provider with a Redis cache and timeouts;
- Prometheus metrics and dashboards;
- a feedback store with a poisoning policy;
- a scheduled calibration refresh;
- a live certificate check.

A short `docs/production-gaps.md` describes each, with the reason it matters.
Knowing what's missing is worth more than half-building it.

---

## Phase 7 — Package it so it reads correctly (2–3 days)

- **README.** Lead with the numbers from thresholds fixed in advance, with any
  swept numbers labeled unattainable beside them. Show the cold-start number
  next to the headline.
- **Model card**, carrying the accumulated failure modes:
  - the inverted depth prior;
  - the scheme decision;
  - the point-in-time classification;
  - survivorship in the phishing feed;
  - the Tranco selection leak;
  - hosted coverage limits;
  - cold-start degradation;
  - calibration's shelf life;
  - the threshold-transfer verdict;
  - why certificate history was dropped.
- **Docs to link in applications:**
  - `docs/adversarial.md` (Phase 5);
  - `docs/point-in-time.md`;
  - `docs/splits-eval-audit.md` and the CC corpus refusal record;
  - `docs/phase3-preregistration.md`.

  Most of these record a decision *not* to do something convenient, which is
  rare in portfolio repositories.
- **Architecture diagram** with latency and cost annotations.
- **60-second demo GIF:** a safe site, a phishing site, and an injection
  attempt failing.
- **Authorship.** Still open, at `README.md:3`, `README.md:491` (copyright)
  and `pyproject.toml:7`. It takes five minutes to fix, and gets more awkward
  with every phase.

---

## Future work (not scheduled)

- **Forward DNS and TLS capture in `collect.py`.** Record them when each URL
  is first seen, so a later corpus can use them without leakage.
- **Certificate history** via crt.sh's Postgres interface or an independent CT
  index, on a population collected with it from the start.
- **Live enrichment:** an RDAP provider with a cache, timeouts and background
  refresh, behind the existing provider interface.
- **Monitoring, a feedback loop and a calibration refresh** — everything in
  `docs/production-gaps.md`.

---

## Order and time

| Order | Phase | Estimate |
|---|---|---|
| 1 | Finish Phase 3 | a few days |
| 2 | Phase 4 — LLM layer, offline | 1–1.5 weeks |
| 3 | Phase 5 — adversarial | 1 week |
| 4 | Phase 6 — minimal serving | 3–4 days |
| 5 | Phase 7 — packaging | 2–3 days |

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
- a product budget reported as unmet;
- a phishing feed filtered by takedown before collection;
- a popularity feature that leaks the label through how the benign sample was
  drawn;
- a shape gate that had to be stratified because phishing is two populations.

Most portfolio projects that add WHOIS features have a takedown leak they
don't know about. This one names three such leaks, with measurements, and
shows its cold-start number next to its warm one.
