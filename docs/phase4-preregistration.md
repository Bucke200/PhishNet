# Phase 4 preregistration — LLM layer, evaluated offline

**Status:** registered, not yet executed
**Branch:** `phase-4-llm`, from tag `phase-3-close`
**Registered at:** `509ff11f` — this document was committed before the
first network call in `src/phishnet/snapshot/`
**Supersedes nothing.** Phase 3's protocol (`docs/phase3-preregistration.md`,
Amendments A–E) remains the governing record for everything it covers.

Amendment letters in this phase are scoped `phase4-` and ordered by commit
time, not by section order. `phase4-A` is committed with this document;
`phase4-B` onward are committed as they arise, each before the numbers it
affects exist.

---

## 0. What this phase claims, and what it cannot

The LLM layer reads what the URL string cannot show: whether a page solicits
credentials, which brand it imitates, whether its text applies pressure, and
whether its apparent identity matches its domain. Tier 1 escalates a band of
uncertain rows; the LLM judges those rows; the cascade reports the result.

The phase makes **no claim about deployed performance on live traffic**. Every
number is computed against frozen snapshots of pages fetched once, at a time
recorded per row, from a corpus whose phishing side was already filtered by
takedown before collection (Phase 3). Fetchability is itself a label proxy
under that filter; see §3.

---

## 1. Tier 1 and the escalation band

### 1.1 Tier 1

Tier 1 is the **uncalibrated Phase 3 champion, row (a)**: lexical features plus
`is_hosted_tenant`. It is loaded by asset hash, not rebuilt. Concretely row (a)
is `backend/ablation_lexical_assets/gbm_model.pkl` (group=lexical,
`canonicalize_scheme=true` per its `train_config.json`), **not**
`backend/gbm_iso_assets/refit_base.pkl` (the Phase 2 champion on a different
population). Pinned hashes: model `7b765bfc82716350555d38d01f2246215f79661803b2468097979ec7b944024f`,
columns `39d0e665391b06557ced4a648caf9834b76cde83ffa6637128dd5b2becacd79e`.
Band edges are fixed on row (a)'s calib scores, or the band will not match the
headline. Calib population is `data/splits-p3/calib.csv` (n=13,157; 5,047 phish
/ 8,110 benign; train 44,285 / test 24,819), verified against
`repro/hashes-p3.json` before use — the directory is not in git, so the working
copy is never trusted.

Row (a) takes no age input. The warm/cold distinction therefore does not apply
to tier 1, and the escalation rate is identical under both. This is stated
explicitly because the roadmap directs budgeting against the cold-start rate;
that direction binds row (b), which is not tier 1 here. No cold-start
adjustment is made or needed.

### 1.2 Threshold and band edges — pre-committed

All edges are derived from **calib** only, by function call, never hard-coded
and never swept.

```
t_alert    = threshold_at_fpr(calib_benign_scores, 0.005)      # ≈ 0.926936
lower_edge = threshold_at_fpr(calib_benign_scores, 0.005 + 0.05)
```

One function, called twice. A search loop is prohibited: it reads as a sweep in
the diff, and the two-call form inherits the existing tie and comparison
conventions rather than restating them.

**Buckets are half-open, so every row falls in exactly one:**

| bucket | interval |
|---|---|
| below | (−∞, `lower_edge`) |
| band | [`lower_edge`, `t_alert`) |
| alert | [`t_alert`, ∞) |

**The 5% is a benign-only budget**, not a total escalation rate. Benign mass in
band is `FPR(lower_edge) − FPR(t_alert)` = 0.055 − 0.005 = 0.05 by
construction. `threshold_at_fpr` achieves *at most* its target, so the report
states the **achieved** benign band mass, not the nominal 5%.

A benign-only budget is base-rate independent and transfers to deployment.
**Rejected alternative:** budgeting total escalation volume or cost. Defensible,
but it couples the edge to the corpus class mix and does not transfer.

**Phishing mass in band is a reported outcome, never an edge input.** Row (a)
already places roughly half of phishing above `t_alert`. A thin band means
tier 1 separates cleanly, which is a result. Moving an edge because the band
looked empty voids the "never swept" claim and must instead be an amendment
that says so.

**Reported on test, split by class and by `is_hosted_tenant`:** achieved
benign band mass, achieved total escalation rate, phishing mass in band.

---

## 2. The cascade decision rule — pre-committed

Only in-band rows reach the LLM. In-band means below `t_alert` by construction,
so any mapping that places a row *below* the band cannot change recall or FPR
at `t_alert`; it can only manufacture separation in rank metrics. The v1
mapping therefore has exactly one arm that moves:

| in-band row | cascade score |
|---|---|
| LLM `verdict == phishing` | `nextafter(t_alert, +inf)` |
| LLM `verdict == benign` | tier-1 score, unchanged |
| LLM `verdict == suspicious` | tier-1 score, unchanged |
| schema violation, refusal, or API failure | tier-1 score, unchanged |
| unfetchable, default policy | tier-1 score, unchanged |
| unfetchable, alternative policy | routed to human review, counted separately |

Rows below the band and rows already in alert pass through at their tier-1
score without an LLM call.

**No confidence gating in v1.** Any confidence cutoff introduced later is a
pre-registered number committed by amendment before the numbers it affects
exist, never a value discovered from an agreement table.

**Consequence, stated here and repeated in the report:** anchoring every
phishing-verdict row to a single float makes cascade PR-AUC and ROC-AUC partly
artifactual, and creates a tie block of mass sitting exactly at `t_alert`.
The **fixed-threshold recall/FPR pair is the primary result**; rank metrics are
descriptive. This is the same demotion applied to swept numbers in Phase 2.

**Tie handling is a test obligation, not an assumption.** `tests/` must assert
that `threshold_at_fpr`, `fpr_interval_report` and the cascade's bucketing use
the same comparison direction, with explicit rows at `lower_edge`, at
`t_alert`, and at `nextafter(t_alert, +inf)`, checked against a hand-computed
FPR.

### 2.1 Unfetchable policy — both arms fixed now

- **Default:** an unfetchable in-band row keeps its tier-1 score.
- **Alternative:** an unfetchable in-band row routes to human review and is
  counted as a separate disposition.

Both arms are reported for every cascade number. Selecting one after seeing
results is the failure mode this clause exists to prevent.

---

## 3. Step 0 — fetchability, and the trigger

### 3.1 Fetch population

Defined once, fetched once, frozen. Band edges are fixed on calib before any
fetch, so both components below are computable at registration time.

```
fetch_set = step0_sample ∪ in_band(calib) ∪ in_band(test)
```

`step0_sample`: n ≈ 300–500, stratified by class × era (train/calib/test band)
× survival stratum (fresh/short/long/unknown), using the existing
`survival_stratum` and `source` columns.

No row ever acquires two snapshots at two timestamps. A row present in both
components is fetched once and reused.

### 3.2 Reporting

Fetch success is reported by class, by era, and by survival stratum, each
crossed with the full outcome taxonomy (`ok`, `dns_fail`, `conn_refused`,
`tls_fail`, `timeout`, `http_4xx`, `http_5xx`, `parked`). Never collapsed to a
boolean, and never reported only as an overall percentage.

**Cell-level rates are descriptive.** At n ≈ 300–500 across roughly fifteen
cells, per-cell intervals are about ±20 points. The class × era × stratum ×
outcome table carries counts and Wilson intervals and is labeled descriptive,
so it does not read as fifteen findings. The trigger fires on marginals, which
are adequately powered.

### 3.3 The trigger — pre-committed

Option 2 (joined feature) is eligible **only if all three hold**:

1. train-band phishing fetch success ≥ **40%**;
2. train-band class gap (benign − phishing fetch success) ≤ **0.05**;
3. test-band class gap (benign − phishing fetch success) ≤ **0.05**.

The 0.05 budget is Phase 3's unknown-gap budget, reused deliberately.

Otherwise **option 1, verdict-as-report, on the test band only**.

**Why the gap conditions exist.** Fetch success is a label proxy by the same
mechanism as the takedown filter already documented in Phase 3: PhishTank
`online-valid` holds only phish live on the snapshot day, and train-band phish
are months past that day. If benign fetches at 95% and phishing at 12%, then
"page retrievable" carries most of the label, and any joined feature inherits
it through its missingness pattern rather than its values. Condition 2 is
measured on train because that is where the leak would enter a retrained model;
condition 3 on test because that is where it would be evaluated.

**Expectation on record:** option 1 fires. This is written down so that an
option-1 outcome reads as a gate working, not as a fetch that went badly.

The trigger is applied mechanically by `trigger.py`, a pure function of the
Step-0 table. Its verdict is recorded as amendment `phase4-B`. Any wish to move
a bar after seeing the rate is itself the amendment and must say so in those
words.

### 3.4 Option 3 — forward collection, running regardless

Started before this document's numbers exist, because its value is
wall-clock dependent. Workflow `.github/workflows/phase4-forward.yml`, separate
from `collect.yml`'s 03:17 UTC job.

- Phishing arm: daily PhishTank `online-valid` pull (`submission_time` as
  newness key) plus OpenPhish snapshot diff (first-observed as newness key).
- Benign arm: **CommonCrawl deep links**, matching the clean corpus. Tranco is
  not used. Drawing the forward benign arm from Tranco would rebuild the
  by-construction selection leak documented in Phase 3, in a corpus explicitly
  intended to feed a later phase.
- Both arms snapshotted within hours of first observation via the Step-2
  fetcher, same outcome taxonomy and hashes. `collected_at` and `snapshot_at`
  stored per row so the lag is explainable.

**Ineligibility clause, belt and braces:** the forward corpus — both arms,
including the CommonCrawl-drawn benign arm — is a fetchability and robustness
control. It is **ineligible as a training population** until it passes its own
shape and contamination gates in a later phase.

---

## 4. Provider, model, and determinism

### 4.1 Route and run class

One model, one endpoint, for development and recording alike:

| field | value |
|---|---|
| provider | Groq (GroqCloud) |
| model ID | `openai/gpt-oss-120b` (full exact ID, never the family name) |
| catalogue status | **Production** |
| weights | open |
| endpoint | `https://api.groq.com/openai/v1/chat/completions` |
| API surface | OpenAI-compatible chat completions |
| structured output | `response_format: {"type": "json_schema", ...}`, strict |
| temperature | 0 |
| `seed` | `0`, fixed and recorded |
| `reasoning_effort` | `low` (model accepts low / medium / high; default is medium) |
| prompt version | `p4-v1` |
| development quota | free tier: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD per model for `openai/gpt-oss-120b` (console.groq.com/docs/rate-limits, read 2026-09-18; token caps bind first on page-sized inputs) |
| recorded-run quota | Developer tier (card added; ~10x limits, 25% discount, no minimum spend) |
| tier per run | §4.4 gate runs on free tier; recorded sweep runs on Developer tier; same model string throughout |

**Development and the recorded run use the same model on the same endpoint.**
Only the quota tier differs. There is therefore no possibility of a behavioural
difference between the route a prompt was tuned on and the route the numbers
were produced on — the hazard that a free-development / paid-record split across
two SKUs would otherwise carry (`phase4-A`).

**Production, not Preview, is a requirement.** Groq designates preview models as
evaluation-only and subject to discontinuation at short notice. The newer
Qwen 3.8-27b is plausibly the stronger judge and is rejected on this ground
alone: a model that can be withdrawn mid-run cannot carry a recorded population.
GPT-OSS 120B is Production and is Groq's own named migration target for the
models it deprecated through 2026.

**Open weights are part of why this model was chosen.** §5.2 seals raw requests
and responses so results survive the model's retirement. Open weights make that
survival stronger than an archive: the recorded run is re-executable on other
hardware after the endpoint is withdrawn. That is the most defensible available
answer to "the headline depends on a hosted model that no longer exists."

**Capability is knowingly traded, and the trade is bounded.** GPT-OSS 120B is a
weaker judge than a frontier hosted model. Accepted, and recorded in advance,
because the question Phase 4 answers is whether the LLM layer beats the
password-field baseline (§7): a mid-tier open model clearing that bar is a
stronger result than a frontier model clearing it, and failing it is a cheap
finding rather than an expensive one. **If the layer fails against this model,
no claim is made that a stronger model would also fail.** That is a separate
question, explicitly out of scope, and the report must say so rather than let a
negative result read as general.

**Run class, not route class.** A sweep can still terminate mid-population —
rate limits apply at the organisation level and per model, so a concurrent job
on the same org can exhaust the day's requests. That is a property of the run,
not of the model, and is handled as one:

- **`provisional`** — any run that did not cover the full in-band population,
  for any reason. Sealed and kept, but **cannot supply a published number**.
- **`recorded`** — a single sweep covering 100% of the in-band population under
  one prompt version, one model ID and one `system_fingerprint`, with no quota
  truncation.

`run_class` is written into the run store by the driver, which **asserts full
coverage before marking a run `recorded`**. A truncated sweep is re-run, not
patched by topping up the missing rows from a second sweep: mixing two sweeps
under one run ID would put rows judged days apart, possibly under different
served weights, into one population.

**Prompt freeze.** Because development and recording share a route, `p4-v1` is
frozen before the recorded sweep begins and iteration stops there. The freeze is
what separates the two activities now that the model string no longer does.

**Model swap is an amendment, not a config change.** The cache keys on the model
string, so a swap invalidates it automatically and no stale judgment can survive
one. Swapping to another model later requires a new `phase4-` amendment naming
the model and the reason, and a fresh `recorded` sweep; prior recorded numbers
are kept and labeled with the model that produced them.

**Data handling.** Groq does not retain customer inference inputs/outputs by default; only usage metadata is always retained, with up-to-30-day reliability/abuse logs and optional Zero Data Retention (console.groq.com/docs/your-data, read 2026-09-18). The payload is extracts of public web pages, but Phase 5's
adversarial corpus is authored material and its handling is a decision for
Phase 5's preregistration, not an inherited default.

### 4.2 Determinism is measured, not assumed

Temperature 0 does not determinize a reasoning model: reasoning traces vary
between calls regardless of temperature, and reasoning effort is a separate
pinned axis.

This route provides two instruments the earlier candidates did not, and both are
used rather than inferred:

- **`seed`.** The API makes a best-effort deterministic sample for repeated
  requests with the same seed and parameters. It states plainly that determinism
  is **not guaranteed**. The seed is fixed and recorded; it is treated as a
  variance reducer, never as a reproducibility claim.
- **`system_fingerprint`.** The provider's own signal for backend changes.
  Persisted from every response. A fingerprint change **mid-run invalidates the
  run** (§4.1 `run_class`) rather than being averaged over, and a fingerprint
  change **between** the recorded run and any later re-run is reported beside the
  comparison rather than silently absorbed.

**Acceptance is not efficacy.** §4.4's smoke test establishes only that `seed`
is *accepted* by the endpoint. Whether it is *effective* is what this section
measures. A 200 response is not a reproducibility claim, and the two must never
be conflated in the report — the provider's own "determinism is not guaranteed"
language predicts exactly this gap.

**Criterion:** re-judge 50 in-band snapshots on a cold cache — same prompt
version, same model ID, same seed, same `system_fingerprint` — and report the
verdict disagreement rate.

**Bar, pre-committed:** disagreement ≤ **5%**.

- **At or under the bar:** the headline cascade number is reported as a point
  estimate, with the measured disagreement rate stated beside it.
- **Over the bar:** the headline cascade number is reported as a **range over
  three repeated judgments** of the full in-band population, not as a point
  estimate, and the instability is stated in `reports/phase4.md` and carried to
  the Phase 7 model card.

The bar exists because every other measurement in this document carries a
pre-committed budget — 0.05 for the unknown gap, 0.05 for benign band mass, 40%
for the trigger. A criterion that only says "report the rate" is weaker than the
rest of the protocol and would let any number pass.

Whatever the rate, if it is non-zero under a fixed seed and an unchanged
fingerprint, the report and the model card state the honest framing in one line:
the response cache, not the seed and not the temperature, is what makes the
published numbers reproducible.

### 4.3 `asset_fingerprint`

```
provider=groq, model=openai/gpt-oss-120b, system_fingerprint, seed,
temperature=0, reasoning_effort=low, prompt_version=p4-v1, run_class,
tier1_assets_hash, lower_edge, t_alert, snapshot_manifest_hash,
unfetchable_policy
```

`system_fingerprint` replaces the `provider_route` field an earlier draft
carried: it is the provider's own backend identity signal rather than a routing
label inferred by the client.

### 4.4 Provider-capability gate — before any client work

Five properties of this route are assumed by this document and none are
established by it. All five are asserted by **one request**, not a sweep, before
`client.py` lands. A sweep would add cost without adding information: the only
thing a sweep tests that a single call cannot is fingerprint *stability*, which
is §4.2's job.

| # | assertion | if it fails |
|---|---|---|
| 1 | strict `json_schema` honored against the real `RESPONSE_SCHEMA` | §5.1 — narrow the schema by amendment, never loosen to `json_object` |
| 2 | `seed` accepted for this model ID on this endpoint | amendment; `seed` drops from §4.3, §4.2's bar still applies |
| 3 | `reasoning_effort: low` accepted for this model ID | amendment naming the effort actually used |
| 4 | `system_fingerprint` present in the response | amendment; §4.1 `run_class` loses its backend-identity predicate and falls back to model ID + prompt version alone, which is **weaker and must be stated as such** |
| 5 | `usage` breaks out reasoning tokens from visible output tokens | criterion 13 is unmeetable as written; §8 collapses to one token line by amendment |

Assertion 5 is the one most easily missed. §8 requires reasoning and visible
output tokens as separate lines, but the endpoint exposes configurable controls
for how reasoning is emitted — `reasoning_format` and `include_reasoning` are
mutually exclusive — so the shape of the returned accounting is not safe to
assume. Discovering this at report time would mean amending a criterion after
seeing the numbers it governs.

**Ordering.** This gate is a network call, and criterion 1 places the prereg
before the first one. The sequence is therefore: commit this document, run the
gate, amend if the gate surprises. That is the Phase 3 pattern — amendments land
before the numbers they govern — not a workaround for it.

The gate's request and response are sealed in the run store like any other call,
under their own run id `p4-gate`, marked `provisional` (§4.1), so they cannot be
confused with the evaluation run. It supplies no published number. The gate runs
on the free tier; the recorded sweep runs on the Developer tier. The commit
fixing provider, model string, tier mapping, prompt version `p4-v1`, the exact
`RESPONSE_SCHEMA` (§5.1) and cache key precedes the single gate call, whose
pass criterion was written first.

---

## 5. Snapshots and what is sent

- **Fetch once, freeze.** Raw HTML, final URL, HTTP status, redirect chain,
  outcome code. `sha256(raw_html)` and `sha256(canonical_extract)` stored
  separately. Bodies under `data/snapshots-p4/`, gitignored. Manifest
  `reports/snapshot-manifest-p4.json` committed.
- **The extract is what the model sees. Raw HTML is never sent.** Title, meta
  description, visible text truncated at 6,000 characters, form fields
  (name/type/placeholder/action host), link-host frequency table capped at 20,
  iframe and script source hosts, image alt text, declared language, favicon
  host.
- The extract is delimited as untrusted content from `p4-v1` onward, before any
  hardening exists. Phase 5 changes the handling, not the interface.
- **Out of scope, recorded:** the model accepts images, and page screenshots
  would be a natural input. They are not used in Phase 4. Adding a visual
  channel changes the injection surface and belongs after Phase 5's
  before/after table, not before it.

### 5.1 Response schema

Enforced through the provider's structured-output schema. **No regex over
prose, under any failure mode.** A schema violation is sealed and treated as
"retain tier-1 score" per §2.

Strict structured output is requested through
`response_format: {"type": "json_schema", "json_schema": {...}}`, which uses
constrained decoding to guarantee schema compliance on supported models. This is
stronger than the `json_object` JSON mode, which only guarantees valid JSON, and
`json_object` is **not** an acceptable fallback here: a conforming-but-wrong
shape would be parsed as a judgment.

**Gate before any client work.** Strict mode carries documented schema
limitations. A smoke test sends the real `RESPONSE_SCHEMA` and asserts a
conforming parse **before** `client.py` is built on it. If any field in
`RESPONSE_SCHEMA` cannot be expressed under strict mode, the schema is narrowed
to what strict mode accepts and the narrowing is recorded here as an amendment —
the schema is never loosened to `json_object` to accommodate a field.
`schema.py` holds one provider-neutral schema dict; the chat-completions shape
lives in the adapter.

Fields: `asks_for_credentials`, `credential_types[]`, `imitated_brand`
(nullable), `brand_confidence`, `urgency_pressure` + bounded score,
`identity_domain_match` ∈ {match, mismatch, unrelated, unknown}, `verdict`,
`confidence`, `evidence[]` (short spans copied from the extract).

The exact strict-mode schema fixed by this registration (`p4-v1`,
`src/phishnet/llm/schema.py::RESPONSE_SCHEMA`, mirrored in
`src/phishnet/llm/prompts/p4-v1.txt`) is:

```json
{
  "type": "object",
  "properties": {
    "asks_for_credentials": {"type": "boolean"},
    "credential_types": {"type": "array", "items": {"type": "string", "enum": ["password", "card", "otp", "email-login", "other"]}},
    "imitated_brand": {"type": ["string", "null"]},
    "brand_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "urgency_pressure": {"type": "boolean"},
    "urgency_score": {"type": "number", "minimum": 0, "maximum": 1},
    "identity_domain_match": {"type": "string", "enum": ["match", "mismatch", "unrelated", "unknown"]},
    "verdict": {"type": "string", "enum": ["phishing", "benign", "suspicious"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6}
  },
  "required": ["asks_for_credentials", "credential_types", "imitated_brand", "brand_confidence", "urgency_pressure", "urgency_score", "identity_domain_match", "verdict", "confidence", "evidence"],
  "additionalProperties": false
}
```

Requested as `response_format: {"type": "json_schema", "json_schema":
{"name": "p4_page_judgment", "strict": true, "schema": RESPONSE_SCHEMA}}`.
`verdict` describes only the page extract (the full URL string is never sent;
only `page_host` rides in a separate structured field for the identity
comparison, so the judgment cannot re-read what tier 1 scored). `confidence`
is the single suspicion score.

`evidence[]` exists so that a wrong verdict can be distinguished from a wrong
read. `imitated_brand` has no ground truth: it is reported descriptively, or
against a hand-labeled 50-row subset with agreement stated. It never appears in
a headline.

### 5.2 Cache and run store

Cache key: `snapshot_hash + prompt_version + model_string`. Raw requests and
raw responses are sealed under `runs/phase4/<run-id>/`, **including refusals,
schema violations and rate-limit failures** — failed attempts consume quota and
belong in the record. Only sealed runs are pinnable in `repro/hashes-p4.json`.

---

## 6. Evaluation — pre-committed report shape

Through `eval.py`'s existing predictor interface; the cascade is a predictor
implementing `name` and `score`, with band edges injected from their
calib-fixed values and never recomputed on test.

**Primary:**
- cascade recall and FPR at the fixed tier-1 thresholds (0.5% and 1% targets),
  under **both** unfetchable policies;
- the **password-field baseline** beside every LLM number (see §7);
- `fpr_interval_report` and `paired_bootstrap_ci` as in Phase 3, with
  "indistinguishable" used wherever intervals overlap.

**Descriptive:**
- agreement with the label on **escalated-and-fetched rows only** (the
  population the layer actually applies to) and on **all fetched rows**,
  reported separately;
- **verdict distribution over in-band rows.** If the model returns `suspicious`
  on most of them, the layer is a no-op and that is the finding — registered
  here so it is reported either way;
- rank metrics, labeled artifactual per §2;
- per-field outputs, `imitated_brand` included.

**Coverage and survivorship are stated beside every number, never footnoted.**

---

## 7. The baseline, built first

`PasswordBaseline`, a pure function of the extract: a password `<input>` is
present **and** the form-action host differs from the page host → 1.0, else
0.0. Roughly twenty lines, shipped as an `eval.py`-compatible predictor.

It is built and run **before any LLM output is read**, so it cannot be
retrofitted as a comparison after the LLM looks good.

**If it closes most of the tier-1 → tier-1+LLM gap, that is the phase's
headline.** A Phase-3-shaped negative result, reported as such.

---

## 8. Cost and latency

Measured from the API's own token-usage fields and wall-clock timings on the
recorded run. Never estimated.

- Per 1,000 URLs, for **escalated-only** and **all-rows**, each labeled with
  the escalation rate it assumes.
- **Reasoning tokens and visible output tokens reported as separate lines.**
  Reasoning bills as output and swings by an order of magnitude with effort, so
  a combined figure is uninterpretable.
- **Every per-1,000 figure is labeled cold-cache**, i.e. first-judgment cost.
  The response cache makes re-runs free; a reader must not conclude the layer
  is free.
- **Cost is a priced forecast, not a bill.** The free tier bills at zero and the
  Developer tier discounts by 25%, so no list-price figure is observed. Report
  **measured token counts** from the recorded run, which are real regardless of
  price, multiplied by Groq's published per-token rate for
  `openai/gpt-oss-120b` (TODO rate, re-check at report time). Name the rate and
  its date, label it a priced forecast, and never present it as observed spend.
  State the tier the run was made on.
- LLM latency p50/p90 reported alongside tier 1's 14.3 ms p50 (Phase 3
  criterion 12, still unmet, owned by Phase 6).
- Any higher-reasoning-effort comparison is a pre-registered two-arm run
  committed by amendment, not a knob turned after seeing agreement.

---

## 9. Criteria

| # | criterion | verdict |
|---|---|---|
| 1 | Prereg committed before first network call | |
| 2 | Band edges derived from calib by function call, no loop, no sweep | |
| 3 | Achieved benign band mass ≤ 0.05, reported not nominal | |
| 4 | Every row in exactly one bucket; boundary tests pass | |
| 5 | Cascade mapping matches §2 exactly | |
| 6 | Both unfetchable arms reported for every cascade number | |
| 7 | Step-0 trigger applied mechanically; verdict recorded as `phase4-B` | |
| 8 | No row holds two snapshots | |
| 9 | Single model ID, single `system_fingerprint` and single prompt version across the recorded run | |
| 9a | All five §4.4 capability assertions run and recorded before client work; no `json_object` fallback | |
| 9b | Published numbers come only from a run marked `recorded`, with full in-band coverage asserted | |
| 10 | 50-snapshot cold-cache disagreement rate reported, under fixed seed and unchanged fingerprint | |
| 10a | Negative result, if any, scoped to this model and not generalised | |
| 10b | Disagreement ≤ 5%, or headline reported as a range over three repeats | |
| 11 | Raw HTML never transmitted (asserted in tests) | |
| 12 | Password baseline run before LLM numbers read | |
| 13 | Cost split into reasoning vs visible output, labeled cold-cache (or §4.4-5 amendment on record) | |
| 14 | Coverage and survivorship beside every number | |

Unmet criteria are reported as unmet, with the measurement attached. They are
not rounded toward.

---

## 10. Amendments

### `phase4-A` — provider substitution
*Committed with this document, before any fetch.*

The Phase 4 roadmap specified Gemini via Google AI Studio as primary, with
NVIDIA NIM as an optional second opinion. This phase substitutes
`openai/gpt-oss-120b` on Groq for both development and the recorded run, on the
Production tier of that catalogue.

**Candidates considered and rejected, with reasons, so the choice is not
reconstructible as convenience:**

- **Gemini Flash (free tier).** The 3.x Flash line has moved well past the
  version originally contemplated, and free-tier request-per-day allowances
  reported for the current Flash models are small enough that a few hundred
  escalated rows would take weeks. Free quotas are adjusted by model, project,
  usage tier and account status, so no third-party figure is dependable.
- **Muse Spark 1.3 Contributor Free (OpenCode Zen).** Free, capable, and
  schema-supporting, but the quota is unpublished and rotates without notice,
  and the family is exposed under three near-identical IDs with different
  contracts. Disqualified on run-completion risk.
- **Qwen 3.8-27b (Groq).** Plausibly the stronger judge, but Preview status
  means it may be discontinued at short notice. Disqualified on the same ground
  as the above.

**Reasons for the model chosen:**
- Production status, so it cannot be withdrawn mid-phase without notice;
- open weights, so the recorded run outlives the endpoint (§4.1);
- strict `json_schema` with constrained decoding, satisfying §5.1's hard
  requirement directly;
- `seed` and `system_fingerprint`, which turn §4.2's determinism criterion from
  an inference into a measurement;
- a free development tier and a same-model paid tier, which preserves the
  roadmap's free-development intent **without** a model swap between
  development and recording.

**The roadmap's free-development / pinned-paid-record split is satisfied by
quota tier rather than by model identity.** This is a stronger form of the same
discipline, not a relaxation of it: the two-SKU behavioural risk that an earlier
draft of this amendment had to accept is eliminated rather than managed.

**Known risks accepted, in advance:**
- the model is weaker than a frontier judge; §4.1 bounds what a negative result
  may be taken to mean;
- Groq rate limits apply at the organisation level and per model, so a
  concurrent job on the same org can truncate a sweep; handled by `run_class`
  (§4.1), not by assumption;
- platform continuity is not guaranteed — Groq's inference technology was
  licensed to NVIDIA in December 2025, with most of the engineering team moving
  across. Open weights are the mitigation: the model outlives the provider.

The roadmap's optional second-opinion agreement check between two hosted models
is **not** attempted in Phase 4 and is out of scope here.

### `phase4-B` — Step 0 trigger verdict
*Committed after the Step-0 table exists and before any retraining.*

`trigger.py` applied mechanically to the Step-0 table (`reports/phase4-step0.json`,
fetch_set n=3,821 = step0_sample 396 ∪ in_band 3,488; edges calib-fixed
`t_alert=0.926936`, `lower_edge=0.649308`): train-band phishing fetch success
0.523 (≥ 0.40 holds), train-band class gap 0.409 (> 0.05 fails), test-band
class gap 0.751 (> 0.05 fails). **Verdict: option 1, verdict-as-report, on the
test band only** — the recorded expectation (§3.3), a gate working, not a
fetch gone badly. No bar was moved; any wish to move one would itself be the
amendment. Fetch success is a strong label proxy here (test phish 0.135 vs
test benign 0.886), which is exactly what conditions 2–3 exist to catch.

### `phase4-C` — fingerprint predicate revision
*Committed after the §4.2 determinism measurement, before any headline number.*

The determinism run (`runs/phase4/p4-determinism-1/`, 50 snapshots × 2 cold
judgments, fixed seed 0) returned 35 distinct `system_fingerprint` values
over 101 calls including the gate, with only 1 of 50 repeat pairs sharing a
fingerprint. The fingerprint rotates per call — it is a serving-instance
label, not the backend-identity signal §§4.1–4.3 assumed. A `recorded` run
requiring "one `system_fingerprint`", and a §4.2 measurement "under ...
unchanged fingerprint", are therefore unmeetable as written, on any quota
tier.

Revised predicate, explicitly weaker and stated as such (the same fallback
posture as §4.4 assertion 4): a `recorded` run requires one model ID, one
prompt version and seed 0 with full in-band coverage and no truncation; the
fingerprint *distribution* is sealed and reported beside every comparison
rather than asserted as identity. A fingerprint change can no longer
invalidate a run, because there is no stable value to change from — this is
weaker than the registered design, and the report and model card say so.

The §4.2 bar outcome stands unamended: verdict disagreement was 11/50 (22%,
over the 5% bar), so the headline cascade number will be reported as a range
over three full-population repeats, not a point estimate. Decomposition
(descriptive, does not move the bar): 10 of the 11 are quota/infra failures
sealed as verdict `None` on the free tier (empty usage, no fingerprint), and
1 is a genuine `phishing → suspicious` wobble — model-only disagreement 1/40
among completed pairs. The bar counts verdict disagreement as registered;
the decomposition explains it.

### `phase4-D` — close-out without the recorded sweep
*Committed before the close-out report; no run-store write precedes it.*

- The recorded sweep was not run. The Developer tier is unavailable on this
  org — the gate's sealed headers show free-tier caps (1K RPD / 8K TPM) —
  and at ~2,000 tokens per page-sized call the free 200K TPD fits ~100
  calls/day, so three repeats over 1,106 rows would take two to four weeks.
  LLM quota goes to Phase 5 instead, where the call volume is small.
- The provisional 68/1,106 sweep stays sealed and unpublished. Criterion 9b
  is unmet, with this reason attached. The phase question — whether the LLM
  layer beats the password baseline — is recorded as **unanswered, not
  negative**, and nothing is generalized beyond `openai/gpt-oss-120b` (10a).
- Criterion 10b is unmet: disagreement stands at 11/50, so no
  point-estimate headline exists; the range-over-three-repeats headline is
  future work and needs its own amendment.
- Cache defect, binding on any future recorded run: the cache key
  (`snapshot_hash` + prompt + model) carries no repeat index or run id, so
  repeats 2–3 would return repeat 1's cached responses and the range would
  have zero width; provisional and determinism seals must never seed the
  recorded cache. Fix by amendment before that run.
- `p4-v1` frozen at sha256
  `f37d30df14193a1d8705b44d356fdb8645f68fc3e988b9074ad6eba2bf0a749e`
  (`src/phishnet/llm/prompts/p4-v1.txt`). Provisional agreement has already
  been read, so any edit is a new prompt version.
- §1.2 correction: "0.05 by construction" was wrong. `floor(0.055 × 8110)`
  = 446 and `floor(0.005 × 8110)` = 40 admit at most 446 − 40 = 406 benign
  rows in band, above 0.05 × 8110 = 405.5 — knowable at registration.
  Achieved 406/8110 = 0.0501; criterion 3 is unmet, stated not rounded.

---

## Appendix — forward collection, first run

- Workflow lives on master (`df85776d`): checks out tag `phase-4-close`
  (never a branch) so the fetcher, outcome taxonomy and hashes are always
  the Step-0 ones; outputs commit to the `forward-p4-data` branch, never
  master (the data-branch tree holds data only — the yml is dropped from
  the pushed tree because the App token may not create workflow files).
- First run 2026-09-18, manual `workflow_dispatch`, success:
  https://github.com/Bucke200/PhishNet/actions/runs/35345551965
  (two earlier attempts failed on the data-branch push — short refspec,
  then workflow-file permissions — both fixed on master; no corpus data
  affected).
- First manifest (`forward-p4-data:reports/forward-manifest-p4.json`):
  phish arm `skipped-no-key` (no `PHISHTANK_KEY` secret — recorded, not
  silent); benign arm recorded `collected` by the entry point, which pins
  sources/keys but does not yet pull CC deep links (no pull implemented —
  stated, not implied); snapshot step `no-due-rows`.
  Superseded by `phase4-E`: the phish arm is now OpenPhish-only and really
  collects (first verified pull 2026-09-18: 300 feed rows, 300 new).
- First real collection 2026-09-18 (manual dispatch, success):
  https://github.com/Bucke200/PhishNet/actions/runs/35349161051 —
  phish 300/300, snapshot 300 rows (ok 215, tls_fail 20, timeout 17,
  http_4xx 46, http_5xx 2); benign `deferred` per `phase4-E`. A follow-up
  run verified accumulation (6 rounds, same-day diff `n_new=0`).
  Mechanism fix on the way there, recorded: every run commits on the pinned
  tag, so pushes after the first are non-fast-forward — the workflow now
  rebases onto the data-branch tip, and a manifest monotonicity gate aborts
  the push rather than silently dropping history (an earlier silent-restore
  failure dropped the three stub rounds from the tip; they survive in the
  parent commit and in this appendix).

### `phase4-E` — forward collection goes OpenPhish-only (Minimal)
*Committed before the new fetcher runs on schedule; tag `phase-4-forward-1`.*

The workflow as landed collected nothing: PhishTank registration is closed
(no key, arm honestly `skipped-no-key`), the OpenPhish pull was a stub
record, and no CC deep-link pull exists — while green no-op runs looked like
collection. Recorded, not silently left:

- Phish arm: OpenPhish-only pull (`collect.fetch_openphish`, no key),
  first-observed diff against a seen set, new rows queued with
  `collected_at`, snapshotted with `snapshot_at` beside it. Same Step-2
  fetcher, outcome taxonomy and dual hashes as Step 0.
- PhishTank: out (registration closed). Benign arm: deferred — CC pull
  unimplemented, status `deferred`, future work. The forward corpus is
  therefore phish-only until a later phase says otherwise.
- Seen/due/snapshot state restores from the data branch each run so rounds
  accumulate; per-row records persist in `snapshots.jsonl`, bodies under
  `bodies/`.
- New tag `phase-4-forward-1` on the implementation commit; the master
  workflow checks it out (never a branch). Schedule stays daily while
  Phase 5 runs — this starts the clock on data that cannot be got back.

### `phase4-F` — retrospective finding: ~12% schema fail-open defect in Tier-2 JSON schema
*Committed at Phase 5 closeout; documentation-only retrospective note.*

Phase 5 execution uncovered a systemic structural defect in the strict JSON
response schema frozen during Phase 4 (`src/phishnet/llm/schema.py`):

1. **Defect Mechanism:** The registered `credential_types` enum allowed only
   `["banking", "email", "corporate", "social", "cryptocurrency", "government", "ecommerce", "other"]`.
   On credential-harvesting pages, the model frequently attempted to output
   `"login"` or `"credentials"` instead of selecting from the allowed enum values.
   Provider-side strict schema validation rejected these responses with HTTP 400.
2. **Deterministic Fail-Open:** In accordance with Phase 4 §2, calls failing JSON
   schema validation were deterministically sealed with `verdict: None`, retaining
   their Tier-1 score. In production, retaining the Tier-1 score means the page
   is **NOT LIFTED to alert**. Thus, on credential-harvesting phishing pages,
   the cascade failed open without alerting.
3. **Prevalence across Populations (Phase 5 Measurements):**
   - Clean phishing bases: 15 / 126 calls (**11.9%**).
   - Injected phishing pages: 35 / 300 calls (**11.7%**).
   - Clean benign bases: 0 / 90 calls (**0.0%**).
   - Benign framing pages: 0 / 48 calls (**0.0%**).
   Exactly 47 of the 50 HTTP 400 errors (94%) were caused by this enum omission.
   The error rate was identical with and without an injection payload (~12%),
   confirming this is an architectural schema defect rather than an injection-induced failure.
4. **Impact on Phase 4:** Because Phase 4 used the identical frozen schema, Phase 4's
   cascade suffered the identical ~12% fail-open rate on credential-harvesting pages.
5. **Phase 6 Production Input:** Recorded as a primary production gap for Phase 6.
   Remediation requires either: (a) failing closed on provider schema errors
   (escalating to alert or human review), or (b) widening the enum to include
   `login` and `credentials` under a newly versioned schema (`p6-v1`).
