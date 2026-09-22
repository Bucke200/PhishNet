# Phase 6 pre-registration — serving, demo, production hardening

Status: REGISTERED, committed on branch `phase-6` at `phase6-A`–`phase6-F`
before any serving number in this document's criteria exists. Amendments
`phase6-A…` follow the Phase 3–5 rule: recorded before the numbers they
affect.

Base: `master` @ `ec95e1f` (tag `phase-5-close` + CI fixes).
Tier 1: Phase 3 row (a), `ablation_lexical_gbm_model.pkl`
(`7b765bfc…4024f`), columns `39d0e665…d79e`, 79 columns, scheme
canonicalized (`train_config.json`: `manifest:drop`).
Frozen thresholds (from `reports/phase4.json`, recomputation asserted, never
swept): `t05 = 0.9269363298832987`, `t10 = 0.8780843789420926`,
`lower_edge = 0.6493076453312958`.

---

## 0. Disclosure: probes run before registration

Three exploratory probes were run on the **calib band only** before this
document was written. They informed the design; none is a Phase 6 result.
All are re-run under the registered protocol below.

1. **Serving identity.** A lean path (`featurise_frame(canonicalize=True)` +
   `hosted_flag` + `predict_proba`, no snapshot, no `first_seen` map) matches
   the headline `EnrichedGbm` path with max abs diff **0.0** on 400 calib
   rows. Row (a)'s vocabulary has no enriched columns, so the snapshot join
   contributes nothing to its scores.
2. **Latency attribution.** On 500 calib URLs, in-process:

   | path | p50 | p99 |
   |---|---|---|
   | `comprehensive_phishing_features` alone | 0.12 ms | 0.30 ms |
   | `featurise_frame` at n=1 | 6.09 ms | 9.23 ms |
   | lean path (frame + sklearn wrapper) | 7.68 ms | 8.74 ms |
   | dict → preallocated row → `booster_.predict` | 0.22 ms | 0.37 ms |

   The last path equals the lean path with max abs diff **0.0** over 2,000
   calib URLs. The Phase 3 attribution ("~7.8 ms fixed per-call **extractor**
   overhead") is wrong in its noun: the extractor costs 0.12 ms; the fixed
   cost is per-call pandas frame construction and coercion (~6 ms) plus the
   sklearn wrapper (~0.7 ms). `production-gaps.md` §7's remediation
   (Cython/Rust) is withdrawn by this document; a compiled extractor would
   buy nothing measurable.
3. **Shortener leak attribution.** 200 calib benign URLs × 5 covered
   shortener hosts (`shortener_wrap`, seed as Phase 5), 1,000 rows:

   | | alert @ t05 | alert @ t10 | in band |
   |---|---|---|---|
   | as served | 0.990 | 0.995 | 0.010 |
   | `is_shortened` forced to 0 | 0.604 | 0.807 | 0.388 |

   Per host at t05 with the flag forced off: bit.ly 0.00, tinyurl 0.24,
   t.co 0.91, is.gd 0.94, cutt.ly 0.93. The flag carries part of the leak;
   the rest is the short-host / random-slug shape. Forcing the flag off is
   also serve-time skew against a model trained with it. **Stripping
   `is_shortened` at serving is rejected** as a remediation. (Consistent
   with Phase 5's 0.992 on a different benign sample.)

---

## 1. Scope

In: a container serving the registered Tier-1 → Tier-2 cascade; the
latency fix; fail-closed Tier-2 handling; schema `p6-v1`; shortener
handling; demo artifacts.

Out (unchanged from roadmap, tracked in `production-gaps.md`): live RDAP +
cache, monitoring stack, feedback-poisoning pipeline, calibration refresh.
Consequence for the existing API: `/report`, MongoDB logging and the
`motor` dependency are removed, not hardened. A feedback endpoint whose
pipeline is cut is an unmaintained write path.

No retraining. Every Phase 3 headline number stays attached to the exact
weights above. Anything that would need new weights (dropping
`is_shortened`, TLD cold-start prior, reputation-independent features) is
future work under its own protocol.

---

## 2. Criteria

### C1 — Serving identity (blocking)

The container's `/predict` Tier-1 score equals `tier1.load_row_a` headline
scores with **max abs diff 0.0** on every row of the calib band and the test
band, scored one URL per request through the serving code path.

At startup the service verifies the model and column SHA256 against
`model_manifest.json` and the three thresholds against `reports/phase4.json`,
and refuses to start on any mismatch (no degraded mode). The response
carries `model_hash`, `thresholds_source`, and `disposition`.

### C2 — Latency, criterion 12 carried over (blocking)

Tier-1 single-URL p50 **< 10 ms**, measured in the Phase 3 serving shape
(in-process, stub provider, per URL, n = 300, same sampling), so the number
is comparable to the 14.3 ms it replaces. Reported beside it,
descriptive and not held to the bar: HTTP end-to-end p50/p90 inside the
container on the same 300 URLs.

The single-URL fast path must be bit-equal (max abs diff 0.0) to
`featurise_frame` on the full calib and test bands. A pinned test enforces
it in CI, as `_features_single` already does for the legacy pipeline.

### C3 — Fail-closed Tier 2 (blocking)

For an in-band row, every Tier-2 outcome other than a valid, parsed judgment
(schema 400, refusal, API error, timeout, unfetchable page) maps to the
**alert** disposition at `nextafter(t_alert, +inf)`. It never retains the
Tier-1 score. Retain is unreachable from serving config; an invariant test
blocks re-adding it (same pattern as the whitelist test).

No new threshold is introduced. The `≥ 0.80` rule floated in
`production-gaps.md` §1 is not adopted: it was never fixed on calib.

Cost, stated and not measured: Phase 5 observed 0/90 schema errors on clean
benign pages (Wilson 95% upper bound ≈ 4.1%). The benign cost of failing
closed on unfetchable pages is bounded by Phase 4's in-band benign
fetchability (~89% fetchable). Both are reported beside C3, never folded
into a headline.

### C4 — Schema `p6-v1` (descriptive)

`credential_types` widens to add `login`, `credentials`, `generic_form`,
`session_token`, and nothing else changes. `p4-v1` / `p5-h1` stay frozen and
loadable.

Replay: the 50 sealed Phase 5 calls that failed validation are re-sent
under `p6-v1` against their frozen extracts (≤ 150 calls with 3 repeats,
within free-tier quota). Reported: the schema-error rate on those pages,
before and after. No evasion or detection claim is made from this replay.
Phase 5's arm verdicts remain attached to `p4-v1` / `p5-h1`.

### C5 — Shortener handling (blocking for the demo, descriptive for numbers)

For hosts on the `is_shortened` list only:
- resolve redirects with HEAD requests, at most 5 hops, 2 s total budget,
  no body fetch;
- score the **final** URL through the unchanged Tier 1;
- report both URLs in the response.

If resolution fails, the disposition is `unresolved_shortener` ("can't
assess"), with no score shown as a verdict. Unit tests cover resolution,
the hop cap, timeout, loops, and a destination that is itself a shortener.

No FPR claim is made: Phase 5's shortened links are synthetic slugs that do
not resolve, so the benign control cannot be re-run on them.

### C6 — Demo artifacts

- The extension calls the container and shows the disposition, the score,
  and the top-k native SHAP contributions (row (a) is a raw LGBMClassifier,
  so `pred_contrib` works and the 501 path retires). Thresholds are read
  from the service, never hard-coded in the extension.
- A scripted recording covers a benign site, a phishing page, and an
  injection page.
- Tier 2 in the demo runs from the **sealed response cache** by default.
  Live Groq calls are opt-in behind an env key, and the demo labels which
  mode produced each verdict.

---

## 3. Registered decisions

1. **Tier-1 serving path:** the fast path (dict → preallocated row →
   `booster_.predict`), gated by C1 and C2 bit-equality. The sklearn
   `predict_proba` wrapper is not used at serving.
2. **Playwright in the image:** a separate `fetcher` service/image, so the
   Tier-1 image stays slim and the latency number isn't measured next to a
   browser.
3. **Endpoints:**
   - `GET /health`: assets verified, thresholds loaded, Tier-2 mode;
   - `POST /predict`: Tier 1, plus Tier 2 when in band;
   - `POST /explain`: SHAP for the scoring model only.

   The old `explain` flag on `/predict` is removed.
4. **CORS:** restricted to the pinned extension ID and localhost; no
   wildcard with credentials.

---

## 4. Order of work

1. Commit this document on `phase-6` (review unit starts here).
2. C1 + C2: new `phishnet.serving` module, replacing the legacy `api.py`
   paths, plus bit-equality tests. Numbers go into `reports/phase6.md`.
3. C3 + C4: cascade dispositions, `p6-v1`, invariant test, replay.
4. C5: resolver + tests.
5. Docker (`backend/Dockerfile` rewritten for the row (a) assets), extension
   update, recording.
6. Close: `reports/phase6.md`, `production-gaps.md` amended (§7 withdrawn,
   §2 re-attributed), `roadmap.md` Phase 6 amended (the Cython/Rust and
   strip-`is_shortened` options withdrawn), tag `phase-6-close`.

---

## 5. Amendments (registered before the numbers)

- **`phase6-A` — Tier-2 serving configuration.** Serving Tier 2 is the
  `p5-h1` hardened prompt plus the frozen pure-function detector
  (`phishnet.adversarial.detect`, `phase5` commit 1), under the fail-closed
  mapping of C3. A detector hit escalates to alert; a valid parsed
  non-phishing judgment retains the Tier-1 score (below `t_alert`, i.e. not
  an alert); every other outcome alerts. This is the only Phase 5 arm that
  passed Criterion 1. Its framing cost (`production-gaps.md` §6: 50.0% on
  the framing sample, `N=8`) is disclosed beside every cascade number and is
  never folded into a headline.
- **`phase6-B` — C4 replay unit.** C4 re-sends the **50 sealed Phase 5 calls
  that failed validation, one for one, under `p6-v1`** (50 calls, not 150).
  Each sealed failure already carries its `prompt_version` and `repeat_idx`,
  so the replay is paired before/after on the same page set. The "≤ 150
  calls with 3 repeats" reading in C4 is superseded: three fresh repeats of
  the same 50 would answer a determinism question C4 does not ask.
- **`phase6-C` — C1 coverage.** C1 scores the **full calib band (13,157
  rows) and full test band (24,819 rows)** through the serving code path,
  one URL per request, against `tier1.load_row_a`. No subsampling.
- **`phase6-D` — C5 resolver method.** Redirects are followed with **GET,
  aborting after the response headers and before any body is read** (HEAD is
  rejected by several covered hosts and `goo.gl` is dead). ≤ 5 hops, 2 s
  total budget, loop detection. The authoritative shortener list is the one
  already used by the `is_shortened` feature
  (`phishnet.features.extraction`); serving must import that list rather
  than re-declare it, so the resolver and the feature cannot drift.
- **`phase6-E` — C6 extension identity.** A `key` is added to
  `extension/manifest.json` so the extension ID is stable across machines,
  and the CORS allowlist is that pinned ID plus localhost (registered
  decision 4 becomes achievable). `extension/background.js` currently
  targets a Render host while `host_permissions` allows localhost only; both
  are aligned to the container.
- **`phase6-F` — Serving-time registrations.**
  - The three thresholds are read from `reports/phase4.json` (bundled into
    the image), not hard-coded; `C1`'s startup check verifies them and the
    model/column hashes, refusing to start on mismatch.
  - Disposition vocabulary: `allow` (Tier-1 below `lower_edge`),
    `alert` (Tier-1 ≥ `t_alert`, or a Tier-2 phishing verdict, or any
    fail-closed Tier-2 failure), `can't_assess` (an in-band row whose
    shortener could not be resolved), and `unresolved_shortener` as the
    C5-specific cause carried alongside `can't_assess`.
  - `phishnet.api` is retired in favor of `phishnet.serving.app`; `motor`
    and `pymongo` drop from `pyproject.toml` with `/report`. The
    `verified_download` call used by the image is scoped to the two row (a)
    artifacts rather than all five manifest entries (the 93 MB urlset
    ensemble is no longer fetched).
  - `roadmap.md` Phase 6 keeps its "Criterion 12" bullet but the mechanism
    filed under it ("Extractor Latency Optimization", Cython/Rust) is
    withdrawn in favor of the serving fast path, consistent with the §7
    withdrawal in `production-gaps.md`.
