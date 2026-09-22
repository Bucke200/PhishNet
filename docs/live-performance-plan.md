# Live-performance remediation plan

Status: **in progress** (config-first, then retrain). Evidence for every claim
below was verified against the working tree on 2026-09-22; file:line references
are current.

**Phase A complete** (`reports/live-eval.md`). Key result: the live false
alarms were the frozen detector's `system-marker` regex matching the bare word
"system" mid-sentence, not the LLM. Fixed via a serving-only detector
(`detect_serving`, T2-3/R6): live FPR 0.167 → 0.000, recall unchanged. The
remaining false negatives are below-band Tier-1 misses (R1–R3/R9) plus three
in-band misses where the phishing servers returned `403 Forbidden` / `Not Found`
to the fetcher — **cloaking / anti-bot defense, not model errors** (F5). This
bounds how much Tier-2 prompt work can improve live recall, and it adds a new
decision: how to disposition blocked/error pages on in-band URLs.

Expanded benign arm (480 Tranco URLs, `reports/live-eval.md`): Tier-1 FPR 1.8%,
live FPR 7.1%, of which **23/35 are fail-closed on fetch failures** (infra/CDN
hosts serving no HTML). Risk-graded fail-closed (T2-9) cuts FPR to ~3.1% with
no measured recall loss — the top precision win, pending a recorded decision.

## Problem statement

The browser extension performs poorly on live traffic the author tested with a
hand-labeled set (PhishTank phishing vs Codeforces/LeetCode-class benign),
showing **false negatives**, **false positives**, and **`can't assess`
everywhere**. This plan diagnoses that against the *actual serving path* and
fixes it, config-first.

Target: **precision-first** (minimize false alarms) for an extension user.
Posture: **pragmatic** — ship fixes, document after via a retrain memo (not a
full pre-registration).

## Root causes

| # | Cause | Evidence |
|---|---|---|
| R1 | Tier 1 has **no TLD signal** (`tld` computed then dropped; vocabulary confirmed to exclude it) | `features/extraction.py:72-73,297-301`; `feature_columns.pkl` has no `tld` |
| R2 | Brand matching is 8 global brands, fixed Levenshtein | `features/extraction.py:318-327` |
| R3 | Model is a training-distribution shape/reputation memorizer | `model-card.md:120`; `production-gaps.md:37` |
| R4 | Tier 2 is **band-gated** `[0.6493,0.9269)`; the content layer never sees below-band rows | `serving/app.py:86` |
| R5 | Deploy/demo defaults to **sealed** Tier 2 → `can't assess` | `backend/Dockerfile:39`; `render.yaml:21`; `serving/tier2.py:134` |
| R6 | Injection **detector auto-alerts** (framing FP) and does not detect phishing | `adversarial/detect.py:13-30`; `serving/tier2.py:112-113` |
| R7 | Benign skew: shorteners 99.2%, in-band login 20% | `production-gaps.md:19`; `model-card.md:175` |
| R8 | Extension UX: `can't assess` shows the **safe** icon; backend hardcoded | `extension/background.js:9,56-59` |
| R9 | Trusted-domain / redirector abuse invisible to URL-only Tier 1 | `serving/shortener.py:19-34`; `features/extraction.py:19-34` |

## Tier 1 — LightGBM URL scorer

### T1-A Features `[code]`
- **T1-1 Restore TLD signal** — stop dropping `tld`; add `tld_len`,
  `is_common_tld`, `is_abuse_tld`, and a train-only encoding of top suffixes.
  Retrain-only (the frozen vocabulary excludes `tld`).
- **T1-2 Expand abuse-TLD list** beyond `tk,ml,ga,cf,gq,xyz` with a cited
  source (`extraction.py:301`).
- **T1-3 Brand coverage + regional** — expand the list and/or replace fixed
  Levenshtein with subword/character similarity against a larger target corpus.
- **T1-4 Reputation-independent structural features** — host-token count,
  random-run detection, per-token entropy, char-class transitions.
- **T1-5 Redirector/shortener coverage** — add `share.google`-class hosts to
  `SHORTENER_DOMAINS` and the resolver.
- **T1-13 Preserve train/serve parity** through `featurise_frame`.

### T1-B Data `[data]`
- **T1-11 Benign rebalancing** — shorteners, hosted tenants, login/account
  pages, regional sites, dev/programming sites.
- **T1-12 Cold-start / unseen-host population** — out-of-era phishing and
  novel-host benign; keep temporal/domain-disjoint split rules.
- **T1-10 `HOSTED_PLATFORMS` extension** with a cited non-feed source, then
  retrain (flipping at serving is forbidden train/serve skew).

### T1-C Model / thresholds `[retrain]`
- **T1-7 Precision-first training** — reduce positive weight from
  `class_weight="balanced"` (`ml_training/train_gbm.py:128-136`).
- **T1-8 Calibration** — row (a) is uncalibrated; add isotonic/sigmoid
  (`ml_training/calibrate_gbm.py`).
- **T1-9 Threshold refit** on calib for the chosen FP budget; record in
  `reports/phase4.json` and update `serving/tier1.py:49-53`.

### T1-D Serving identity & pins `[code]`
- **T1-14** new weights → release `models-v1`, update
  `src/phishnet/model_manifest.json`, `snapshot/tier1.py:34-36`, and the C1
  identity test.

## Tier 2 — LLM page judge

- **T2-1** keep live on `p6-v1` (widened `credential_types`, `schema.py:26-34`);
  do not let `client.judge`'s `p4-v1` default leak back in (`llm/client.py:73`).
- **T2-2** precision-first prompt: treat consistent-brand login pages as
  benign; require mismatched-brand credential harvesting for `phishing`.
- **T2-3** stop auto-alerting on detector hits; route detector hits to the LLM
  (`serving/tier2.py:112-113`). **Partially done (2026-09-22):** the
  `system-marker` false-positive is fixed by a serving-only detector
  (`detect_serving`); the broader "detector auto-alerts at all" policy is
  still open.
- **T2-4** stricter mapping experiments (LLM `phishing` **and** high Tier-1).
- **T2-5** confidence gating on the LLM `confidence` field.
- **T2-6** domain–brand grounding to kill the 20% login FP (an allowlist is
  deliberately blocked — `serving/cascade.py:16-18`).
- **T2-7 bounded transient retry — DONE (2026-09-22).** `_judge_with_retry`
  in `serving/tier2.py`: one transport/5xx retry after 2 s; a 429 returns
  immediately (the registered 60 s backoff outlives a serving request). A
  single provider blip can no longer turn an in-band page into a
  failure-alert.
- **T2-8** resolve `share.google`-class redirectors before scoring.
- **T2-9 failure policy — DONE (2026-09-22), superseded by T2-10.** The
  one-threshold `failure_floor` (graded) remains available, but the live
  deployment now uses the mechanism-aware policy (T2-10) because a single
  floor conflated active cloaking (403/block) with dead links (404/DNS).

## Cross-cutting

- **X-1..X-4 extension — DONE (2026-09-22).** Backend URL configurable via an
  options page + `chrome.storage` (`extension/options.html`, `options.js`;
  optional host permission requested on save); `can't assess` now uses the
  neutral app icon and is titled "Not assessed"; notifications are deduped per
  URL (60 s); the Tier-2 mode is always shown in the message.
- **X-5** decide/document sealed-vs-live default (`README.md:251-252`,
  `render.yaml:21`).
- **X-6** fixture `tests/fixtures/live-labeled.csv` (provenance + leak flag).
- **X-7** harness `scripts/live_eval.py`. **X-8** FP attribution.
- **X-9** `reports/live-eval.md` with sampling caveats.
- **X-10** `docs/retrain-memo.md`. **X-11** update model card / gaps / roadmap.

## Sequencing

1. **Phase A — Measure** (X-6, X-7, X-8, X-9): fixture + harness + baseline
   runs at `tier2=off`, `sealed`, live@`0.6493`, live@`0.3`, strict mapping.
   Output: the FN/FP decomposition.
2. **Phase B — Cheap fixes** (X-1..X-5, T1-5, T2-3, T2-7, T2-8, T2-4/5/9).
3. **Phase C — Retrain** (T1-1..T1-4, T1-10..T1-12, T1-7/8/9, T1-14).
4. **Phase D — Tier-2 quality** (T2-1/2/6, T2-9 decision).
5. **Phase E — Docs + commit.**

## Decisions (2026-09-22)

1. **R6 / detector `system-marker` (serving).** Added `detect_serving()` with a
   line-start `system-marker`; the frozen Phase 5 `detect()` and its pinned
   recall table are unchanged. Live FPR from this cause: 2/12 → 0/12.
2. **T2-9 / risk-graded fail-closed.** Enabled via
   `PHISHNET_TIER2_FAILURE_FLOOR=0.85`; default unset keeps the registered
   alert-on-failure behavior. Chosen because 23 of 35 live benign false alarms
   were fetch failures on infra hosts in the low half of the band, and grading
   recovered them with no measured recall loss. Deviates from Phase 5's
   unconditional fail-closed property; documented here as the decision of
   record pending a formal amendment. **Deployment:** `docker-compose.yml`
   and `backend/.env.example` now ship `0.85` for the live stack, and the
   compose `PHISHNET_TIER2_FLOOR=0.3` test knob was removed (the live stack
   now uses the registered band). The sealed demo image is unchanged.
3. **T2-10 / mechanism-aware failure policy — DONE (2026-09-22).** Live
   testing showed the single graded floor was wrong: it downgraded
   `dddforging` (a dead phishing domain, DNS failure, Tier-1 0.778) to
   `can't assess`, while the underlying signals are not comparable. The
   fetcher now returns a **structured outcome** (`{ok, error: http_403 |
   http_404 | http_5xx | blocked | dns | refused | tls | origin_timeout |
   other, status_code}`) at HTTP 200, so a target failure is never confused
   with an RPC failure; Playwright and requests normalize status handling;
   the fetcher's origin budget is 8 s and the provider→fetcher RPC budget is
   25 s. `decide()` dispatches on the mechanism
   (`FAILURE_MECHANISM_THRESHOLDS`): `http_403`/`blocked` alert across the
   band, `dns`/`refused`/`tls`/`http_5xx` at ≥ 0.70, `origin_timeout` at
   ≥ 0.80, `http_404` and internal RPC failures never alert, LLM-side
   failures stay fail-closed. Enabled by
   `PHISHNET_TIER2_FAILURE_POLICY=mechanism` (default `closed`).
   Verified live: `dddforging` → `alert` (`tier2_failure:dns`).
4. **WAF challenge detection — DONE (2026-09-22).** Cloudflare/Akamai/DataDome/
   PerimeterX return **HTTP 200** with a JS challenge, so the status check
   cannot catch them and a title-only check missed interstitials whose title
   is the origin domain or empty (Cloudflare "Under Attack", DataDome,
   PerimeterX). `block_error()` now scans the title **and** visible text for
   human-facing challenge phrases and the raw HTML for challenge-specific
   tokens (`__cf_chl`, `challenge-platform`, `cf-browser-verification`,
   `captcha-delivery.com`, `px-captcha`), and maps a hit to `blocked` — never
   to the LLM as page content. Verified on 6 representative interstitials
   (all now caught) and a normal phishing page (not caught).
   **Refinements (2026-09-22):** response headers are captured, so
   `cf-mitigated: challenge` maps to `blocked` with `trigger_type: header`;
   the loose "access denied" / "reference #" text markers were replaced by the
   strict Akamai reference regex (`akamai_reference`), avoiding generic
   application 403s; and Cloudflare Access / Zero Trust gates
   (`/cdn-cgi/access/`) are a distinct **`auth_gateway`** mechanism that alerts
   only at ≥ 0.80, so a legitimate enterprise portal (0.65–0.75) is not
   alerted on while a spoofed gate still alerts. **Deferred:** early-abort on
   the navigation response (`page.on("response")`) is a latency optimization,
   not a correctness fix; the new `trigger_type` telemetry will show whether
   throughput justifies it.
5. **Structured decision telemetry — DONE (2026-09-22).** The serving app logs
   one JSON line per `/predict` (`_log_decision`): `event`, `outcome`,
   `reason`, `tier1_score`, `in_band`, `tier2_kind`, `tier2_reason`,
   `trigger_type`, `trigger_match`, `host`. The trigger detail flows fetcher →
   provider (`Tier2Outcome.trigger_type`/`trigger_match`) → payload → log, so a
   single over-indexing marker (e.g. a WAF token firing on benign traffic) is
   visible instead of accumulating silently. No request payloads or keys are
   logged; only the host.
6. **Verification.** `decide()`/`predict_one`/env/endpoint wiring tested in
   `tests/test_serving_failure_policy.py` and
   `tests/test_serving_failure_mechanism.py`; retry in
   `tests/test_live_tier2_retry.py`; fetcher contract, classification and WAF
   interstitials in `tests/test_fetcher_contract.py` /
   `tests/test_fetcher_classification.py`; telemetry in
   `test_serving_failure_mechanism.py`. Full suite 506 passed, 2 skipped.

## Risks / open decisions

- **Precision vs band gating (R4):** the only URL-side fix for trusted-domain
  abuse is letting Tier 2 see more, which raises FP. Quantified in Phase A.
- **Risk-graded fail-closed** changes a measured Phase 5 safety property.
- **Detector demotion** must escalate to the LLM, never silently allow.
- **Benign sample breadth** must be expanded before any low-FP claim is made.
- **`phase4-G`** is the next free amendment letter (A–F are used); the prior
  session summary's "next is `phase4-F`" was wrong.
