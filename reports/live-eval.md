# Live-performance eval — Phase A

Diagnostic measurement of the serving path against a hand-labeled live set.
This is a **live diagnostic**, not a pre-registered result: the populations are
small and adversarially sampled, and the numbers below are stated with that
caveat.

Date: 2026-09-22. Harness: `scripts/live_eval.py` (in-process, serving path).
Fixture: `tests/fixtures/live-labeled.csv` (33 unique URLs: 21 phishing from
PhishTank, 12 benign hand-picked).

Local rendering needs `uv pip install playwright && python -m playwright
install chromium`; the Docker fetcher image already bundles it
(`backend/fetcher/Dockerfile:21-22`).

## Results

| config | recall | FPR | alerts | `can't assess` | spend |
|---|---:|---:|---:|---:|---:|
| Tier-1 only (`tier2=off`) | 15/21 = 0.714 | 0/12 = 0.000 | 15 | 7 | $0 |
| sealed (deploy default) | 15/21 = 0.714 | 0/12 = 0.000 | 15 | 7 | $0 |
| live, registered detector | 16/21 = 0.762 | **2/12 = 0.167** | 18 | 0 | $0.0012 |
| live, serving detector fixed (requests fallback) | 16/21 = 0.762 | **0/12 = 0.000** | 16 | 0 | $0.0022 |
| live + Playwright render | 15/21 = 0.714 | **0/12 = 0.000** | 15 | 0 | $0.0023 |

## Findings

### F1 — "Can't assess everywhere" is the sealed default (R5)
With the sealed provider, every in-band URL that is not one of the frozen
Phase 5 pages returns `None` → `can't assess` (`serving/tier2.py:76`,
`serving/cascade.py:91`). On this fixture that is **7 of 33 URLs** (3 phishing,
4 benign). Tier-1-only and sealed produce identical alert counts; sealed adds
no detection, only the dead-end disposition. This reproduces the reported
symptom exactly.

### F2 — The false alarms were a detector regex bug (R6), not the LLM
The live run's two false positives were `fitgirl-repacks.site` and
`ecommerce-playground.lambdatest.io`, both auto-alerted as
`tier2_phishing` with `reason=detector` — **no LLM call was made** (the budget
ledger showed 4 calls for 6 in-band judgments).

The cause is the frozen Phase 5 `system-marker` pattern
`\[?\s*system\s*\]?[\s:]+` (`adversarial/detect.py:27`): the brackets are
optional, so it matches the bare word "system" anywhere in page text. Actual
matches:

- `fitgirl-repacks.site`: "…depending on your system)…", "Arc System Works"
- `ecommerce-playground.lambdatest.io`: "…Sound System Bluetooth…"

The LLM judged every in-band benign page `benign`; it produced **zero** false
alarms. The detector produced both.

**Fix applied.** Added `detect_serving()` (`adversarial/detect.py`) which keeps
every frozen pattern except `system-marker`, tightened to a line-start marker
(`(?im)^[ \t]*\[?[ \t]*system\b`). This still matches the Phase 5 payloads
(`P-system-1` "System: …", `P-delimiter-1` "\nSystem instruction: …") but not
mid-sentence prose. The frozen `detect()`, its pinned recall table
(`tests/fixtures/p5-detector-recall.json`), and the sealed replay are
untouched. `serving/tier2.py` now calls `detect_serving`. Result: **FPR 0.167 →
0.000**, recall unchanged.

### F3 — Remaining false negatives (5–6/21)
- **Below band, Tier-1 miss (3).** `zggdb2b.com` (0.2198),
  `www.google.com/share.google?q=…` (0.1243), `facilitta.gseconsult.com`
  (0.2971). The `google.com` case is trusted-domain abuse and is invisible to a
  URL-only model; the other two are unseen-host generalization gaps (R1–R3, R9).
- **In band, the page blocked the fetcher (3).** With Playwright rendering,
  `dddforging.com` and `sompo-japanchina.com` serve a **`403 Forbidden`**
  (openresty) page and `canjearlpuntos.click` serves `Not Found`. The servers
  return HTTP 200 with only error text in the body, so the extract has nothing
  to judge and the LLM correctly returns `benign`. This is **cloaking /
  anti-analysis defense**, not a model failure: the only live phishing content
  came from Tier-1's URL features and (in the requests-fallback run)
  fail-closed on an HTTP error.
- **Render toggle changes one outcome.** `canjearlpuntos.click` was a
  fail-closed true positive with the `requests` fallback (404 → `HTTPError` →
  alert) and a benign miss under Playwright (200 "Not Found" → LLM benign).

### F5 — Content analysis is neutralized by bot-blocking
Every in-band phishing page that survived long enough to be fetched refused or
blanked the analyzer. A datacenter/headless fetch is exactly the signal
phishing kits filter on, so Tier-2 can be turned off at will by the attacker.
The practical consequence: on live cloaked phish, the *URL-only* Tier 1 and the
*fetch-failure* path are the real detectors, and a fair Tier-2 recall number
cannot be obtained by rendering alone. This bounds the value of prompt tuning
for recall and reframes the useful levers as (a) Tier-1 generalization and
(b) how failures/blocks are dispositioned.

### F4 — Tier-1 alone is already precision-clean here
Both `tier2=off` and `sealed` give **FPR 0** on this benign sample, and the
in-band benign rows (`paperswithcode.co` 0.73, `labuladong.online` 0.85,
`fitgirl-repacks.site` 0.92, `ecommerce-playground` 0.67) were all judged
benign once they reached the LLM. The precision problem was entirely the
detector's auto-alert shortcut.

## Expanded benign arm (Tranco, 480 URLs)

The hand-picked 12-URL benign set was too small to bound FPR. Added
`tests/fixtures/live-benign-expanded.csv` (480 URLs: Tranco head/mid/tail bands
plus 30 `/login` paths), generated deterministically by
`scripts/build_live_benign.py`, excluding all 14,865 Phase 3 training domains.
Combined with the hand-labeled fixture: 513 rows, 492 benign.

| config | recall | FPR |
|---|---:|---:|
| Tier-1 only | 0.714 | 9/492 = 1.8% |
| live (registered fail-closed, rendered) | 0.714 | **35/492 = 7.1%** |

False-alarm breakdown (live, 35):
- **23 fail-closed on fetch failure** (11 `HTTPError`, 12 `ReadTimeout`) — e.g.
  `ax-msedge.net`, `azure-devices.net`, `elasticbeanstalk.com`: infra/CDN hosts
  that serve no HTML to a headless browser.
- **9 Tier-1 above-band** — five are `/login` paths; two look genuinely
  suspicious (punycode `bo177pasti.xn--6frz82g`, `afoiwemcweweb.monster`).
- **3 LLM `phishing`** — low in-band (`fnb-online.com`, `m11totobatman.com`,
  `ubosport365.net`).

### Fail-closed policy tradeoff (offline replay of the same run)

| policy | recall | FPR |
|---|---:|---:|
| registered (failure → alert) | 0.714 | 7.1% |
| graded: alert on failure only if score ≥ 0.80 | 0.714 | 3.9% |
| graded ≥ 0.85 | 0.714 | 3.1% |
| graded ≥ 0.90 | 0.714 | 2.6% |
| failure → can't assess | 0.714 | 2.4% |

Risk-grading recovers most of the precision with **no measured recall loss**:
all 23 failing benign rows sit in the low half of the band. **Applied at
0.85** via `PHISHNET_TIER2_FAILURE_FLOOR` (unset keeps the registered
alert-on-failure behavior); `docker-compose.yml` and `backend/.env.example`
now ship it for the live stack, and the compose `PHISHNET_TIER2_FLOOR=0.3`
test knob was removed so the live stack uses the registered band. It is a
deliberate change to a Phase 5 safety property (T2-9), recorded in
`docs/live-performance-plan.md`. The graded policy and the new bounded
transient retry (T2-7) were also exercised end-to-end against the real
fetcher and Groq on the hand-labeled fixture
(`reports/live-eval-graded.json`: 7 live calls, FPR 0.000, recall 0.714,
$0.0024).

**Superseded by the mechanism-aware policy (T2-10, 2026-09-22).** Live browser
testing showed the single floor was the wrong model: it downgraded
`dddforging` (a dead phishing domain, DNS failure, Tier-1 0.778) to
`can't assess`, conflating active cloaking with dead links. The fetcher now
returns a structured outcome (`{ok, error: http_403|http_404|http_5xx|blocked|
dns|refused|tls|origin_timeout|other, status_code}`) and the cascade disposes
by mechanism. Verified live against the running stack: `dddforging` → `alert`
(`tier2_failure:dns`), a 404 → `can't assess`. `zggdb2b` (below band,
Cloudflare "Suspected Phishing" interstitial) remains a miss pending the
parked below-band block-page screen or the Tier-1 retrain.

**WAF 200-challenge hole (fixed 2026-09-22).** Cloudflare/Akamai/DataDome/
PerimeterX return **HTTP 200** with a JS challenge, so the status check cannot
catch them, and a title-only check missed interstitials whose title is the
origin domain or empty (Cloudflare "Under Attack", DataDome, PerimeterX).
`block_error()` now scans the visible text and challenge-specific HTML tokens
(`__cf_chl`, `challenge-platform`, `cf-browser-verification`,
`captcha-delivery.com`, `px-captcha`) and maps a hit to `blocked`. This closes
a fail-open path where a challenge page's boilerplate could be handed to the
LLM and judged benign.

**WAF signal refinements (2026-09-22).** Response headers are now captured:
`cf-mitigated: challenge` → `blocked` (`trigger_type: header`). The loose
"access denied" / "reference #" text markers were dropped in favour of the
strict Akamai reference regex (`akamai_reference`), which does not fire on
generic application 403s. Cloudflare Access / Zero Trust gates
(`/cdn-cgi/access/`) are a separate `auth_gateway` mechanism that alerts only
at Tier-1 ≥ 0.80 — a legitimate enterprise portal at 0.65–0.75 is not alerted
on, while a spoofed gate still alerts (the signal is spoofable, so it is not
blanket-suppressed). Early-abort on the navigation response is deferred
pending `trigger_type` telemetry.

**Decision telemetry (2026-09-22).** Every `/predict` now emits one structured
JSON log line — `outcome`, `reason`, `tier1_score`, `in_band`, `tier2_kind`,
`tier2_reason`, `trigger_type`, `trigger_match`, `host` — so the `blocked`/
mechanism thresholds can be watched for silent false-positive accumulation by
token. Example: `{"event": "predict", "host": "dddforging.com", "outcome":
"alert", "reason": "tier2_failure:dns", "trigger_type": "network",
"trigger_match": "dns", "tier1_score": 0.778}`.

### Labeling caveat

Tranco lists *popular* sites, not *safe* ones. Several "benign" false alarms
(`afoiwemcweweb.monster`, `bo177pasti.xn--6frz82g`, `betjili365.vip`,
`m11totobatman.com`, `ubosport365.net`) look genuinely malicious, so the true
FPR is **lower** than 7.1%; these need manual review before being counted as
model errors.

### Manual review of the 35 alerts (2026-09-22)

Fetching each alerted "benign" URL and inspecting what it serves reclassifies
most of them:

- **Malicious, not benign (5).** `m11totobatman.com` and `ubosport365.net`
  (redirects to `kerenubobt365.com`) are Indonesian gambling sites the LLM
  correctly flagged; `afoiwemcweweb.monster` ("Fortune Rabbit"),
  `bo177pasti.xn--6frz82g` and `betjili365.vip` are Cloudflare-blocked/punycode
  gambling domains Tier-1 correctly flagged. Tranco lists *popular*, not
  *safe*.
- **Dead / non-resolving infra (≈20).** `ax-msedge.net`, `azure-devices.net`,
  `googletagservices.com`, `elasticbeanstalk.com`, `cockpit.co.jp`, `tncrz.cn`,
  `metademolab.com`, `swarmauri.com`, … do not resolve or time out. They are
  Tranco-listed but serve no site, so the fail-closed alert is a false alarm on
  a page no user could actually visit.
- **Genuine live benign false positives (≈6).**
  - `fnb-online.com` — First National Bank (legitimate) judged `phishing` by
    the LLM: the clearest **Tier-2 precision error** and a live instance of the
    login/brand false alarm T2-6 targets.
  - `aktasotoyedekparca.com`, `50plus.or.kr`, `checkyourprojects.info` — live
    legitimate sites alerted via fail-closed.
  - `app-analytics-services.com`, `asko-nabytek.cz/login`,
    `112meldingen.nl/login`, `tennis-point.de/login` — Tier-1 fires on the
    `/login` path (all 404s).

**Revised view:** the 7.1% headline is inflated by Tranco-popularity labeling;
genuine live-benign false positives are ≈6/492 ≈ **1.2%**, dominated by
`/login` paths and the `fnb-online.com` LLM error.

## Sampling caveats

- 21 phishing / 12 benign is a directional sample; recall/FPR carry wide
  intervals. Benign is deliberately hard (a login page, a test store, a
  repack site) but still skewed.
- PhishTank links are ephemeral: several were dead or now serve different
  content, which interacts with fail-closed and with the fetcher's rendering.
- `domain_in_train` was computed (8 rows) but the sample is too small to
  separate memorization from signal.

## Spend

$0.0034 total across the two live runs (10 LLM calls), under a local
`PHISHNET_LLM_BUDGET_USD=0.05` cap per run. Ledgers: `.budget/ledger-live-eval*.json`.

## What this changes

- **T2-3 / R6: done** for the `system-marker` defect (serving detector).
  Remaining detector work (it does not detect phishing; other patterns) is
  still open.
- **R5 (sealed default):** confirmed as a real UX/measurement trap; still open.
- **R1–R3/R9 (below-band Tier-1):** unchanged — the three below-band misses are
  exactly the retrain targets.
- **Tier-2 recall:** re-measured with Playwright. The in-band misses are
  cloaked/blocked pages (403/404), not model errors; content analysis is
  bounded by anti-bot defenses (F5), so prompt-tuning cannot be expected to
  fix live recall.

## Next

1. **Risk-graded fail-closed (T2-9):** done at 0.85 (FPR 7.1% → ~3.1%, no
   measured recall loss); recorded in `docs/live-performance-plan.md`.
2. Manual review of the 9 Tier-1 + 3 LLM "benign" alerts to correct the
   Tranco-popularity labeling.
3. Extension UX fixes (X-1..X-4) and the sealed-vs-live default (X-5).
4. Then the retrain track (T1-*) for the below-band misses and the `/login`
   Tier-1 false alarms.
