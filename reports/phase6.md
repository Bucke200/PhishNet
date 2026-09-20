# Phase 6 — serving, demo, and production hardening

Protocol: `docs/phase6-preregistration.md` (Amendments `phase6-A…F`).
Results artifact: `reports/phase6.json`. Review unit: tag `phase-6-close`.

Base: `master` @ `ec95e1f` on branch `phase-6`. Tier 1 is the Phase 3 row (a)
LightGBM (`ablation_lexical_gbm_model.pkl`,
`7b765bfc…4024f`; columns `39d0e665…d79e`, 79 columns), scheme-canonicalized.
Frozen thresholds (from `reports/phase4.json`, verified at startup, never
swept): `t_alert = 0.9269363298832987`, `t_1pct = 0.8780843789420926`,
`lower_edge = 0.6493076453312958`.

## Criteria

| # | Criterion | Class | Verdict |
|---|---|---|---|
| C1 | Serving identity, max abs diff 0.0 on calib + test | blocking | **PASS** |
| C2 | Tier-1 single-URL p50 < 10 ms | blocking | **PASS** |
| C3 | Fail-closed Tier-2, Retain unreachable | blocking | **PASS** |
| C4 | `p6-v1` schema replay of the 50 failures | descriptive | 50/50 parsed (1.0 → 0.0) |
| C5 | Shortener resolution, no FPR claim | blocking (demo) | resolver + tests |
| C6 | Demo artifacts | — | extension + script + container transcript |

## C1 — Serving identity

`phishnet.serving.Tier1Servable` is a pandas-free LightGBM fast path. Against
the Phase 3 headline scorer (`snapshot.tier1.score_band`, eval-mode
`EnrichedGbm`) it is **bit-equal on every row of both bands**:

| band | n | max abs diff |
|---|---:|---:|
| calib | 13,157 | 0.0 |
| test | 24,819 | 0.0 |

Startup loads the model and column SHA256 against `model_manifest.json` and
the three thresholds against `reports/phase4.json` (trust anchor: the
registered constants), and refuses to start on any mismatch — pinned by
`tests/test_serving_identity.py`. Row (a)'s vocabulary has no enriched
columns, so the snapshot join is not on the serving path at all.

## C2 — Latency (criterion 12)

Tier-1 single-URL latency, Phase 3 serving shape (in-process, stub provider,
per URL, n = 300, seed 0, 20 warmups):

| path | p50 | p90 | p99 |
|---|---:|---:|---:|
| **serving fast path (ships)** | **0.45 ms** | 0.65 ms | 1.88 ms |
| extractor only | 0.29 ms | 0.43 ms | 0.77 ms |
| `featurise_frame` at n = 1 | 10.61 ms | 11.72 ms | 13.56 ms |
| fast row + `booster_.predict` | 0.61 ms | 0.80 ms | 1.45 ms |

Criterion 12 is **met**: 0.45 ms p50 against the 14.3 ms it replaces (Phase 3
recorded 14.53 ms in `reports/phase3-ablation.json`; the report text carried
14.3). The §0 probe's noun correction holds: the extractor is 0.29 ms; the
fixed cost was per-call pandas frame construction (~10 ms here) plus the
sklearn wrapper. `production-gaps.md` §7's Cython/Rust remediation is
withdrawn — a compiled extractor would buy nothing measurable.

HTTP end-to-end **inside the container** (`docker run -p 8000:8000`,
descriptive, not held to the bar): p50 **7.14 ms**, p90 11.01 ms, p99
28.79 ms on the same 300 URLs, with the response `tier1_score` bit-equal to
the in-process scorer (max abs diff 0.0 over 300 checked rows) and
`/health` reporting the pinned model hash. The image needed `libgomp1`
(LightGBM's OpenMP runtime), found only by running it; `backend/Dockerfile`
installs it.

## C3 — Fail-closed Tier-2

Serving Tier 2 is the registered `p5-h1` prompt plus the frozen detector
(`phase6-A`): a detector hit escalates before any LLM call; a valid parsed
`benign`/`suspicious` verdict keeps the sub-threshold Tier-1 score; and every
other outcome — schema 400, refusal, API error, timeout, unfetchable, or no
Tier-2 configured — maps to **alert** at `nextafter(t_alert, +inf)`. The
Retain policy is unreachable: no mapping, flag, or config produces it, and an
invariant test (quoted-sentinel + behavioral pin) blocks re-adding it, the
same pattern as the whitelist test.

Tier 2 is invoked only for in-band rows; out-of-band rows never fetch a page
or call the model, so ordinary traffic costs only the local score. A
configured provider with no verdict for a URL (e.g. a live page outside the
sealed demo set) is `can't assess` / `tier2_no_verdict`, distinct from a
missing provider (`tier2_not_configured`).

No new threshold is introduced; `production-gaps.md` §1's provisional
`≥ 0.80` rule is not adopted (never fixed on calib).

Cost, stated and not measured: Phase 5 saw 0/90 schema errors on clean benign
pages (Wilson 95% upper bound ≈ 4.1%), and Phase 4's in-band benign
fetchability is ~89%, so failing closed on unfetchable in-band pages is
bounded but non-zero. Both are reported beside C3, never folded into a
headline.

## C4 — `p6-v1` schema

`credential_types` widened to add `login`, `credentials`, `generic_form`,
`session_token`; nothing else changed. `p4-v1`/`p5-h1` stay frozen and
loadable (`p5-h1` is prompt-only over the `p4-v1` schema).

The 50 sealed Phase 5 calls that failed validation were re-sent 1:1 against
their frozen, hash-verified extracts under `p6-v1`
(`runs/phase6/p6-replay/`):

| | value |
|---|---|
| source failures | 50 |
| replayed | 50 (44 fresh, 6 deduplicated under the single `p6-v1` prompt) |
| parsed | 50 |
| schema-error rate | 1.0 → **0.0** |
| transient 429s | 4 (backed off) |

No evasion or detection claim is made from this replay; Phase 5's arm verdicts
remain attached to `p4-v1`/`p5-h1`.

## C5 — Shortener handling

Covered shortener hosts are resolved before scoring (GET, aborting before any
body; ≤ 5 hops; 2 s budget; loop detection), then the **final** URL is scored
through the unchanged Tier 1. Unresolved → `can't assess` with reason
`unresolved_shortener`, and no score is shown as a verdict. The host list is
imported from the extractor, so the resolver and the `is_shortened` feature
cannot drift.

The §0 attribution re-run on 200 calib benign URLs × 5 covered hosts (1,000
rows) is consistent with the draft and with Phase 5:

| | alert @ t05 | alert @ t10 | in band |
|---|---:|---:|---:|
| as served | 0.987 | 0.998 | 0.013 |
| `is_shortened` forced to 0 | 0.595 | 0.812 | 0.396 |

Most of the effect is the short-host / random-slug shape, not the flag;
forcing the flag off is also serve-time skew. Stripping `is_shortened` is
rejected. **No FPR claim is made** — Phase 5's shortened links are synthetic
slugs that do not resolve, so the benign control cannot be re-run on them.

## C6 — Demo artifacts

- The extension (`extension/`) calls the container and shows the
  disposition, the Tier-1 score, and top-k native SHAP contributions;
  thresholds are read from `/health`, never hard-coded. `/report` and the
  feedback write path are removed. A manifest `key` pins the extension ID
  (`cphacgebncakdmjbpoibajnihhbbcjec`) for the CORS allowlist
  (`phase6-E`); the service allowlists that ID plus localhost, without
  credentials.
- Tier 2 runs from the **sealed Phase 5 cache** by default; live Groq calls
  are opt-in behind `GROQ_API_KEY` + `PHISHNET_FETCHER_URL`, and the response
  labels the mode. The separate Playwright fetcher image
  (`backend/fetcher/Dockerfile`, `phishnet.fetcher.app`) backs live mode.
- Both images were built and run: the fetcher rendered `example.com`, and the
  live cascade was smoke-tested once on the phishing scenario URL (fetched,
  judged `benign` under `p6-v1`, `tier2_mode: live`). Live and sealed can
  disagree — the sealed Phase 5 verdict for that page is `phishing` — which
  is exactly why the response and the demo label the mode.
- `scripts/p6_demo.py` drives the three scenarios; the transcript is
  `reports/phase6-demo.json`:

| scenario | disposition | Tier-1 | Tier-2 | top SHAP |
|---|---|---|---|---|
| phishing | alert | 0.8713 | phishing (LLM) | subdomain_count, special_char_ratio |
| benign | allow | 0.7892 | benign (LLM) | has_file_extension_in_path, subdomain_count |
| injection | alert | 0.8127 | detector hit | security_terms_count, subdomain_count |

The transcript above was produced through the running container
(`scripts/p6_demo.py --base http://localhost:8000`) and matches the
in-process run exactly. The browser GIF recording is the remaining operator
step (screen-capture the extension against the container).

## Registered decisions

1. **Tier-1 serving path:** dict → preallocated row → `booster_.predict`;
   the sklearn `predict_proba` wrapper is not used at serving (`phase6-C`).
2. **Playwright in a separate image** so the latency number is not measured
   next to a browser.
3. **Endpoints:** `GET /health`, `POST /predict`, `POST /explain`; the old
   `explain` flag on `/predict` is removed.
4. **CORS:** pinned extension ID + localhost; no wildcard with credentials.
5. **`phishnet.api` retired**; `motor`/`python-dotenv` dropped with
   `/report`. `verified_download` is scoped to the two row (a) artifacts.

## Limitations

- The browser GIF recording (C6) is the only operator-run artifact left; the
  container itself was built and exercised (identity + demo transcript).
- The sealed Tier 2 is an offline replay of Phase 5 verdicts; the live path
  was smoke-tested with a single call, not evaluated at any scale.
- C5 makes no FPR claim. C4 makes no evasion or detection claim.
- The C3 alert-on-failure cost is bounded by Phase 4/5 measurements but not
  newly measured.
