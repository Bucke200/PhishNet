# Phase 7 report — finalization and packaging

Protocol: `docs/phase7-preregistration.md` (Amendments `phase7-A` sealed
Tier-2 URL keys are scheme-canonicalized; `phase7-B` live Tier 2 is
fail-loud and the two-layer stack is `docker compose up`).
Status: **open** — P7-1…P7-4 and P7-6 are met; **P7-5 (demo GIF) is
operator-run and pending**, so the phase does not close and no
`phase-7-close` tag exists yet.

Base: `master` @ `013f715c`. The frozen system (weights, thresholds,
prompts, schemas, code paths) is untouched; the only code added is
documentation tests.

## Criteria

| # | criterion | status |
|---|---|---|
| P7-1 | README claims trace | **met** (table below; pinned by `tests/test_phase7_docs.py`) |
| P7-2 | Model card completeness | **met** |
| P7-3 | Docs map + link integrity | **met** |
| P7-4 | Architecture diagram | **met** (`docs/architecture.md`) |
| P7-5 | 60-second demo GIF | **pending** (operator-run) |
| P7-6 | Authorship | **met** |
| P7-7 | Closing gate | **pending** (blocked on P7-5) |

## P7-1 — README claim trace

| README claim | source |
|---|---|
| 50.4% recall @ 0.40% FPR (`t05`) | `reports/phase3.md`; `reports/phase4.json` |
| 60.7% recall @ 0.98% FPR (`t10`) | `reports/phase3.md`; `reports/phase4.json` |
| cold-start 53.4% (row (b), age unknown) | `reports/phase3.md` |
| 0.45 ms in-process / ~7 ms HTTP | `reports/phase6.json` |
| Phase 4 ceiling +0.0347 | `reports/phase4.md` |
| webflow.io miss | `docs/production-gaps.md` §8 |
| sealed/live Tier-2 modes | `reports/phase6.md` (C6) |
| fail-closed Tier-2 | `reports/phase6.md` (C3) |

Banned stale phrasings (asserted absent): a hosted `phishnet-pavv.onrender.com`
backend, `serves \`phishnet.api\``, `"explain": true`, and the old
`prediction (0 = legit)` response shape.

## P7-2 — Model card

`docs/model-card.md` now carries every roadmap item: intended use; the
inverted depth prior (benign 1.75 vs phishing 1.03; https 97.6% vs 77%) and
the scheme decision (`canonicalize_scheme`); point-in-time classification;
survivorship and the RDAP 404 rate by class; the Tranco selection leak and
age confound; hosted coverage limits; age's gate failure and conditional
result; cold start; calibration shelf life and threshold transfer; why CT
was dropped; and the Phase 4 close-out (fetchability 0.135 vs 0.886,
fingerprint rotation, 22% determinism with the response cache, ceiling
0.0347, scope ending at `phase4-D`).

## P7-3 — Documentation map

`docs/adversarial.md` was added as a pointer page (Phase 5 evidence stays in
`reports/phase5-adversarial.md`). README links the model card,
`docs/adversarial.md`, `docs/architecture.md`, `docs/production-gaps.md`,
`docs/point-in-time.md`, `docs/splits-eval-audit.md`, `docs/WAIVERS.md`,
`docs/phase3-preregistration.md`, and the phase reports. Every relative
Markdown link in `README.md` and `docs/` resolves
(`tests/test_phase7_docs.py`).

## P7-4 — Architecture diagram

`docs/architecture.md` shows the serving path (extension → `/predict` →
Tier-1 fast path → band gate → Tier-2 sealed/live → disposition) with
measured annotations only: 0.45 ms / ~7 ms Tier-1, 1,308 ms LLM p50, $0.367
per 1,000 escalated, band edges 0.6493 / 0.9269. Each annotation cites its
report.

## P7-5 — Demo GIF (pending)

Not produced here: it requires a browser session against the container.
Required by the prereg: ≤ 60 s; safe site (`allow`, "not a safety
guarantee"), phishing page (`alert`), injection page (detector escalation);
on-screen values must equal `reports/phase6-demo.json`; recorded against the
sealed default.

**Operator runbook:**

1. `docker build -f backend/Dockerfile -t phishnet-serving .`
2. `docker run --rm -p 8000:8000 phishnet-serving` (sealed default; no key,
   offline).
3. `chrome://extensions` → Developer mode → **Load unpacked** →
   `extension/`.
4. Visit the three scenario URLs listed in `reports/phase6-demo.json`
   (`phishing`, `benign`, `injection`) and record ≤ 60 s.
5. Confirm on screen: phishing → `alert`; benign → `allow` + "not a safety
   guarantee"; injection → `alert` (detector). SHAP features are shown.
6. Save as `img/phase7-demo.gif`; hash it (`Get-FileHash` / `sha256sum`) and
   add the hash here, then embed it in the README Demo section.

**Mode is load-bearing.** The transcript is a sealed replay of the frozen
Phase 5 verdicts. A live-mode container fetches the current pages, which have
changed since Phase 5: on 2026-09-20 the "phishing" scenario URL
(`adobesign`) judged `benign`, the "benign" redirect page (`linkis.com`)
tripped the detector, and the "injection" URL (`globalbersama`) judged
`benign` — the inverse of the transcript. `scripts/p6_demo.py --base` now
reads `/health` and refuses unless the container is sealed (override with
`--allow-live`). Record the GIF in sealed mode; live mode is a different,
unstable demo and is not what the transcript describes.

Only then can P7-7 run.

## P7-6 — Authorship

Recorded decision: the project is authored by **Srinjay Panja**, MIT
licensed, copyright **2025–2026**. `README.md` byline and copyright, and
`pyproject.toml` authors, agree. The roadmap's "still open" note is
resolved.

## P7-7 — Closing gate (pending)

Blocked on P7-5. When the GIF lands: run `ci` / `repro` / `eval` green on the
closing commit, mark `roadmap.md` Phase 7 done, and tag `phase-7-close`.

## Notes

- The Phase 3 test band is not committed (`data/` is git-ignored; GitHub
  push protection flags phishing URLs in it as secrets). The C1 identity
  test runs the calib band in CI and skips the test band there, running the
  full calib+test locally where the band exists.
- No number in this phase was re-measured. Packaging copies existing
  figures and names their source.
