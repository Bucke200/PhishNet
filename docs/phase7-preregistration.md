# Phase 7 pre-registration — finalization and packaging

Status: REGISTERED, committed on branch `phase-7` before any artifact named
in its criteria exists. Amendments `phase7-A…` follow the Phase 3–6 rule:
recorded before the numbers they affect.

Base: `master` @ `013f715c` (Phase 6 closeout, README rewrite, CI fix).
Frozen inputs this phase must not touch: row (a) weights
`7b765bfc…4024f`, columns `39d0e665…d79e`, thresholds `t_alert =
0.9269363298832987`, `t_1pct = 0.8780843789420926`, `lower_edge =
0.6493076453312958`.

Phase 7 is packaging, not measurement: it makes the finished system read
correctly and closes the project. Every figure it publishes is copied from
an existing report, with the source named.

---

## 0. Disclosure: work landed before registration

Two packaging changes were already merged to `master` before this document:

1. **README rewrite (`1ccb751b`).** It leads with the fixed-threshold
   numbers, corrects the retired system (URLSet ensemble → Phase 3 row (a);
   no Render backend, MongoDB, `/report`, or `explain` 501), and documents
   the sealed/live serving modes. It is **re-verified** under P7-1 rather
   than treated as a Phase 7 result.
2. **`backend/.env.example`** rewritten for live-mode env (Groq key, fetcher
   URL, Tier-2 mode) instead of the removed MongoDB URI.

No model, threshold, feature, or serving behavior was changed by either.
`data/` stays git-ignored; the Phase 3 test band is not committed (GitHub
push protection flags phishing URLs in it as secrets), so the C1 identity
test skips that band in CI and runs it locally.

---

## 1. Scope

**In:**
- a final consistency pass over README, the model card, and the docs map;
- `docs/adversarial.md` (listed by the roadmap, currently absent);
- an architecture diagram with measured latency and cost annotations;
- the 60-second demo GIF (operator-run) and its frozen transcript;
- authorship confirmation;
- the closing gate: workflows green, `reports/phase7.md`, tag.

**Out** (recorded in `production-gaps.md` and `roadmap.md` future work):
- any retraining, threshold change, or feature-list change — including the
  `webflow.io` hosted-coverage gap (`production-gaps.md` §8), which needs a
  fresh population and new weights;
- new measurements or new LLM calls: packaging copies existing numbers only;
- the Phase 4 recorded sweep;
- any hosted deployment.

---

## 2. Criteria

### P7-1 — README claims trace (blocking)

Every quantitative claim in `README.md` is listed in the trace table of
`reports/phase7.md` with the artifact it comes from. Required values:
50.4% @ 0.40% FPR, 60.7% @ 0.98%, cold-start 53.4%, 0.45 ms in-process /
~7 ms HTTP, Phase 4 ceiling +0.0347, and the webflow.io miss. Banned stale
claims: a hosted `onrender` backend, MongoDB setup, `explain: true` → 501,
or `phishnet.api` as the serving app. `tests/test_phase7_docs.py` pins the
required and banned strings.

### P7-2 — Model card completeness (blocking)

`docs/model-card.md` carries every item the roadmap lists: the inverted
depth prior and the scheme decision; point-in-time classification;
survivorship with the RDAP 404 rate by class; the Tranco selection leak and
the Tranco-age confound; hosted coverage limits; age's gate failure and the
conditional result; cold-start degradation; calibration shelf life and the
threshold-transfer verdict; why certificate history was dropped; and the
Phase 4 close-out (fetchability as a label proxy 0.135 vs 0.886; fingerprint
rotation and the weaker run predicate; 22% determinism with the response
cache as what makes numbers reproducible; scope ending at `phase4-D`,
ceiling 0.0347). The same test asserts each item is present.

### P7-3 — Documentation map and links (blocking)

`README.md` links `docs/adversarial.md`, `docs/point-in-time.md`,
`docs/splits-eval-audit.md`, `docs/WAIVERS.md`,
`docs/phase3-preregistration.md`, the model card, and `production-gaps.md`.
Every relative Markdown link in `README.md` and `docs/` resolves on disk;
the link check is part of `tests/test_phase7_docs.py`.

### P7-4 — Architecture diagram (blocking)

A committed diagram (`docs/architecture.md`, ASCII or mermaid) of the
serving path: extension → `/predict` → Tier-1 fast path → band gate →
Tier-2 (sealed or live: fetcher, detector, LLM) → disposition. Annotations
are measured values only, each citing its report: Tier-1 p50 0.45 ms
in-process / ~7 ms HTTP; LLM p50 1308 ms and $0.367/1k escalated at Groq
listed rates (2026-09-18); out-of-band rows never call Tier 2.

### P7-5 — Demo GIF (blocking, operator-run)

≤ 60 s, three scenarios in order: a safe site (`allow`, labeled "not a
safety guarantee"), a phishing page (`alert`), and an injection page
(detector escalation). On-screen values must equal
`reports/phase6-demo.json`; the recording runs against the container with
the sealed Tier-2 default. The GIF hash and the transcript it was checked
against are recorded in `reports/phase7.md`. Until it lands, the phase does
not close.

### P7-6 — Authorship (blocking)

The README byline, the copyright year(s), and `pyproject.toml` authors
agree; the license is MIT. The roadmap's "still open" note is resolved by a
recorded decision (name and year range), not by omission.

### P7-7 — Closing gate (blocking)

On the closing commit: `ci`, `repro`, and `eval` workflows green; no data or
secret added; `reports/phase7.md` written; `roadmap.md` Phase 7 marked done;
tag `phase-7-close`.

---

## 3. Registered decisions

1. **No new numbers.** Packaging copies figures from existing reports and
   names the source; nothing is re-measured, re-swept, or re-called.
2. **`docs/adversarial.md` is a pointer page**, not a duplicate: it links
   `reports/phase5-adversarial.md`, the Phase 5 prereg, and the model card's
   Tier-2 section, so each number has one home.
3. **The GIF is the only operator-run artifact.** It is checked against the
   frozen transcript rather than trusted from memory.
4. **The webflow.io coverage gap is stated, never hidden** (README results
   table, model card, `production-gaps.md` §8). Packaging is not a fix.
5. **The README rewrite is re-verified, not repeated** (P7-1 audits it).
6. **The frozen system is not touched.** No code path, threshold, prompt,
   schema, or model artifact changes in this phase; the only code added is
   documentation tests.

---

## 4. Order of work

1. Commit this document on `phase-7`.
2. P7-1/P7-2/P7-3: consistency pass, model-card additions,
   `docs/adversarial.md`, `tests/test_phase7_docs.py`.
3. P7-4: architecture diagram.
4. P7-5: record the GIF; freeze transcript + hash.
5. P7-6: authorship decision.
6. P7-7: `reports/phase7.md`, roadmap update, tag `phase-7-close`.

---

## 5. Amendments (registered before the numbers)

(none yet)
