# PhishNet Production Gaps (Phase 6 Inputs)

This document synthesizes the structural limitations, feature leaks, and architectural vulnerabilities identified across Phases 1 through 5. Each item represents a measured operational finding that bounds real-world deployment and defines the engineering requirements for Phase 6 (serving, demo, and production hardening).

---

## 1. Schema Fail-Open Defect (~12% on Phishing)

- **Empirical Finding:** Across all 564 Phase 5 calls (`openai/gpt-oss-120b`, 3 cold repeats), exactly 50 calls (8.9%) failed provider-side strict JSON schema validation (`status: 400`). The errors were concentrated entirely on phishing pages: **11.9% (15/126)** on clean phishing and **11.7% (35/300)** on injected phishing, against **0.0% (0/90)** on clean benign and **0.0% (0/48)** on framing. Exactly 47 of the 50 errors (94%) occurred because the strict schema's `credential_types` enum omitted `login` and `credentials`.
- **Cascade Consequence:** Under Phase 4 §2, calls failing validation were deterministically sealed with `verdict: None`, retaining their Tier-1 score. In the deployed cascade, retaining Tier-1 score means the page is **NOT LIFTED to alert**. Consequently, the cascade fails open without alerting on ~12% of credential-harvesting phishing pages, swamping prompt hardening gains at the cascade level.
- **Phase 6 Remediation:**
  1. *Immediate Fallback Rule:* Implement a fail-closed schema exception handler: if a Tier-2 call fails strict validation, route the page to human review or lift to alert if Tier-1 score exceeded a provisional threshold (e.g. $\ge 0.80$).
  2. *Schema Version Migration (`p6-v1`):* Widen the `credential_types` enum to include `login`, `credentials`, `generic_form`, and `session_token`, accompanied by automated unit tests validating enum coverage against production harvest pages.

---

## 2. Link Shortener Feature Leak (`is_shortened`)

- **Empirical Finding:** In the Phase 5 lexical arm (`phase5-F`), transforming 200 benign URLs with covered link shorteners (bit.ly, tinyurl.com, etc.) caused a **99.2% alert rate** at the fixed 0.5% FPR threshold (and 99.6% at 1.0% FPR), compared to a 0.5% baseline on clean benign URLs (`[+0.4110, +0.4740]` paired difference).
- **Cascade Consequence:** Tier 1 treats link shorteners indiscriminately as phishing indicators. This is an artifact of training data composition: shorteners in historical training sets were almost exclusively phishing links, creating a severe source-composition leak identical to the takedown leak.
- **Phase 6 Remediation (re-attributed by `reports/phase6.md`, `phase6-D`):**
  1. *Pre-scoring resolution — taken.* Follow redirects and score the final
     destination URL; unresolved → `can't assess`, never a score. The §0
     probe showed the flag carries only part of the leak (forcing
     `is_shortened` to 0 drops the alert rate from 0.987 to 0.595 at
     `t05`), so **stripping the flag is rejected**: it is also
     serve-time skew against a model trained with it.
  2. *Corpus re-balancing — future work.* Sampling benign shortened links
     into the Tier-1 training corpus is the only fix that removes the
     short-host/random-slug shape leak; it needs new weights and its own
     protocol (Phase 6 does not retrain).

---

## 3. Lexical Collapse on Unseen Hosts and Synthetic/Rare TLDs

- **Empirical Finding:** In the Phase 5 lexical host-swap control (`phase5-F`), swapping phishing URLs to random `swap-<6 alnum>.example` domains dropped Tier-1 recall from 54.5% to **3.0%** at $t_{0.5\%}$ (and open redirects collapsed to **0.0%**).
- **Cascade Consequence:** Tier 1 behaves primarily as a host and TLD reputation model rather than an intrinsic URL syntax classifier. Unseen hosts on novel, synthetic, or long-tail gTLDs receive low suspicion scores and fail to enter the escalation band (0.65 - 0.93), escaping detection entirely.
- **Phase 6 Remediation:**
  1. *Reputation-Independent Features:* Introduce structural token entropy, ngram character models, or subword embeddings that evaluate lexical randomness independently of known host memorization.
  2. *TLD Cold-Start Prior:* Assign non-zero prior suspicion to domains registered on newly delegated or high-abuse gTLDs when domain age is unknown.

---

## 4. In-Band Benign Login False Alarms (20.0%)

- **Empirical Finding:** Across all 6 Phase 5 held-out sweeps (both arms, 3 cold repeats), the clean benign false-alert rate was exactly **20.0% (3/15)**. All three false alerts occurred on `samehost_login` base pages (`clean-benign-login-northvale`, `clean-benign-login-parcelyn`, `clean-benign-login-tesserapay`).
- **Cascade Consequence:** The LLM judges ordinary password authentication forms as phishing when the apparent brand name does not resolve to an internationally recognized corporate host. In production, ~5% of benign traffic falls into the escalation band (0.65 - 0.93), and ~89% of that is fetchable. An in-band login false-alarm rate of 20% would consume the entire 0.5% cascade FPR budget on legitimate corporate authentication portals.
- **Phase 6 Remediation:**
  1. *Enterprise Domain Allowlist:* Maintain a verified internal domain/host directory that exempts authenticated corporate portals from Tier-2 phishing promotion.
  2. *Domain-Brand Consistency Grounding:* Ground LLM evaluation with domain registration age and SSL organization certificates to differentiate legitimate internal portals from external spoofing attempts.

---

## 5. Retain Policy is Unsafe (Forbidden from Deployment)

- **Empirical Finding:** Under the pre-registered Retain mapping (retaining Tier-1 score on detector hits), ordinary cascade evasion surged from 16.0% (17/106) to **62.3% (66/106)** across repeats (diff **-0.4623 [-0.6373, -0.2843]**), decisively failing Criterion 1.
- **Cascade Consequence:** Retain allows an attacker who includes detector-triggering keywords to suppress LLM evaluation and fail open to the sub-alert Tier-1 score, turning a security sensor into an evasion mechanism.
- **Phase 6 Constraint:** The Retain policy is formally prohibited from production serving. Detector hits must either escalate to alert or route to synchronous secondary verification; they must never suppress detection.

---

## 6. Detector Trivial Evadability and Framing False-Alarm Cost

- **Empirical Finding:**
  1. *Trivial Evadability:* The pure-function regex detector achieved 5/6 recall on isolated payloads and 64.3% (27/42) on page extracts (blind to `form_placeholder`, `link_dilution`, and `P-authority-1`). 14 of 14 adaptive aware candidates bypassed it on the very first author draft without iteration.
  2. *Framing Cost:* Under Escalate, the detector immediately lifted **50.0% (4/8)** of benign pages containing injection text to alert (descriptive lower bound), falsely penalizing benign pages discussing phishing or containing security text.
- **Cascade Consequence:** Static regex detection is incapable of stopping adaptive adversaries and introduces unacceptable false alarms on benign text.
- **Phase 6 Remediation:**
  1. Rely primarily on instruction-hierarchy prompt hardening (`p5-h1`) rather than keyword regex filters.
  2. If pre-LLM filtering is retained, replace regex matching with AST-based DOM sanitization and semantic classifiers trained to detect adversarial intent rather than keyword matches.

---

## 7. Tier-1 Serving Latency Overhead (Criterion 12) — WITHDRAWN, RESOLVED

- **Status (Phase 6, `phase6` / `reports/phase6.md`):** criterion 12 is met.
  Serving p50 is **0.45 ms** (n=300, in-process, seed 0), against the 14.3 ms
  it replaces.
- **Correction to this section's original finding:** the "~7.8 ms fixed
  per-call *extractor* overhead" named the wrong component. The extractor is
  0.29 ms; the fixed cost was per-call pandas DataFrame construction (~10 ms
  on the Phase 6 host) plus the sklearn `predict_proba` wrapper.
- **Remediation actually taken:** a pandas-free serving fast path (dict →
  preallocated row → `booster_.predict`), bit-equal to the Phase 3 headline
  scorer on every calib/test row. The Cython/Rust remedy below is
  **withdrawn**: a compiled extractor would buy nothing measurable.
- ~~Compile the lexical feature extraction routine in Cython, Rust, or C
  extensions.~~

---

## 8. Hosted-Platform Coverage Gap (`webflow.io`), measured live

- **Trigger:** a live phishing URL, `https://mtaskiellgeuin.webflow.io/`
  (a MetaMask-login page), served `allow` on 2026-09-20.
- **Measurement (frozen row (a) champion):** Tier-1 score **0.4468**, below
  `lower_edge` 0.6493, so Tier 2 is never invoked. The largest negative SHAP
  contribution is `is_hosted_tenant` (**-1.343**): `webflow.io` is absent
  from `HOSTED_PLATFORMS`, so the feature reads 0. The pinned PSL private
  section also does not carry `webflow.io`, so the list's stated curation
  rule did not catch it either.
- **Counterfactual (in-sample, clearly labeled):** with
  `is_hosted_tenant` set to 1, the same row scores **0.9832** (an outright
  alert). Across the **50 `webflow.io` phishing rows in the training band**
  (calib/test contain **zero**):

  | | alert ≥ `t05` | in band | below band |
  |---|---:|---:|---:|
  | as served | 5 | 22 | 23 |
  | `is_hosted_tenant = 1` | 48 | 1 | 1 |

- **Why this was not patched in Phase 6:** the 50 rows are **training** rows,
  so the counterfactual is in-sample and there is no held-out webflow
  population to validate against; and the weights were fit with the flag
  reading 0 on those rows, so flipping it at serving is the same
  training/serving mismatch this project rejected for `is_shortened`
  (§2). Changing the list now would be tuning after a failure.
- **Phase 7+ remediation (needs new weights / its own protocol):**
  1. extend `HOSTED_PLATFORMS` with a cited vendor-published source for
     `webflow.io` (and audit other free-site builders the same way), then
     retrain so the feature is fit under the extended list;
  2. the reputation-independent features already listed in §3 would attack
     the same miss from the lexical side and generalize to platforms not yet
     enumerated;
  3. until then, below-band dispositions are "no alert at the calibrated
     operating point", not a safety guarantee — the extension wording was
     corrected accordingly (`extension/background.js`).
