# Local LLM implementation plan (product track)

Status: **planning document — no code changes, not an amendment.**
Supersedes the `docs/roadmap.md` "Hosted API, no local model" deviation
(Ollama dropped for a hardware constraint) for Phase 7+ work. It does **not**
change Phase 4/5, whose registered model remains `openai/gpt-oss-120b` on
Groq (`docs/phase4-preregistration.md` `phase4-A`,
`docs/phase5-preregistration.md` §1).

The point of going local is not "a smarter judge". It is, in priority order:

1. **Grounding** (brand/domain, RDAP age, cert org) — the measured FPR-budget
   risk is a grounding problem, not a language problem
   (`docs/production-gaps.md` §4).
2. **Determinism** — fixed weights + greedy decoding replace the response
   cache and remove the 22% disagreement problem
   (`docs/model-card.md` "Determinism 11/50").
3. **Offline feature generation → Tier-1 retrain** — the roadmap's
   "joined feature" option, made affordable by zero per-call cost
   (`docs/roadmap.md` §Phase 4, option 2).
4. Fine-tuning, last, because the tier's structural ceiling is
   **+0.0347 recall** (`docs/model-card.md`) and no model lifts it.

---

## 1. Measured hardware baseline

| item | value | notes |
|---|---|---|
| GPU | NVIDIA RTX 3050 Laptop, **4,096 MiB** (3,717 free), compute 8.6 | Ampere; CUDA 12.x builds fine |
| iGPU | AMD Radeon (512 MB shared) | ignore for LLM |
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 threads @ 3.3 GHz | AVX2, no AVX-512 |
| RAM | 15.4 GB total (~13–14 GB usable) | shared with OS and the app |
| Disk | 187.6 GB free | sufficient |
| OS | Windows 11 Pro for Workstations, build 26100 | `llama.cpp` prebuilt CUDA binaries |
| Python | 3.13.2 (repo venv) | no `torch`/`transformers`/`llama_cpp` installed |
| Driver | 616.92 | supports current CUDA runtime |

### Feasibility verdict

| model class | Q4_K_M weights | fits 4 GB fully? | generation (projected) | per call* | 3,318-call pass |
|---|---|---:|---:|---:|---:|
| 1.5B–2B | ~1.0–1.5 GB | **yes** | 60–90 tok/s | 5–8 s | 5–8 h |
| 3B–4B | ~2.0–2.7 GB | **yes** (4k ctx) | 40–70 tok/s | 6–11 s | 6–10 h |
| 7B | ~4.4 GB | no (hybrid `-ngl` ~60%) | 15–25 tok/s | 16–28 s | 15–26 h |
| 7B (CPU only) | ~4.4 GB | n/a | 4–6 tok/s | 70–110 s | 65–100 h |
| 14B | ~9 GB | no | ~2–3 tok/s CPU | 3–5 min | weeks |

\* per call assumes the measured extract size: ~1,432 prompt + ~245 completion
tokens (`runs/phase4/cache` means). All speed numbers are projections; §7 L1
measures them and gates the model choice.

**Conclusion:** the only model class that supports bulk sweeps on this
machine is **3B–4B Q4 fully offloaded** (or 1.5B for iteration). 7B is
usable for *serving* (4.46% escalation rate, `docs/model-card.md`) but not
for a 3,318-call recorded pass. Fine-tuning larger than ~3B requires renting
a GPU (§6).

---

## 2. Where the LLM sits in the system (target architecture)

```text
URL ──► Tier 1 (LightGBM, 0.45 ms) ──► out of band: allow/alert
                                        │ in band (~4.5%)
                                        ▼
                          Tier 2 (local LLM, grounded)          ── serving path
                                        │
                       offline: same model over train-band extracts
                                        ▼
                       semantic features ──► retrain Tier 1    ── joined-feature path
```

The joined-feature path is the highest-value use: it converts the LLM's
semantic signal into a calibrated, fast, deterministic Tier-1 feature and
removes per-request LLM failure modes at serving. The local model makes the
offline generation affordable (no quota, no per-call cost).

---

## 3. Lane A — deterministic local inference

### 3.1 Stack

- **Runtime:** `llama.cpp` **pinned release** (Windows CUDA 12 build),
  invoked as `llama-server` so the repo keeps its HTTP `judge()` shape and
  needs no Python ML dependency.
- **Artifacts:** GGUF models pinned by SHA256 using the existing
  `src/phishnet/verified_download.py` discipline (releases + hash check, no
  unpinned pulls).
- **Models (pick one per lane, record hash + build commit in the asset
  fingerprint):**
  - bulk/iteration: `Qwen3-4B-Instruct` or `Qwen2.5-3B-Instruct` Q4_K_M
  - quality candidate: `Qwen2.5-7B-Instruct` Q4_K_M (hybrid offload)
  - fine-tune base: `Qwen2.5-1.5B/3B` (local QLoRA) or 14B (rented, §6)

### 3.2 Determinism protocol

- Greedy decode: `--temp 0 --top-k 1 --top-p 1 --seed 0`, fixed `-c 4096`.
- Pin `llama.cpp` build commit + model hash + quant level; a change is a new
  fingerprint, not an upgrade.
- Measure, never assume: 50 extracts × 2 cold runs, bar **≤ 5%**, target
  **0%** (the hosted route measured 22%). If GPU nondeterminism appears,
  first try `--no-kv-offload`/single-sequence, then CPU-only as fallback.
- Passing L1 means the recorded headline can be a **point estimate**, i.e.
  1,106 calls instead of 3,318.

### 3.3 Schema and fail-closed behavior

- Constrain decoding with the existing `p6-v1` schema (widened
  `credential_types`; `docs/production-gaps.md` §1). Prefer the server's
  `response_format: json_schema`; fallback `--grammar-file` (GBNF generated
  from `strict_response_format`).
- A schema failure maps to the Phase 4 §2 policy **but fail-closed**: never
  "retain Tier-1" as the only path. For a phishing page that is a silent
  fail-open; route to alert or human review (`production-gaps.md` §1
  remediation 1).

### 3.4 Repo integration

| item | path |
|---|---|
| client | `src/phishnet/llm/local_client.py` — same `judge()` signature/return as `client.py`, so `CascadePredictor` and `p4_sweep.py` work unchanged |
| config | `PHISHNET_LLM_BACKEND=local`, `PHISHNET_LOCAL_URL`, `PHISHNET_LOCAL_MODEL`, `PHISHNET_LOCAL_NGL`, `PHISHNET_LOCAL_TIMEOUT` |
| prompt | freeze a grounded prompt as `p7-g1` (Lane B); one prompt version per run |
| run store | `runs/phase7-local/<run-id>/` + `judgments.jsonl` (same schema) |
| budget | `budget.py` still tracks tokens; local cost is `$0`, cap may stay unset |
| tests | schema conformance, determinism bar, fail-closed mapping, `judge()` contract parity |

---

## 4. Lane B — grounding (do this first; model-agnostic)

1. **Brand→domain allowlist** (curated, cited): canonical host per protected
   brand; seeded from Tranco head + the extract corpus' `imitated_brand`
   values (`docs/production-gaps.md` §4 remediation 1).
2. **RDAP age + cert org** joined into the prompt as a grounding block
   (the Phase 3 enrichment code already exists; RDAP is point-in-time).
3. Prompt shape: keep the extract block unchanged; add a delimited
   `grounding:` block; new prompt version → new frozen hash.
4. **Acceptance:** in-band benign FPR on the `samehost_login` set drops from
   the measured **20% (3/15)** (`production-gaps.md` §4) toward the 5% bar;
   adversarial reach unchanged.

Grounding is cheap, model-independent, and attacks the one LLM defect that
can consume the entire 0.5% cascade FPR budget.

---

## 5. Lane C — offline feature generation → Tier-1 retrain

- Generate the LLM fields (`asks_for_credentials`, `credential_types`,
  `imitated_brand`, `identity_domain_match`, `urgency_pressure`) for the
  **train band** only; hash-verify no test/calib row is included.
- Add them as a feature group to `ml_training`; retrain row (a); evaluate on
  the frozen splits with the existing `eval.py` harness and the Phase 3
  gates.
- Expected effect is measured, not assumed: the hosted judge's recall ceiling
  is +0.0347, so the bar is a **cascade FPR improvement on in-band benign**
  and no recall regression — not a headline jump.
- Hardware: per 1,000 calls ≈ **1.7–3 h** (3B GPU). Budget the train-band
  size before committing; if it exceeds a few thousand rows, do it in
  overnight chunks (the cache makes it resumable).

---

## 6. Lane D — fine-tuning (last)

### 6.1 Data

- Positive: fetched-ok phishing extracts. Negative: fetched-ok benign, with
  **hard negatives** = `samehost_login` pages (§4).
- Adversarial: Phase 5 injected extracts (`reports/phase5-adversarial.md`)
  so hardening is not regressed.
- Splits: reuse `build_splits.py`; **no test/calib row may appear in
  training**; record dataset hashes.

### 6.2 Method and VRAM budget

| base | method | VRAM needed | on this machine |
|---|---|---:|---|
| 1.5B | QLoRA 4-bit, seq 1,024, batch 1, grad-ckpt | ~3 GB | **yes** |
| 3B | QLoRA 4-bit, seq 1,024, batch 1, paged 8-bit AdamW, CPU optimizer offload | ~4–6 GB | **tight**; expect slow/offload |
| 7B | QLoRA 4-bit | ~8–10 GB | **no** → rent |
| 14B | QLoRA 4-bit | ~16 GB | **no** → rent |
| any | full fine-tune | ≫ | no |

Rented lane: one 24 GB GPU (4090/A10/L4) for a few hours at roughly
$1–3/h is cheaper than days of local iteration.

### 6.3 Acceptance gates (all on frozen data)

| gate | threshold |
|---|---|
| G1 determinism | ≤ 5% disagreement (target 0) on 50×2 |
| G2 schema | 100% valid JSON (grammar-constrained) |
| G3 in-band benign FPR | ≤ 5% (hosted measured 20%) |
| G4 adversarial | evasion ≤ hosted escalate arm on the Phase 5 set |
| G5 cascade | no recall regression vs Tier-1 on the frozen eval |
| G6 contamination | dataset hash check proves no test rows |
| G7 scope | claims limited to the pinned model+quant+prompt |

---

## 7. Staged plan

| stage | work | hardware | exit gate |
|---|---|---|---|
| **L0** | install pinned `llama.cpp` CUDA build; SHA256-pin 3B and 7B GGUFs | GPU | `llama-server` responds; hashes recorded |
| **L1** | latency + determinism harness on 50 extracts | GPU | measured s/call; G1 |
| **L2** | grounded prompt (`p7-g1`); in-band login test | GPU | G3 |
| **L3** | offline feature gen on train band; Tier-1 retrain | GPU, overnight | cascade no-regression + FPR gain |
| **L4** | QLoRA 1.5B/3B local (or 7B/14B rented) | GPU / rented | G2, G4, G6 |
| **L5** | adversarial suite + close-out; new prereg + amendment | GPU | G4, G5, G7 |

No stage may publish a number without its gate; each new model/prompt/quant
is a new amendment, exactly as `phase4-A` was for the provider substitution.

---

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| 4 GB VRAM limits model size | 3B Q4 for bulk; 7B only for serving; rent for ≥7B tuning |
| GPU nondeterminism | greedy + fixed build; test; CPU fallback for the recorded pass |
| quant loss vs hosted 120B | measure on the same cascade metrics; the ceiling is +0.0347, so the bar is FPR, not recall |
| fine-tune regression on attacks | include Phase 5 adversarial extracts; rerun the suite |
| test contamination | split hashes in CI, same discipline as `test_dataset_identity.py` |
| build drift breaks reproducibility | pin `llama.cpp` commit + GGUF hash; a change is a new fingerprint |
| model over-refuses security content | check refusals on harvest pages during L1; prefer non-safety-tuned instruct models |

---

## 9. Hardware upgrade thresholds

| VRAM | unlocks |
|---|---|
| 4 GB (now) | 3B Q4 full offload; 1.5B/3B QLoRA |
| 8–12 GB | 7B Q4 full offload; 7B QLoRA; 3B at 8k ctx |
| 16–24 GB | 14B Q4; 14B QLoRA; 7B full fine-tune |
| 48 GB+ | 32B Q4; serious sweeps into the feature path |

## 10. Explicit non-goals

- Not a Phase 4/5 substitution: those phases stay on `gpt-oss-120b`.
- Not a serving-time replacement for Tier 1: the joined-feature path puts the
  signal in the GBM; the LLM stays a narrow in-band adjudicator.
- Not a fix for fetchability, host reputation, hosted-platform coverage, or
  the shortener corpus leak — those are the measured bounders
  (`docs/production-gaps.md` §2, §3, §8) and remain the highest-recall work.
