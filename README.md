# PhishNet - Phishing URL detection, measured honestly

[![ci](https://github.com/Bucke200/PhishNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Bucke200/PhishNet/actions/workflows/ci.yml)
[![repro](https://github.com/Bucke200/PhishNet/actions/workflows/repro.yml/badge.svg)](https://github.com/Bucke200/PhishNet/actions/workflows/repro.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](README.md#license)

**Built by Srinjay Panja**

## What this is

PhishNet is a phishing-URL detector with a browser extension, a FastAPI
serving container, and — more importantly — an evaluation pipeline that
records what works, what does not, and what it refuses to claim.

The champion is a single LightGBM over lexical URL features plus one hosting
signal (`is_hosted_tenant`), trained on a temporal, domain-disjoint split. It
does **not** use WHOIS/age at serving (ineligible: it failed its own
contamination gate), and it is **not** retrained between phases — every number
below is attached to the same frozen weights (`ablation_lexical_gbm_model.pkl`,
SHA256-pinned in `src/phishnet/model_manifest.json`).

Serving is a two-stage cascade: Tier 1 scores every URL string; in-band rows
(`0.6493 <= score < 0.9269`) go to Tier 2 (a hardened LLM page judgment plus a
frozen injection detector), and any Tier-2 failure **alerts** (fail closed).

### Results at a glance (fixed thresholds, never swept)

| what | number | note |
|---|---|---|
| Recall @ 0.5% FPR (`t05`) | **50.4%** (achieved FPR 0.40%) | indistinguishable from the budget |
| Recall @ 1.0% FPR (`t10`) | **60.7%** (achieved FPR 0.98%) | indistinguishable |
| Cold start, row (b), age forced unknown | **53.4%** | losing age costs ~25pp vs the age-known mix |
| Tier-1 serving latency p50 | **0.45 ms** in-process / ~7 ms HTTP | criterion 12 met (was 14.3 ms) |
| Phase 4 LLM layer | **unanswered** (bounded) | structural ceiling +0.0347 recall at most |
| Webflow-class hosted phishing | **missed** | recorded, retrain work (`docs/production-gaps.md` §8) |

Swept-on-test numbers appear only where labeled unattainable. The protocol is
`docs/phase3-preregistration.md`; results are `reports/phase3.md`,
`reports/phase5-adversarial.md`, and `reports/phase6.md`; the model card is
`docs/model-card.md`.

---

## Using it

The extension talks to a local serving container. There is no hosted demo
backend: the earlier Render deployment served the retired hard-vote model and
was removed in Phase 6.

1. **Start the container** (sealed demo mode — offline, replays the registered
   Phase 5 verdicts):
    ```bash
    docker build -f backend/Dockerfile -t phishnet-serving .
    docker run --rm -p 8000:8000 phishnet-serving
    ```
2. **Load the extension:** `chrome://extensions` → enable Developer mode →
   **Load unpacked** → select the `extension/` folder.
3. **Browse.** The notification shows the disposition (`alert` / `allow` /
   `can't assess`), the Tier-1 score, and the top SHAP features. `allow` means
   "no alert at the calibrated operating point" — not a safety guarantee.

For live Tier-2 (real page fetch + LLM judgment) instead of the sealed replay,
see *Deploying Your Own Backend* below.

---

This project consists of three main components:
1.  **Serving (`src/phishnet/serving/`):** FastAPI app over the frozen row (a)
    LightGBM, with the Tier-1 → Tier-2 cascade, shortener resolution, and
    native SHAP.
2.  **Machine Learning:** the training, ablation, and calibration scripts in
    `ml_training/`, plus the evaluation harness (`eval.py`, `predictors.py`).
    The champion is the Phase 3 row (a) ablation; datasets are not shipped.
3.  **Browser Extension (`extension/`):** calls the container and renders the
    disposition, score, and top SHAP contributions.

## Features

*   Real-time URL analysis via the browser extension, with native LightGBM SHAP
    explanations.
*   Two-stage cascade: a calibrated lexical champion, then an LLM page judgment
    with a frozen injection detector for in-band rows only.
*   Fail-closed Tier-2: a schema/refusal/timeout/unfetchable outcome alerts
    rather than silently retaining a score.
*   Shortener resolution: follows redirects and scores the final URL;
    unresolved → "can't assess", never a verdict score.
*   Reproducible: SHA256-pinned weights, frozen thresholds, sealed run stores,
    and a golden dataset-identity test suite.

## Tech stack

**Machine learning**

| Tool | Role |
|---|---|
| LightGBM 4.6 | The served champion (Phase 3 row (a)): one gradient-boosted tree model over 79 URL features; native tree-SHAP via `pred_contrib` |
| scikit-learn 1.6 | Legacy 4-model hard-vote ensemble (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting), calibration (isotonic/sigmoid), metrics |
| pandas / NumPy | Feature tables, splits, report rendering |
| tldextract 5.1 | Registrable-domain parsing on a **pinned public-suffix snapshot** (no network at runtime) |
| python-Levenshtein | Brand-distance features (`*_min_distance`) |
| pyarrow | Parquet reader for the Common Crawl wave fetch |

**Serving and API**

| Tool | Role |
|---|---|
| FastAPI + Uvicorn | `GET /health`, `POST /predict`, `POST /explain` |
| Pydantic v2 | Request/response validation (and URL normalization) |
| Docker / Docker Compose | Slim Tier-1 image + a **separate Playwright fetcher image**, so latency is not measured beside a browser |
| LightGBM native SHAP | Per-feature contributions in the `/explain` response |

**LLM layer (Tier 2, in-band rows only)**

| Tool | Role |
|---|---|
| Groq API — `openai/gpt-oss-120b` | Strict JSON-schema page judgment (`temperature 0`, `seed 0`, `reasoning_effort low`) |
| Playwright (Chromium) | Renders the page in the fetcher image (`backend/fetcher/Dockerfile`) |
| BeautifulSoup 4 | HTML → frozen canonical extract (title, visible text, form fields, link hosts) |

**Data pipeline**

| Tool | Role |
|---|---|
| PhishTank `online-valid` + OpenPhish | Phishing feeds (temporal labels; takedown-filtered before collection) |
| Tranco 46VQX 1M | Benign seed domains (pinned, citable list ID) |
| Common Crawl `CC-MAIN-2026-34` (fallback `-30`) | Benign deep-link corpus |
| **AWS Athena + Amazon S3** | Per-domain queries against the Common Crawl columnar index at `s3://commoncrawl/cc-index/table/cc-main/warc/`; hosted-tenant queries write to an S3 results bucket |
| boto3 | Athena client (fetch-only, imported lazily so the locked runtime stays minimal) |
| RDAP / WHOIS | Domain-age enrichment (point-in-time; ineligible for the headline) |
| crt.sh | Certificate-transparency history (dropped unmeasured, Amendment E) |

**Tooling and CI**

| Tool | Role |
|---|---|
| uv + hatchling | Locked dependency resolution and packaging |
| ruff + mypy | Lint/format and strict typing |
| pytest | 470+ tests, including golden dataset-identity and serving-identity gates |
| GitHub Actions | `ci`, `repro`, `eval`, `collect`, `phase4-forward` workflows |

> **Not used:** AWS Glue. The Common Crawl index is queried directly with
> Athena over the public S3 table; no Glue crawler/catalog is involved.

## Data pipeline

```text
PhishTank / OpenPhish ─┐
                       ├─► data/raw/*.jsonl (append-only daily snapshots)
Tranco 46VQX ──────────┘        │
                                ▼
Common Crawl index ──► AWS Athena (S3) ──► build_cc_benign.py ──► validate_cc_benign.py
 (CC-MAIN-2026-34)      per-domain query      fetch + select        hard mechanism gates
                                │
                                ▼
                        build_splits.py ──► data/splits-p3/{train,calib,test}.csv
                        (temporal phish,        │  + leakage / shape audit
                         domain-hash benign)    ▼
                                        RDAP/WHOIS enrichment (point-in-time)
                                                │
                                                ▼
                        ml_training/train_ablation.py ──► backend/ablation_lexical_assets/
                                                │
                                                ▼
                        eval.py + predictors.py ──► reports/*.json|md
                                                │
                                                ▼
                        src/phishnet/serving (FastAPI) ──► extension/
```

| Stage | Entry point | Notes |
|---|---|---|
| Collect | `collect.py` (`collect.yml`) | PhishTank needs an app key; OpenPhish is key-free; Tranco via API key |
| Benign corpus | `build_cc_benign.py` | Common Crawl columnar index via Athena (`--source columnar`, default) or CDX probe; Parquet wave path (`fetch_cc_wave.py`) |
| Validate | `validate_cc_benign.py` | Scheme-gap, path-depth, length-inversion, overlap gates; never trains |
| Split | `build_splits.py` | Temporal cutoff for phishing, registrable-domain hash for benign; no per-URL random splitting |
| Enrich | `src/phishnet/enrichment/` | RDAP age + point-in-time join; CT dropped |
| Train | `ml_training/train_ablation.py` | Phase 3 row (a)/(b) ablations; `train_gbm.py`, `calibrate_gbm.py` for the Phase 2 lineage |
| Evaluate | `eval.py`, `predictors.py` | Fixed report; any `.score(urls)` predictor |
| Serve | `src/phishnet/serving/` | Tier-1 → Tier-2 cascade, shortener resolution, fail-closed |
| Harden | `src/phishnet/adversarial/`, `scripts/p5_*.py` | Injection detector + adversarial evaluation (Phase 5) |

> The Common Crawl benign acquisition is **implemented and offline-tested but
> the final fetch has not been run** (no `data/raw/benign-cc-*` artifact).
> See `docs/cc-benign-acquisition.md`.

## Project Structure

```
PhishNet/
├── .github/workflows/          # ci.yml (pytest + mypy), collect.yml (daily feeds), eval.yml (PR gate)
├── backend/                    # Deployment layout
│   ├── urlset_ml_assets/       # (Ignored) Fetched/generated model assets
│   ├── cc_ml_assets/           # (Ignored) CC-ensemble retraining output
│   ├── gbm_assets/             # (Ignored) Single-GBM output (Phase 2 lineage)
│   ├── gbm_iso_assets/         # (Ignored) Isotonic run: wrapper + refit base + sidecar
│   ├── gbm_sig_assets/         # (Ignored) Sigmoid run (same layout)
│   ├── Dockerfile              # Tier-1 serving image (uv-based; build from repo root)
│   └── fetcher/Dockerfile      # Separate Playwright fetcher image (live Tier 2)
├── dist/                       # (Ignored) Build output
├── extension/                  # Browser extension files
│   ├── icons/                  # Extension icons
│   ├── background.js           # Extension logic
│   └── manifest.json           # Extension configuration
├── img/                        # Demo screenshots
├── ml_training/                # Scripts for ML model training
│   ├── __init__.py             # Package marker (importable in tests)
│   ├── preprocess_urlset.py    # Preprocessing script for urlset.csv
│   ├── train_urlset.py         # Training script for the URLSet model
│   ├── train_cc_split.py       # Same arch, retrained on data/splits-cc
│   ├── train_gbm.py            # One-variable swap: single LightGBM, no scaler
│   └── calibrate_gbm.py        # Domain-hash carve + isotonic/sigmoid + threshold knob
├── docs/                       # preregistrations (phases 3–6), model-card, production-gaps, audits
├── repro/                      # hashes.json + verify.py + check_golden.py (successor identity)
├── src/phishnet/               # Canonical packaged app
│   ├── features/extraction.py  # Canonical feature extractor
│   ├── serving/                # Phase 6: Tier-1 fast path, cascade, shortener, app
│   ├── fetcher/                # Separate Playwright fetcher service (live Tier 2)
│   ├── llm/                    # Prompts + response schemas (p4-v1/p5-h1/p6-v1) + Groq client
│   ├── snapshot/               # Fetch/extract/join, eval-mode Tier-1 reference
│   ├── enrichment/             # RDAP age + point-in-time join (CT dropped)
│   ├── adversarial/            # Phase 5 injection detector + lexical transforms
│   ├── verified_download.py    # SHA256-verified model-artifact downloader
│   ├── model_manifest.json     # Artifact names, URLs, and hashes
│   └── urlset_ml_assets/       # (Ignored *.pkl) Runtime model assets + README
├── tests/                      # pytest suite (features, serving identity, cascade, schema, dataset identity, …)
├── collect.py                  # Phase 1: append-only daily feed snapshot -> data/raw/
├── build_splits.py             # Phase 1: temporal + domain-disjoint splits -> data/splits/
├── build_cc_benign.py          # Common-Crawl benign corpus (columnar/Athena primary, CDX probe)
├── validate_cc_benign.py       # Corpus + trial-split gates (scheme/depth/length, advisory band)
├── eval.py                     # Phase 1: harness, any .score(urls) predictor in, fixed report out
├── predictors.py               # Ensemble adapters + soft votes + CC/GBM/calibrated predictors + SHAP
├── test_eval.py                # Phase 1: tests for silent metric edge cases
├── Makefile                    # Phase 1 targets: report/collect/split/baseline/eval/canary/test
├── data/                       # (Ignored) raw log (raw/) + generated splits (splits/)
├── reports/                    # Generated evaluation reports (<tag>.json + <tag>.md)
├── .dockerignore               # Root-context Docker ignores
├── docker-compose.yml          # Live two-layer stack (fetcher + serving)
├── pyproject.toml              # Exact deps; pytest/ruff/mypy config
├── uv.lock                     # Locked dependency set (CI uses --locked)
├── README.md                   # This file
└── update.md                   # Local status notes
```

**Note:** Virtual environments, datasets, model binaries, and secrets are not stored in the repository (see `.gitignore`). Model files are fetched at deploy time — see Model Artifacts below.

## Model Artifacts (Not in Git)

Runtime model files are deployment artifacts and are never committed
(git-ignored; GitHub also caps files at 100 MB). The **served** artifacts are
the Phase 3 row (a) pair:

*   `ablation_lexical_gbm_model.pkl` — the champion weights
*   `ablation_lexical_feature_columns.pkl` — the pinned 79-column vocabulary

The legacy `urlset_ensemble_model.pkl` / `scaler.pkl` / `feature_columns.pkl`
remain in the manifest only for the frozen Phase 1–2 eval path; they are no
longer served.

*   **Run/deploy:** fetch and SHA256-verify from the `models-v1` GitHub
    Release (sources and hashes pinned in `src/phishnet/model_manifest.json`):
    ```bash
    uv sync
    uv run python -m phishnet.verified_download
    # Tier-1 image fetches just the two row (a) artifacts:
    uv run python -m phishnet.verified_download \
      --only ablation_lexical_gbm_model.pkl \
      --only ablation_lexical_feature_columns.pkl
    ```
    The loader resolves the asset directory via `$PHISHNET_ML_ASSETS_DIR` (the
    Docker image sets it to `/app/models`), falling back to
    `src/phishnet/urlset_ml_assets/`.
*   **Retrain:** see Training the Model below.

## Setup Instructions

> **Note:** There is no hosted backend. You need a local serving container
> (see *Using it*) for the extension to work. MongoDB is not used anywhere —
> the `/report` feedback write path was removed in Phase 6.

### Development (tests, types, CI)

Requires Python `>=3.10` and `uv`. From the repository root:

```bash
uv sync --locked   # locked deps, incl. dev tools
uv run pytest      # test suite (tests/)
uv run mypy src tests ml_training   # strict type-check
```

GitHub Actions runs the same three steps on every push and pull request (`.github/workflows/ci.yml`).

### Deploying Your Own Backend

Build the image from the repository root (a `backend/`-only context cannot see the root dependency files):

```bash
docker build -f backend/Dockerfile -t phishnet-serving .
```

The container fetches the two verified row (a) artifacts on start (see Model Artifacts above) and serves `phishnet.serving.app` on port 8000 (`/health`, `/predict`, `/explain`). The legacy `phishnet.api` hard-vote pipeline and its MongoDB `/report` endpoint were removed in Phase 6.

Tier 1 scores any URL string. Tier 2 (in-band rows only, `0.6493 <= score < 0.9269`) has two modes:

```bash
# Sealed (default): replays the registered Phase 5 verdicts; live pages that
# are not in the demo set return "can't assess" / tier2_no_verdict. Offline,
# no key. This is what the recorded demo uses.
docker run --rm -p 8000:8000 phishnet-serving

# Live (both layers): Playwright fetcher + Groq judgment, one command.
# Put GROQ_API_KEY in .env first (see backend/.env.example).
docker compose up --build
```

Out-of-band rows never call Tier 2, so ordinary browsing costs nothing beyond the local score. `PHISHNET_TIER2_MODE=live` is fail-loud: if the key or fetcher URL is missing, the container refuses to start rather than silently serving "can't assess" for every in-band URL.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PHISHNET_TIER2_MODE` | `sealed` | Tier-2 provider: `sealed` (offline replay), `live` (fetch + Groq), `disabled` |
| `PHISHNET_TIER2_FLOOR` | `0.6493` | Score at/above which Tier 2 runs (testing knob; registered value is `lower_edge`) |
| `PHISHNET_FETCHER_URL` | — | Fetcher endpoint for live mode (e.g. `http://fetcher:8100/fetch`) |
| `GROQ_API_KEY` | — | Required by live mode; also read from `.env` |
| `PHISHNET_EXTENSION_ID` | pinned ID | CORS allowlist for the browser extension (set in the image) |
| `PHISHNET_ML_ASSETS_DIR` | packaged `urlset_ml_assets/` | Where the verified model artifacts live (`/app/models` in the image) |
| `PHISHNET_THRESHOLDS_FILE` | `reports/phase4.json` | Threshold source, verified at startup |
| `PHISHNET_MODELS_BASE_URL` | release URL | Mirror for `verified_download` (SHA256 still enforced) |
| `TLDEXTRACT_CACHE` | `.tld_cache` | Pinned public-suffix snapshot location |
| `TRANCO_API_KEY`, `TRANCO_ACCOUNT_EMAIL` | — | Tranco list resolution in `collect.py` |
| `PHISHTANK_KEY` | — | PhishTank feed pull in `collect.yml` |

## Training the Model (Optional)

> The **served champion** (Phase 3 row (a)) is trained by
> `ml_training/train_ablation.py` on the `data/splits-p3` bands, not by the
> legacy urlset scripts below. This section documents the legacy Phase 1 path;
> retraining the champion is a separate, gated workflow (`build_splits.py
> --phase3`, then the ablation trainer).

Retraining the legacy ensemble needs a dataset the repo does not ship: put a
`urlset.csv` with `domain` and `label` columns at `data/urlset.csv` (`data/`
is git-ignored).

1.  Install dependencies from the repository root (`pyproject.toml` + `uv.lock` are the source of truth, no separate virtual-environment setup needed):
    ```bash
    uv sync
    ```
2.  From the repository root, preprocess then train (paths used by the scripts, e.g. `data/...` and `backend/...`, are resolved relative to the repository root, so run them from there rather than from `ml_training/`):
    ```bash
    uv run python ml_training/preprocess_urlset.py
    uv run python ml_training/train_urlset.py
    ```
    *   Preprocessing (single canonical extractor via `phishnet.features.extraction`) writes `processed_data.pkl`, `scaler.pkl`, and `feature_columns.pkl` to `backend/urlset_ml_assets/` (see the `*_FILE` constants at the top of each script).
    *   Training loads those files and writes `urlset_ensemble_model.pkl` alongside them (it runs on import, so plain `python` execution is enough).
3.  Point the app at the fresh assets with `$PHISHNET_ML_ASSETS_DIR` (e.g. `backend/urlset_ml_assets/`) or restart the deployed backend to pick them up.

### CC / GBM / calibration lineage (research, same frozen vocabulary)

All three keep the canonical extractor and the frozen 78-column vocabulary;
each changes exactly one thing:

```bash
uv run python ml_training/train_cc_split.py   # same ensemble arch, CC population -> backend/cc_ml_assets/
uv run python ml_training/train_gbm.py        # single LGBMClassifier, no scaler (native units) -> backend/gbm_assets/
uv run python ml_training/calibrate_gbm.py [--method isotonic|sigmoid] [--target-fpr 0.005]
# domain-hash carve of train only (fresh seed, test never read), refit base +
# prefit calibrator + threshold knob -> backend/gbm_{iso,sig}_assets/ + calibration-report.json
```

The served champion is the Phase 3 row (a) ablation (lexical +
`is_hosted_tenant`), not the `models-v1` ensemble and not the Phase 2
`gbm_refit` lineage. Those remain in the repo as evaluated history; see
*Phase 3 headline* and *Honest operating points*.

### Prediction API

```bash
# score + disposition (Tier 1 always; Tier 2 only for in-band rows)
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/login"}'

# native LightGBM SHAP for the scoring model (the old 501 is gone)
curl -X POST localhost:8000/explain -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/login", "top_k": 5}'

curl localhost:8000/health
```

`/predict` returns `disposition` (`alert` / `allow` / `can't assess`),
`score` (the verdict score, `null` when unresolved), `tier1_score`, `in_band`,
`reason`, `tier2_mode` (`sealed` / `live` / `disabled`), `tier2` (the judgment
or failure), and the `model_hash` / `thresholds_source` provenance.
`/explain` returns top-k native tree-SHAP `{feature, contribution}` plus the
`bias` term, from the model that scored. `/health` reports the pinned hashes,
thresholds, and Tier-2 mode.

---

## Evaluation Harness (Phase 1)

Everything after this phase is judged by one command:

```bash
make report        # rebuild splits from the raw log, re-run the frozen baseline
```

### Layout

| File | What it does |
|---|---|
| `collect.py` | Append-only daily snapshot of phishing feeds and Tranco deep links → `data/raw/` |
| `build_splits.py` | Temporal + domain-disjoint splits, campaign capping, leakage audit → `data/splits/` |
| `eval.py` | The harness. Any `.score(urls)` predictor in, one fixed report out → `reports/` |
| `predictors.py` | Ensemble adapters + canaries + soft-vote / CC / GBM / calibrated predictors + native SHAP |
| `test_eval.py` | Tests for the metric edge cases that fail silently |
| `ml_training/train_cc_split.py` | Same ensemble arch, retrained on `data/splits-cc` → `backend/cc_ml_assets/` |
| `ml_training/train_gbm.py` | One-variable swap: single `LGBMClassifier`, no scaler → `backend/gbm_assets/` |
| `ml_training/calibrate_gbm.py` | Domain-hash carve of train only + isotonic/sigmoid (prefit) + `--target-fpr` threshold knob → `backend/gbm_{iso,sig}_assets/` |
| `Makefile` | `report/collect/split/baseline/eval/canary/test/clean` targets |
| `.github/workflows/collect.yml` | Daily cron snapshotting feeds into `data/raw/` (start on day 1) |
| `.github/workflows/eval.yml` | PR gate evaluating the candidate model against `reports/baseline.json` |

### Order of operations

**Day 1 — start the cron before anything else.** The OpenPhish community feed has
no timestamps; it is a snapshot of what is live right now. The first-observed date
comes from the collection log (`.github/workflows/collect.yml`). PhishTank's
`online-valid` dump carries `submission_time`, so it backfills real dates
immediately — request an app key on day 1 (approval is not instant).

> Pushing anything under `.github/workflows/` needs a token with the `workflow`
> scope. If the file silently does not appear in the repo, that is why.

**Days 1–2 — benign side.** Pin a Tranco list ID from <https://tranco-list.eu>
(the permanent ID, not "top 1M as of today"). `collect.py --benign` crawls each
domain's homepage for same-registrable-domain internal links (subdomains
included), then `/sitemap.xml`, then a few linked pages for depth-2 links, and
keeps at most one homepage per domain. The homepage row records the final URL
after redirects (HTTPS with an HTTP fallback) — never an assumed
`https://{domain}/` — so the URL scheme reflects what was actually reached.
Domains that deny robots, fail to fetch, or serve non-HTML yield no rows
rather than a fabricated homepage. Every benign row carries `seed_domain`,
`crawl_status`, `http_status`, `link_depth`, and the Tranco provenance fields.
The scheduled crawl covers 1200 domains (`--per-domain 12`); scale the domain
target — not per-URL sampling — if the phishing-to-benign ratio needs to move,
and watch the leakage audit rather than the raw counts.

**Days 3–4 — splits and audit.** `make split` prints the shrinkage at every stage
and ends with the leakage audit. Treat a `LEAKING` verdict as a hard stop.

**Day 5 — freeze the baseline.** `make baseline` writes `reports/baseline.json`.
Commit it. It will be a bad number. That is the point — it is the denominator for
every later claim.

**Days 6–7 — wire the gate.** `make canary` should show `random` landing near the
base rate and `url_shape_canary` doing poorly. If the canary does well, go back to
day 3.

### Methodology (frozen, do not change)

*   **Split by class.** The negative class is time-invariant by construction:
    benign URLs are collected contemporaneously, so temporal splitting is
    applied to phishing positives while benign negatives are deterministically
    partitioned by registrable-domain hash. Phishing cutoff `--split-date`
    (else now minus `--test-days`) applies to positives only; each benign
    registrable domain goes wholly to train or test via
    `sha256("<neg-hash-seed>:<domain>")` mapped to `[0, 1)` against
    `--benign-test-fraction` (default `0.2`, seed
    `phishnet-neg-split-v1`, both recorded in the manifest). No per-URL random
    splitting, no rebalancing.
*   **PR-AUC as the headline** (`average_precision_score`, not trapezoid AUC).
*   **Recall at FPR ≤ 0.5%**, with the threshold reported (walk real score values;
    ties respected, no ROC interpolation). The harness threshold is swept on
    test and therefore unattainable live; deployable numbers fix the threshold
    on held-out data first (see Honest operating points below).
*   **Recall (TPR) at FPR ≤ 0.1%**, reported the same way: the threshold walks
    real score values (never interpolated), alongside the actually achieved
    FPR and whether the 0.1% budget was hit exactly — the empirical ROC is
    discrete, so attainment is reported, never implied.
*   **Shape gates: builder halts only on `LEAKING`.** The drafted 0.60
    single-threshold builder gate was proposed and **withdrawn** (it vetoed
    the project's own successor while missing inverted signals); only a
    `LEAKING` verdict (> 0.85) refuses to write, `suspicious` builds with a
    warning. Acceptance for new corpora lives in `validate_cc_benign.py` as
    mechanism hard gates (scheme rate gap ≤ 0.04, two-sided path-depth
    |AUC − 0.5| ≤ 0.05, URL-length inversion ≥ 0) plus a warn-only 0.70
    shape band. Frozen splits predate all of this (their recorded audit
    values stand; see `docs/splits-eval-audit.md` and `docs/WAIVERS.md`).
*   **Precision at deployment prevalence** (default 1e-4) + false warnings per
    10,000 URLs browsed.
*   **Bootstrap CIs resampled by registrable domain**, not by row.
*   **Canonical CRLF dataset bytes.** The builder writes splits with a pinned
    CRLF lineterminator and `.gitattributes` checks out CRLF on every
    platform, so file hashes (notably the `baseline.json` dataset identity)
    are byte-identical on Windows and Linux. Never "normalize" these files.
*   **Within-dataset deltas only.** A `--compare` baseline from a different
    dataset sha256 gets its delta column suppressed with a stated reason
    (PR-AUC's no-skill floor is the base rate); same-dataset comparisons
    name both predictor contracts.
*   **Asset identity.** Reports record model + columns + scaler-presence
    hashes; same predictor name with different assets is flagged, never
    silently compared — including scaler added/removed with the vocabulary
    unchanged.
*   **Two degeneracy warnings.** Fewer than 10 distinct scores (step
    functions), and separately a top-score tie exceeding the FP budget
    (score piles, e.g. isotonic ties at 1.0, which the first check cannot
    see) — both collapse the operating point, for different reasons.
*   **Reliability diagrams.** Every report with scores in [0, 1] renders
    calibration bins plus an ASCII reliability strip (x = mean score,
    o = empirical rate).
*   **Straddling domains dropped from test, not train**; campaign cap (5 URLs per
    domain in test); pinned public suffix list with the source recorded in the
    manifest; distinct-score-count check (flags predictors with < 10 levels —
    the hard-voting `VotingClassifier` baseline is measured via member vote
    fractions and is expected to trip this flag).
    **Invariant: train/test share zero eTLD+1** (public-suffix-aware grouping,
    never raw-host comparison — `login.example.com` and `www.example.com`
    are one domain). The builder drops any registrable domain seen on both
    sides from test while preserving phishing temporal purity; golden tests
    recompute eTLD+1 from URLs so a stale column cannot hide leakage.

### Evaluation power and the successor population

FPR ≤ 0.5% is measured in false-positive *events*. The harness warns whenever
fewer than 20 FPs fit the budget (i.e. fewer than ~4,000 benign URLs in test):

| test set | benign | FP budget `⌊0.005·n⌋` | FPR resolution `1/n` |
|---|---|---|---|
| `data/splits/test.csv` (Phase 1, frozen) | 462 | 2 | 0.22% |
| `data/splits-large/test.csv` (Phase 2 enlarged) | 1,668 | 8 | 0.06% |
| `data/splits-eval/test.csv` (successor) | 4,321 | 21 | 0.02% |

The default `--benign-test-fraction 0.2` is sized for *training* splits (most
benign stays in train). The Phase 2 comparison never retrains, so parking
~18,000 benign URLs in train starves the measurement. The successor rebuilds
the enlarged deep-link corpus with `--benign-test-fraction 0.5` at the Phase 2
cutoff into its own dir (`make eval-split`), leaving `data/splits/`,
`data/splits-large/`, and `reports/baseline.json` byte-identical. Same hash
seed, so its benign test domains are a superset of the Phase 2 ones; the audit
reads `suspicious` (ROC ~0.753, same band as the frozen Phase 1 split), which
the builder permits with a warning — only `LEAKING` halts. Because the
successor test reuses benign domains that sit in the frozen *train* files, it
is valid only for models never trained on these splits (true of the frozen
`models-v1`); never train on `train.csv` and report on the successor test.

### Honest operating points (Phase 2 history; fixed thresholds, not test sweeps)

Phase 2's champion was the refit lineage (`gbm_refit`): the 80%-fit GBM
whose threshold, calibration slice, and evaluation are all mutually held
out. It was superseded by the Phase 3 row (a) ablation (below). The
full-train GBM (`gbm_single`, PR 0.9261 vs 0.9236 — noise) is a
non-shipped intermediate: no held-out slice exists for its threshold, so
it has no deployable operating point. Stated first, because everything
below explains it.

A threshold swept on test (the harness default) is the best point
*found*, which no live system can do. A deployable number fixes the
threshold on held-out data first — here a 6,529-row calibration slice
carved from `splits-cc/train.csv` by registrable-domain hash (fresh
seed, zero domain overlap, test never read) — then applies it to
`splits-eval/test.csv` (2,065 phishing / 4,321 benign; FPR ≤ 0.5% =
21 FPs). Same budget, same test, every row below:

| predictor | threshold (fixed, pre-registered) | recall | FPR | FP vs 21 |
|---|---|---|---|---|
| full-train GBM, T from calib slice it trained on (not shipped) | 0.800596 | 80.58% | 4.61% | **199 - 9.5x over budget** |
| **champion `gbm_refit`, 0.5%-aimed** | 0.937037 | 59.47% | 0.65% | 28 - over budget |
| **champion `gbm_refit`, 0.3%-aimed (margin)** | 0.937990 | 58.93% | 0.60% | 26 - over budget |
| champion + sigmoid, 0.3%-aimed | 0.978391 | 59.47% | 0.65% | 28 (same cell as 0.5%-aimed: monotonic rescaling) |
| champion + isotonic | 0.984043 | 58.93% | 0.60% | 26 (same cell as 0.3%-aimed) |

Acceptance verdict: **unmet**. No pre-registered threshold achieves
FPR ≤ 0.5% on this population; the nearest honest row is 58.93% @
0.60%. The 51.28% @ 0.44% quoted earlier belonged to the deleted scaled
weights - superseded, stated plainly. Aiming lower still (0.2%? 0.1%?)
until a threshold lands inside would be fitting the margin to test, so
the firewall stands: the margin needs its own distribution (further
populations) or the three-band design, not another peek.

For contrast, the test-swept (unattainable) points: 51.67% / 51.72% /
51.72% / 0.00% - isotonic's swept point collapsed outright (ties at 1.0
exceed the budget, so the walk exits above 1.0; the harness now names
this degeneracy class explicitly). Four readings:

*   The full-train row is honest about a broken procedure, not a
    deployable point: a threshold picked where the model trained buys
    overfit separability and blows the budget ~10x on test-era data.
    Thresholds must come from data the model never saw - which is why
    the champion is the refit, not the full train.
*   Both aimed thresholds missed (28 and 26 vs 21): threshold transfer
    error (~0.1-0.15pp FPR) swamps a 21-event budget. This is the drift
    finding showing up in threshold transfer, after calibration levels
    and operating cells. Phase 4 sizes the band on achieved numbers
    with a margin drawn from the transfer-error distribution - aiming
    0.3% to land 0.5% cost 0.5pp recall and still missed by 5 events -
    and reports the shortfall (0.10pp here).
*   The sigmoid map preserves the base ranking exactly (PR 0.9236 =
    uncalibrated, as monotonicity demands) while isotonic's steps cost
    -0.023 PR - but *both* land at Brier ~0.19 on test, so the level
    failure is era drift, not calibrator capacity. The ranking model is
    stable across the temporal cut; the calibration map is not, and must
    be refit on recent data (see `ml_training/calibrate_gbm.py --train`).
    Features are native units throughout (the scaler was dropped; the
    ablation re-ran to confirm the no-op: PR -0.001).
*   Phase 4 escalation bands must be sized on fixed-threshold rows, never
    swept ones - and band membership shifts under calibration, so the
    band is defined in calibrated-score space or re-derived after it.

### Phase 3 headline (fixed thresholds on the calib band, three-band population)

On `data/splits-p3` (train 44,285 / calib 13,157 / test 24,819; each
row fixed on its own calib scores — row (a) 0.926936 @0.5%,
0.878084 @1%; row (b) 0.917390 @0.5%, 0.865044 @1% — judged by the
wider-interval rule): the lexical baseline with `is_hosted_tenant`
reads **50.4% recall at 0.40% FPR (indistinguishable, [0.29%,
0.53%])**. Domain age is ineligible for the headline (test-band
unknown gap 0.059 > 0.05); paired lift +0.22–0.34 at each row's own
operating point (+0.28–0.33 conditional on age-known rows, a harder
mix). Certificate history was dropped unmeasured (Amendment E). Cold
start: losing age costs ~25pp recall (78→53%) and triples FPR — the
number that sizes the Phase 4 band. Transfer concludes nothing
either way (intervals straddle / no 1% comparator). Tier-1 serving
p50 is **0.45 ms** (Phase 6 met criterion 12; the earlier 14.3 ms
attributed the cost to the extractor, but it was per-call pandas frame
construction plus the sklearn wrapper — see `reports/phase6.md`). Full
table first in `reports/phase3.md`; model card in
`docs/model-card.md`; protocol in `docs/phase3-preregistration.md`;
roadmap in `docs/roadmap.md` and the acceptance-criteria table in `reports/phase3.md` §6 (criteria text formerly `docs/plan.md` §2.2, removed after the roadmap superseded it).

### Collection provenance (`source`)

`source` correlates with the label by construction — every feed is
single-class (`phishtank`/`openphish` always phishing, `tranco:*` always
benign) — and a source-oracle would score 1.0 on any split. That number is
meaningless as a leakage signal, which is why the methodology does not use
it. Instead:

*   `source` is provenance metadata, never a model input. Predictors
    implement `.score(urls)` on URL strings only, through 78 URL-derived
    feature columns; `eval.py` passes `df["url"]` and reads `source` solely
    for per-source slice reporting. A regression test pins the column
    vocabulary against provenance-derived names.
*   The leakage audit and the `url_shape_canary` detect the *consequence* of
    provenance that actually matters: systematic URL-shape differences
    induced by collection (e.g. homepage-heavy benign vs deep phishing
    paths). The audit is source-blind — it sees lengths, depths, and query
    counts, never feed labels — so it measures URL/content separability, and
    the per-source slices show where it concentrates.
*   `tranco:VALIDATION-POOL` (66 head-of-Tranco domains, ranks 1–90, crawled
    with the same pipeline ~10 minutes before the main run to validate the
    crawler; 52 seeds overlap the main pool and deduplicate to earliest) is
    retained, not removed: it is deployment-relevant head traffic, it is
    reported as its own slice in every eval report, and removing it would
    rewrite the frozen splits while cutting Phase 1 benign test data by
    ~13% where power is scarcest (59 of 462).

### Reproducing the evaluation populations

Prerequisites: a fresh clone, Python 3.13, `uv sync --locked` (pinned
pandas/scikit-learn/tldextract). No model artifacts are needed to rebuild
splits. Dataset bytes are canonical CRLF (`.gitattributes` checks out CRLF
on every platform; the builder pins CRLF output), so hashes below reproduce
on Windows and Linux.

```bash
# Phase 1 identity (frozen): must print 385aa409c222
python -m pytest -m golden --strict-markers -p no:cacheprovider -q
python repro/check_golden.py

# Successor population: rebuild into a scratch dir and verify pinned hashes
make eval-split OUT=$RUNNER_TEMP/repro
python repro/verify.py --hashes repro/hashes.json --dir $RUNNER_TEMP/repro
```

`make eval-split` stages the manifest-exact input set (both benign-2026-09-13
snapshots + both phishing feeds), rebuilds with the pinned cutoff and
`--benign-test-fraction 0.5`, and verifies `train.csv`/`test.csv`/
`manifest.json` against `repro/hashes.json`. The volatile run timestamp
lives in the `run-meta.json` sidecar (intentionally unpinned), so the three
pinned files diff byte-cleanly. `repro/check_golden.py` fails loudly if the
golden marker set ever collects empty (e.g. after moving test files).

### Adding a predictor

```python
class MyModel:
    name = "lgbm-v3"

    def score(self, urls: list[str]) -> list[float]:
        return self.model.predict_proba(featurise(urls))[:, 1].tolist()
```

```bash
make eval PRED=mymodule:MyModel
```

---

## Documentation map

| doc | what it covers |
|---|---|
| `docs/phase3-preregistration.md` | the Phase 3 protocol (the clearest evidence of how the work was run) |
| `docs/phase4-preregistration.md` | the LLM layer design (unanswered, bounded) |
| `docs/phase5-preregistration.md` | the adversarial-hardening protocol |
| `docs/phase6-preregistration.md` | the serving/demo protocol and amendments A–F |
| `docs/model-card.md` | intended use, leaks, cold start, calibration shelf life |
| `docs/adversarial.md` | Phase 5 pointer page (results in `reports/phase5-adversarial.md`) |
| `docs/architecture.md` | serving topology with measured latency/cost annotations |
| `docs/production-gaps.md` | measured gaps and future work (§7 withdrawn, §8 webflow.io) |
| `docs/point-in-time.md` | point-in-time feature discipline |
| `docs/splits-eval-audit.md`, `docs/WAIVERS.md` | shape audit and unregenerable populations |
| `reports/phase3.md`, `reports/phase5-adversarial.md`, `reports/phase6.md` | results |
| `docs/roadmap.md` | phase history and future work |

---

## Demo

Screenshots below predate the Phase 6 extension (which now shows the
disposition, score, and top SHAP contributions):

**Phishing detected (extension warning):**

![Phishing Detected](img/phishing.png)

**Phishing but Chrome Secure can't detect:**

![Phishing but Chrome Secure can't detect](img/phishing%20but%20chrome%20secure%20cant%20detect.png)

**Safe site detected:**

![Safe Site Detected](img/safe.png)

---

## License

MIT License  

Copyright 2025–2026 Srinjay Panja

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Notes on Large Files

- This repository does **not** store model/data files or virtual environments. These are ignored via `.gitignore` (see Model Artifacts above for how to obtain them).
- `.gitattributes` still declares LFS filters for `*.pkl` and similar, but nothing in the repo uses LFS: binaries ship as GitHub Release assets, never via git.
