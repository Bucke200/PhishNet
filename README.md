# PhishNet - Phishing Detection System

**Built by Srinjay Panja**

## Introduction

PhishNet is a system designed to detect phishing URLs in real-time. It utilizes a machine learning model (URLSet Ensemble) trained on URL characteristics, combined with a backend API and a browser extension for seamless integration. When you browse the web, the extension sends the current URL to the backend API, which uses the trained model to predict whether the URL is likely malicious (phishing) or legitimate.

---

## 🚀 Live Deployment & Easy Usage

**PhishNet is already deployed and ready to use!**

- The backend is live at: [https://phishnet-pavv.onrender.com](https://phishnet-pavv.onrender.com)
- Anyone can use the PhishNet browser extension from anywhere — no server setup required!

> **Note:** Because of the free Render plan, the backend will "spin down" after a period of inactivity. The first request after a period of inactivity can be delayed by 50 seconds or more while the server wakes up. Subsequent requests will be fast.

### How to Use
1. **Install the Extension:**
    - Download **just the `extension` folder** from this repository (no need to clone the entire repo) or click on this to download directly: [https://downgit.github.io/#/home?url=https://github.com/Bucke200/PhishNet/tree/master/extension](https://downgit.github.io/#/home?url=https://github.com/Bucke200/PhishNet/tree/master/extension).
    - Open your browser's extensions page (e.g., `chrome://extensions` for Chrome).
    - Enable Developer Mode.
    - Click "Load unpacked" and select the `extension` folder you downloaded.
2. **Browse the Web:**
    - The extension will automatically check URLs using the live backend.
    - You’ll see notifications if a site is flagged as phishing.

---

This project consists of three main components:
1.  **Backend:** A FastAPI application that serves the ML model predictions via an API endpoint.
2.  **Machine Learning (URLSet Ensemble):** A model trained using features extracted from the `urlset.csv` dataset (dataset not shipped with the repo; see Training the Model). The training scripts are included.
3.  **Browser Extension:** A simple browser extension that communicates with the backend API to check URLs as you visit them.

## Features

*   Real-time URL analysis via browser extension.
*   Phishing detection powered by an ensemble of 4 machine learning models: Random Forest, Logistic Regression, Decision Tree, and Gradient Boosting.
*   FastAPI backend for efficient API request handling.
*   Modular structure with separate components for the backend, ML training, and extension.
*   Includes scripts for data preprocessing and model retraining.

## Project Structure

```
PhishNet/
├── .github/workflows/          # ci.yml (pytest + mypy), collect.yml (daily feeds), eval.yml (PR gate)
├── backend/                    # Deployment layout
│   ├── urlset_ml_assets/       # (Ignored) Fetched/generated model assets
│   ├── .env.example            # Example environment file for MongoDB URI
│   └── Dockerfile              # Backend image (uv-based; build from repo root)
├── dist/                       # (Ignored) Build output
├── extension/                  # Browser extension files
│   ├── icons/                  # Extension icons
│   ├── background.js           # Extension logic
│   └── manifest.json           # Extension configuration
├── img/                        # Demo screenshots
├── ml_training/                # Scripts for ML model training
│   ├── __init__.py             # Package marker (importable in tests)
│   ├── preprocess_urlset.py    # Preprocessing script for urlset.csv
│   └── train_urlset.py         # Training script for the URLSet model
├── src/phishnet/               # Canonical packaged app
│   ├── features/extraction.py  # Canonical feature extractor
│   ├── api.py                  # FastAPI application
│   ├── verified_download.py    # SHA256-verified model-artifact downloader
│   ├── model_manifest.json     # Artifact names, URLs, and hashes
│   └── urlset_ml_assets/       # (Ignored *.pkl) Runtime model assets + README
├── tests/                      # pytest suite (features, wiring, downloads)
├── collect.py                  # Phase 1: append-only daily feed snapshot -> data/raw/
├── build_splits.py             # Phase 1: temporal + domain-disjoint splits -> data/splits/
├── eval.py                     # Phase 1: harness, any .score(urls) predictor in, fixed report out
├── predictors.py               # Phase 1: legacy ensemble adapter + leak canaries
├── test_eval.py                # Phase 1: tests for silent metric edge cases
├── Makefile                    # Phase 1 targets: report/collect/split/baseline/eval/canary/test
├── data/                       # (Ignored) raw log (raw/) + generated splits (splits/)
├── reports/                    # Generated evaluation reports (<tag>.json + <tag>.md)
├── .dockerignore               # Root-context Docker ignores
├── pyproject.toml              # Exact deps; pytest/ruff/mypy config
├── uv.lock                     # Locked dependency set (CI uses --locked)
├── README.md                   # This file
└── update.md                   # Local status notes
```

**Note:** Virtual environments, datasets, model binaries, and secrets are not stored in the repository (see `.gitignore`). Model files are fetched at deploy time — see Model Artifacts below.

## Model Artifacts (Not in Git)

Runtime model files (`urlset_ensemble_model.pkl`, `scaler.pkl`, `feature_columns.pkl`) are deployment artifacts and are never committed (git-ignored; GitHub also caps files at 100 MB).

*   **Run/deploy:** fetch and SHA256-verify them from the `models-v1` GitHub Release (sources and hashes are pinned in `src/phishnet/model_manifest.json`):
    ```bash
    uv sync
    uv run python -m phishnet.verified_download
    ```
    The backend resolves the asset directory via `$PHISHNET_ML_ASSETS_DIR` (the Docker image sets it to `/app/backend/urlset_ml_assets`), falling back to `src/phishnet/urlset_ml_assets/`.
*   **Retrain:** see Training the Model below; it writes fresh assets to `backend/urlset_ml_assets/`.

## Setup Instructions

> **Note:** Manual backend installation and local MongoDB setup are NOT required. The backend is already deployed and ready to use. Most users only need to install the extension as described above.

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
docker build -f backend/Dockerfile -t phishnet-backend .
```

The container fetches verified model artifacts on start (see Model Artifacts above) and serves `phishnet.api` on port 8000.

## Training the Model (Optional)

Retraining needs a dataset the repo does not ship: put a `urlset.csv` with `domain` and `label` columns at `data/urlset.csv` (`data/` is git-ignored).

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
| `predictors.py` | Legacy ensemble adapter + the canaries that check the dataset |
| `test_eval.py` | Tests for the metric edge cases that fail silently |
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
    ties respected, no ROC interpolation).
*   **Recall (TPR) at FPR ≤ 0.1%**, reported the same way: the threshold walks
    real score values (never interpolated), alongside the actually achieved
    FPR and whether the 0.1% budget was hit exactly — the empirical ROC is
    discrete, so attainment is reported, never implied.
*   **Shape-only acceptance gate: 0.60.** A train/test split is not usable if
    the URL-shape-only audit model separates its classes with ROC-AUC above
    `SHAPE_ONLY_ROC_AUC_GATE = 0.60` (`build_splits.py` refuses to write such
    a split and records threshold + pass/fail in the manifest). Committed
    before any dataset rebuild or retraining; applies to future split
    validation, not retroactively tuned to any observed result. Frozen
    splits predate the gate (their recorded audit values stand).
*   **Precision at deployment prevalence** (default 1e-4) + false warnings per
    10,000 URLs browsed.
*   **Bootstrap CIs resampled by registrable domain**, not by row.
*   **Canonical CRLF dataset bytes.** The builder writes splits with a pinned
    CRLF lineterminator and `.gitattributes` checks out CRLF on every
    platform, so file hashes (notably the `baseline.json` dataset identity)
    are byte-identical on Windows and Linux. Never "normalize" these files.
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

### Champion and honest operating points (fixed thresholds, not test sweeps)

The champion is the refit lineage (`gbm_refit`): the 80%-fit GBM whose
threshold, calibration slice, and evaluation are all mutually held out.
The full-train GBM (`gbm_single`, PR 0.9261 vs 0.9236 — noise) is a
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

## Demo

Below are screenshots demonstrating PhishNet in action:

**Phishing detected (extension warning):**

![Phishing Detected](img/phishing.png)

**Phishing but Chrome Secure can't detect:**

![Phishing but Chrome Secure can't detect](img/phishing%20but%20chrome%20secure%20cant%20detect.png)

**Safe site detected:**

![Safe Site Detected](img/safe.png)

---

## License

MIT License  

Copyright 2025 Srinjay Panja

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Notes on Large Files

- This repository does **not** store model/data files or virtual environments. These are ignored via `.gitignore` (see Model Artifacts above for how to obtain them).
- `.gitattributes` still declares LFS filters for `*.pkl` and similar, but nothing in the repo uses LFS: binaries ship as GitHub Release assets, never via git.
