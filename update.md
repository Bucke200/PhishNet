# PhishNet — Actual State (verified 2026-09-12)

Source of truth: the working tree, local `pytest`/`mypy` runs, and the
GitHub Actions API. Nothing below is claimed from plans alone.
(`update.md` itself is git-ignored; it is a local-only note.)

## Completed (verified)

- **CI workflow** — `.github/workflows/ci.yml` committed (`45b26192`) and
  pushed on `chore/repo-hardening`: checkout → Python 3.10 → uv →
  `uv sync --locked` → `uv run pytest` → `uv run mypy src tests ml_training`.
- **GitHub Actions green** — run #2 on `e43944a7`:
  `status: completed, conclusion: success`; every step (incl. `Run tests`
  and `Type-check with mypy`) `success`.
  Run #1 on `45b26192` failed only on mypy (21 errors in committed
  `ml_training/` legacy scripts); fixed by `e43944a7`, no checks weakened.
- **Legacy `ml_training/` type-clean (committed)** — `e43944a7` fixed the 21
  mypy errors with real annotations plus the standard
  `sklearn.* ignore_missing_imports` override (scikit-learn ships no stubs;
  first-party strictness unchanged). Reproduced remote-exact locally:
  `mypy`: no issues in 8 files; `pytest`: 13 passed.
- **Dead `backend/.dockerignore` removed** — the only image
  (`backend/Dockerfile`) builds from the repo root
  (`docker build -f backend/Dockerfile .`), so Docker only reads the root
  `.dockerignore`; no compose/render configs, CI Docker steps, scripts, or
  docs reference `backend/.dockerignore`. Root comment updated.
- **Dev dependencies pinned exact** — `mypy==2.3.1`, `ruff==0.16.7`
  (were `>=`), matching `uv.lock`; `pytest`/`pandas-stubs` already exact,
  all runtime deps already `==`. `uv lock` regen touched only the two
  specifier lines (no version/resolution change, still 64 packages);
  `uv sync --locked` passes after.

## Implemented and locally verified, now committed

- **Architecture consolidation (committed `f2f958aa`, pushed)** — legacy flat modules
  deleted (`backend/main.py`, `backend/feature_extraction.py`,
  `backend/download_models.py`, `ml_training/feature_extraction.py`);
  canonical package `src/phishnet/` (`api.py`, `features/extraction.py`,
  `verified_download.py`, `model_manifest.json`); reworked
  `ml_training/preprocess_urlset.py`, `backend/Dockerfile`, `pyproject.toml`.
  Remote CI run #3 on the full tree: completed/success.
- **Feature-extraction consolidation** — single canonical
  `phishnet.features.extraction`; `test_canonical_wiring.py` proves
  production, training, and tests share it (not mocked).
- **Local validation of full worktree** — `pytest`: 24 passed;
  `uv run mypy src tests ml_training`: no issues in 11 files;
  `git diff --check`: clean.
- These changes are committed (`f2f958aa`) and pushed; remote CI run #3
  on the full tree is green.

## Implemented but not fully verified

- **Docker/serving entrypoint** — `backend/Dockerfile` (root context,
  `uv sync --locked --no-dev`, `CMD` runs `verified_download` then
  `uvicorn phishnet.api`). The image has never been built in CI or
  evidenced locally; no Docker CI job exists.
- **Model-artifact pipeline** — manifest + SHA256-verified downloader from
  GitHub Release `models-v1`; real `*.pkl` files are git-ignored deployment
  artifacts. Wiring is unit-tested with synthetic payloads only.

## Not yet implemented / not verified

- **Real-model inference compatibility test** — no test loads the real
  `*.pkl` artifacts; compatibility is unproven.
- **Hardening commit + push** - done (`f2f958aa`); remote CI run #3 on the
  full tree is green.

## In progress

- None. This cleanup's validation is complete: `uv sync --locked` passes,
  `pytest` 24 passed, `mypy` clean on 11 files, `git diff --check` clean.
  Cleanup changes left uncommitted per commit-safety rule.

## Known advisories (not failures)

- `DeprecationWarning`: FastAPI `on_event` in `src/phishnet/api.py`
  (pre-existing, tests still pass).
- Actions annotation: Node.js 20 deprecation for
  `checkout`/`setup-python`/`setup-uv` (upstream warning only).

## Attribution

- Preserved as-is: `pyproject.toml` author `Srinjay Panja`, README
  “Built by Srinjay Panja” + `Copyright 2025 Srinjay Panja`, commits by
  srinjay / Srinjay Panja — internally consistent; username alone is not
  evidence for a change. No `LICENSE` file exists (license text in README).

## Session log (2026-09-12, branch `chore/repo-hardening`)

1. **CI audit (read-only)** — no `.github/` existed; derived required
   Python 3.10, `uv sync --locked`, `uv run pytest`,
   `uv run mypy src tests ml_training` from `pyproject.toml`/`uv.lock`/
   `README.md`/`backend/Dockerfile`. No files created or modified.
2. **CI creation** — wrote `.github/workflows/ci.yml` (checkout, Python
   3.10, uv, locked sync, pytest, mypy; `contents: read`; no secrets),
   validated (`Test-Path`, `git diff --check`), committed `45b26192`,
   pushed normally (no force). Unrelated worktree changes untouched.
3. **Local validation before push** — `mypy` failed (8 errors:
   untyped test helpers in `tests/test_verified_download.py`) and `pytest`
   failed to collect (`import ml_training` not on `sys.path` under pytest).
   Fixed genuinely, no weakening: added real annotations
   (`Literal[False]`, `Iterator[bytes]`, fixture types) and
   `pythonpath = ["."]` under `[tool.pytest.ini_options]`. Re-ran:
   mypy clean, 24 passed. Kept uncommitted at the time.
4. **Publish** — `45b26192` pushed; remote Actions run #1
   (`34646283732`) failed on mypy only (pytest passed).
5. **Remote diagnosis + fix** — logs need auth, so reproduced the exact
   committed tree in a scratch worktree: 21 mypy errors, all in legacy
   `ml_training/` (6× sklearn `import-untyped`, 5× missing annotations,
   10× int/float/str dict assignments). Fixed there (annotations +
   `sklearn.* ignore_missing_imports`), verified green
   (mypy 8 files, pytest 13 passed), then staged only those 3 blobs into
   the main index via `hash-object`/`update-index` — worktree bytes
   verified identical before/after. Committed `e43944a7`, pushed normally.
6. **Actions green** — run #2 (`34646961816`) on `e43944a7`:
   completed/success, all steps green. Scratch worktree removed.
7. **Cleanup (this session)** — dead `backend/.dockerignore` deleted,
   dev pins, attribution audit (preserved), this file rewritten.
   Nothing committed; all user worktree changes preserved.
