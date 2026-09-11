# Runtime model assets (not stored in this repository)

The phishing-detection model files loaded by `phishnet.api` live in this
directory when the package is deployed:

- `urlset_ensemble_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

These files are deployment artifacts, not source files: they are fetched at
deploy time (see `backend/download_models.py`) or supplied by the operator,
and are intentionally not committed. `phishnet.api` locates this directory
without depending on the current working directory — via the
`PHISHNET_ML_ASSETS_DIR` environment variable when set, otherwise alongside
the installed `phishnet` package.
