# Runtime model assets (not stored in this repository)

The model files loaded by `phishnet.serving.tier1` live in this directory
when the package is deployed:

- `ablation_lexical_gbm_model.pkl` (Phase 3 row (a) weights)
- `ablation_lexical_feature_columns.pkl` (the pinned 79-column vocabulary)

The legacy urlset ensemble artifacts (`urlset_ensemble_model.pkl`,
`scaler.pkl`, `feature_columns.pkl`) are no longer served; they remain in the
manifest only for the frozen Phase 1–2 eval path.

These files are deployment artifacts, not source files: they are fetched at
deploy time (see `phishnet.verified_download`, verified against
`model_manifest.json` from the `models-v1` GitHub Release) or supplied by the
operator, and are intentionally not committed. The serving loader locates
this directory without depending on the current working directory — via the
`PHISHNET_ML_ASSETS_DIR` environment variable when set, otherwise alongside
the installed `phishnet` package.
