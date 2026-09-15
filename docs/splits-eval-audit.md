# splits-eval leakage audit (`suspicious`, kept)

`data/splits-eval/manifest.json` records `shape_only_roc_auc: 0.7530`,
verdict `suspicious`: benign mean path depth 1.75 vs phishing 1.03,
benign mean URL length 50.54 vs 45.84. The builder permits this band with
a warning — only `LEAKING` (> 0.85) halts — and this file is why no
rebuild was ordered to "fix" it.

## It is not a regression

The frozen Phase 1 split audits in the same band (`0.7564`,
`suspicious`, 1.84 vs 1.03). A depth-stratified rebuild chasing 0.5
would destroy the successor's byte-pinned identity (`repro/hashes.json`)
to fix a number the frozen baseline shares.

## Per-feature breakdown (splits-eval test, 6,386 rows)

| shape feature | single-feature AUC | benign mean | phish mean |
|---|---|---|---|
| path_depth | 0.306 (inverted) | 1.75 | 1.03 |
| len(path) | 0.287 (inverted) | 24.38 | 12.35 |
| len(netloc) | 0.700 | 14.29 | 22.01 |
| is_https | 0.397 | 97.6% | 77.0% |
| len(url) | 0.441 | 50.54 | 45.84 |

The 0.753 total is a multi-feature residual — inverted depth/path plus
phishing-side netloc length plus a pre-existing https gap — not one knob.
The withdrawn 0.60 single-threshold gate could not see this structure;
the validator-side mechanism gates (`validate_cc_benign.py`: scheme gap,
two-sided depth, length inversion) replace it for future corpora.

## Rules this decision rests on

*   Frozen and successor populations are immutable measurement
    instruments (hash-pinned, CRLF-canonical). Never rebalance data to a
    metric.
*   Splits are built by temporal + domain-hash rule only — no per-URL
    subsampling, no rebalancing (`build_splits.py` refuses `LEAKING`,
    warns otherwise; `test_eval.py::test_no_builder_shape_halt_below_leaking`
    pins it).
*   The successor test reuses benign domains from frozen train files: valid
    only for models never trained on these splits.

## Forward pointers

*   Calibration maps do not transfer across this cut (Step 5 finding):
    recalibrate on deployment-era data; see `ml_training/calibrate_gbm.py`.
*   Fixed thresholds transfer approximately, not exactly (Step 5): size
    operating budgets on achieved numbers with margin.
*   A three-band population (train < T1 < calib < T2 < test) would give
    calibration an era-appropriate held-out slice; it needs its own
    directory and hashes, leaving this one untouched.
