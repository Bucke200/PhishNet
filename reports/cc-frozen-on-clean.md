# PhishNet eval — `legacy_ensemble(models-v1)`

2026-09-15T08:43:47+00:00 · git `7710c230` · dataset `test.csv` sha256 `71e511028e81`

3,417 URLs · 1,943 phishing (56.9%) · 1,946 registrable domains · 2026-07-10 → 2026-09-13

## Read this first

- Only 1474 negatives, so an FPR of 0.500% is 7 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.
- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function — read the operating point only, and fix the predictor to emit probabilities.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.7064 | [0.6766, 0.7354] | -0.2442 |
| ROC-AUC | 0.7108 | — | -0.1606 |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 0.00%] | +0.00% |
| Achieved FPR | 0.00% | — | +0.00% |
| Recall @ FPR≤0.10% | 0.00% | — | — |
| Achieved FPR (0.10% budget) | 0.00% | — | — |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 1,943 · TN 1,474

Strict point: threshold **1.000000** — nearest attainable point at FPR 0.0000% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.2633 · ECE 0.2155 · MCE 0.2848

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.500] | 684 | 0.135 | 0.213 | -0.079 |
| 1 [0.500–0.750] | 684 | 0.688 | 0.475 | +0.213 |
| 2 [0.750–1.000] | 683 | 0.941 | 0.656 | +0.285 |
| 3 [1.000–1.000] | 683 | 1.000 | 0.728 | +0.272 |
| 4 [1.000–1.000] | 683 | 1.000 | 0.772 | +0.228 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,396 | 657 | 0.00% | 0.00% | 0.5919 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9955 |
| click | 118 | 118 | 0.00% | — | — |
| shop | 115 | 115 | 0.00% | — | — |
| org | 114 | 14 | 0.00% | 0.00% | 0.1889 |
| sbs | 108 | 108 | 0.00% | — | — |
| lol | 107 | 107 | 0.00% | — | — |
| top | 89 | 89 | 0.00% | — | — |
| net | 88 | 20 | 0.00% | 0.00% | 0.2413 |
| ru | 67 | 2 | 0.00% | 0.00% | 0.0335 |
| de | 63 | 4 | 0.00% | 0.00% | 0.2222 |
| cfd | 53 | 53 | 0.00% | — | — |
| pro | 50 | 46 | 0.00% | 0.00% | 0.9363 |
| edu ⚠︎ | 32 | 0 | — | 0.00% | — |
| fr ⚠︎ | 28 | 9 | 0.00% | 0.00% | 0.3825 |
| xyz ⚠︎ | 27 | 27 | 0.00% | — | — |
| gov ⚠︎ | 25 | 0 | — | 0.00% | — |
| io ⚠︎ | 24 | 2 | 0.00% | 0.00% | 0.0739 |
| lat ⚠︎ | 24 | 24 | 0.00% | — | — |
| help ⚠︎ | 22 | 22 | 0.00% | — | — |
| pl ⚠︎ | 18 | 5 | 0.00% | 0.00% | 0.3833 |
| cn ⚠︎ | 16 | 7 | 0.00% | 0.00% | 0.5263 |
| it ⚠︎ | 16 | 0 | — | 0.00% | — |
| live ⚠︎ | 16 | 4 | 0.00% | 0.00% | 0.5000 |
| site ⚠︎ | 16 | 16 | 0.00% | — | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 1,605 | 1,027 | 0.00% | 0.00% | 0.7983 |
| <30 | 1,054 | 573 | 0.00% | 0.00% | 0.6274 |
| 60-99 | 515 | 249 | 0.00% | 0.00% | 0.7241 |
| 100-199 | 201 | 79 | 0.00% | 0.00% | 0.5845 |
| >=200 ⚠︎ | 42 | 15 | 0.00% | 0.00% | 0.3886 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,815 | 1,815 | 0.00% | — | — |
| cc:CC-MAIN-2026-34 | 1,460 | 0 | — | 0.00% | — |
| openphish | 128 | 128 | 0.00% | — | — |
| cc:CC-MAIN-2026-30 ⚠︎ | 14 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 37.2 ms · p90 37.9 ms · p99 39.3 ms · max 43.6 ms

Batch throughput: 3,475 URLs/s
