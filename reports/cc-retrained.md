# PhishNet eval — `cc_retrained(hard-vote)`

2026-09-15T08:44:21+00:00 · git `7710c230` · dataset `test.csv` sha256 `71e511028e81`

3,417 URLs · 1,943 phishing (56.9%) · 1,946 registrable domains · 2026-07-10 → 2026-09-13

## Read this first

- Only 1474 negatives, so an FPR of 0.500% is 7 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.
- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function — read the operating point only, and fix the predictor to emit probabilities.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.8891 | [0.8709, 0.9097] | -0.0615 |
| ROC-AUC | 0.9097 | — | +0.0383 |
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

Brier 0.1161 · ECE 0.1121 · MCE 0.2091

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.000] | 684 | 0.000 | 0.029 | -0.029 |
| 1 [0.000–0.750] | 684 | 0.431 | 0.222 | +0.209 |
| 2 [0.750–1.000] | 683 | 0.915 | 0.772 | +0.143 |
| 3 [1.000–1.000] | 683 | 1.000 | 0.895 | +0.105 |
| 4 [1.000–1.000] | 683 | 1.000 | 0.927 | +0.073 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,396 | 657 | 0.00% | 0.00% | 0.8227 |
| info | 251 | 250 | 0.00% | 0.00% | 1.0000 |
| click | 118 | 118 | 0.00% | — | — |
| shop | 115 | 115 | 0.00% | — | — |
| org | 114 | 14 | 0.00% | 0.00% | 0.3019 |
| sbs | 108 | 108 | 0.00% | — | — |
| lol | 107 | 107 | 0.00% | — | — |
| top | 89 | 89 | 0.00% | — | — |
| net | 88 | 20 | 0.00% | 0.00% | 0.7285 |
| ru | 67 | 2 | 0.00% | 0.00% | 0.0462 |
| de | 63 | 4 | 0.00% | 0.00% | 0.8000 |
| cfd | 53 | 53 | 0.00% | — | — |
| pro | 50 | 46 | 0.00% | 0.00% | 0.9342 |
| edu ⚠︎ | 32 | 0 | — | 0.00% | — |
| fr ⚠︎ | 28 | 9 | 0.00% | 0.00% | 0.7944 |
| xyz ⚠︎ | 27 | 27 | 0.00% | — | — |
| gov ⚠︎ | 25 | 0 | — | 0.00% | — |
| io ⚠︎ | 24 | 2 | 0.00% | 0.00% | 0.1465 |
| lat ⚠︎ | 24 | 24 | 0.00% | — | — |
| help ⚠︎ | 22 | 22 | 0.00% | — | — |
| pl ⚠︎ | 18 | 5 | 0.00% | 0.00% | 1.0000 |
| cn ⚠︎ | 16 | 7 | 0.00% | 0.00% | 1.0000 |
| it ⚠︎ | 16 | 0 | — | 0.00% | — |
| live ⚠︎ | 16 | 4 | 0.00% | 0.00% | 0.3636 |
| site ⚠︎ | 16 | 16 | 0.00% | — | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 1,605 | 1,027 | 0.00% | 0.00% | 0.9205 |
| <30 | 1,054 | 573 | 0.00% | 0.00% | 0.8335 |
| 60-99 | 515 | 249 | 0.00% | 0.00% | 0.9156 |
| 100-199 | 201 | 79 | 0.00% | 0.00% | 0.8306 |
| >=200 ⚠︎ | 42 | 15 | 0.00% | 0.00% | 0.9159 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,815 | 1,815 | 0.00% | — | — |
| cc:CC-MAIN-2026-34 | 1,460 | 0 | — | 0.00% | — |
| openphish | 128 | 128 | 0.00% | — | — |
| cc:CC-MAIN-2026-30 ⚠︎ | 14 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 37.3 ms · p90 38.3 ms · p99 39.9 ms · max 47.9 ms

Batch throughput: 3,469 URLs/s
