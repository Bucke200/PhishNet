# PhishNet eval — `legacy_ensemble(models-v1)`

2026-09-12T22:19:44+00:00 · git `5eed0cf1` · dataset `test.csv` sha256 `961efe5c9bfc`

3,738 URLs · 2,070 phishing (55.4%) · 1,963 registrable domains · 2026-08-22 → 2026-09-12

## Read this first

- Only 1668 negatives, so an FPR of 0.500% is 8 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.
- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function — read the operating point only, and fix the predictor to emit probabilities.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.8305 | [0.8057, 0.8524] | — |
| ROC-AUC | 0.8564 | — | — |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 0.00%] | — |
| Achieved FPR | 0.00% | — | — |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 2,070 · TN 1,668

## Calibration

Brier 0.1559 · ECE 0.1001 · MCE 0.1935

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.000] | 748 | 0.000 | 0.044 | -0.044 |
| 1 [0.000–0.750] | 748 | 0.354 | 0.316 | +0.038 |
| 2 [0.750–1.000] | 748 | 0.829 | 0.635 | +0.194 |
| 3 [1.000–1.000] | 747 | 1.000 | 0.811 | +0.189 |
| 4 [1.000–1.000] | 747 | 1.000 | 0.964 | +0.036 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,692 | 705 | 0.00% | 0.00% | 0.7062 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9955 |
| lol | 131 | 131 | 0.00% | — | — |
| org | 126 | 15 | 0.00% | 0.00% | 0.3325 |
| shop | 126 | 126 | 0.00% | — | — |
| click | 125 | 125 | 0.00% | — | — |
| sbs | 120 | 120 | 0.00% | — | — |
| top | 90 | 90 | 0.00% | — | — |
| net | 82 | 19 | 0.00% | 0.00% | 0.4907 |
| cfd | 60 | 60 | 0.00% | — | — |
| gov | 53 | 0 | — | 0.00% | — |
| pro | 53 | 53 | 0.00% | — | — |
| edu ⚠︎ | 40 | 0 | — | 0.00% | — |
| ru ⚠︎ | 38 | 2 | 0.00% | 0.00% | 0.1250 |
| de ⚠︎ | 32 | 5 | 0.00% | 0.00% | 0.8333 |
| fr ⚠︎ | 29 | 13 | 0.00% | 0.00% | 0.8271 |
| xyz ⚠︎ | 28 | 28 | 0.00% | — | — |
| lat ⚠︎ | 27 | 27 | 0.00% | — | — |
| io ⚠︎ | 26 | 1 | 0.00% | 0.00% | 0.0833 |
| help ⚠︎ | 24 | 24 | 0.00% | — | — |
| it ⚠︎ | 20 | 0 | — | 0.00% | — |
| pl ⚠︎ | 20 | 5 | 0.00% | 0.00% | 0.4333 |
| site ⚠︎ | 17 | 17 | 0.00% | — | — |
| casa ⚠︎ | 15 | 15 | 0.00% | — | — |
| co.uk ⚠︎ | 15 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 2,097 | 1,116 | 0.00% | 0.00% | 0.8783 |
| <30 | 920 | 593 | 0.00% | 0.00% | 0.7749 |
| 60-99 | 503 | 262 | 0.00% | 0.00% | 0.8747 |
| 100-199 | 193 | 83 | 0.00% | 0.00% | 0.7160 |
| >=200 ⚠︎ | 25 | 16 | 0.00% | 0.00% | 0.6306 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,986 | 1,986 | 0.00% | — | — |
| tranco:top-1m-20260913 | 1,614 | 0 | — | 0.00% | — |
| openphish | 84 | 84 | 0.00% | — | — |
| tranco:VALIDATION-POOL | 54 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 37.1 ms · p90 37.8 ms · p99 39.2 ms · max 47.1 ms

Batch throughput: 3,509 URLs/s
