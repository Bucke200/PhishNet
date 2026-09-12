# PhishNet eval — `soft_vote(models-v1)`

2026-09-12T22:20:09+00:00 · git `5eed0cf1` · dataset `test.csv` sha256 `961efe5c9bfc`

3,738 URLs · 2,070 phishing (55.4%) · 1,963 registrable domains · 2026-08-22 → 2026-09-12

## Read this first

- Only 1668 negatives, so an FPR of 0.500% is 8 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.9035 | [0.8845, 0.9193] | — |
| ROC-AUC | 0.8885 | — | — |
| Recall @ FPR≤0.50% | 14.35% | [1.35%, 29.12%] | — |
| Achieved FPR | 0.48% | — | — |
| Precision (test set) | 97.38% | — | — |
| Precision @ prevalence 0.0100% | 0.30% | — | — |

Operating threshold **0.973026**. At a deployment prevalence of 0.0100%, this fires **48.0 false warnings per 10,000 URLs browsed**.

TP 297 · FP 8 · FN 1,773 · TN 1,660

## Calibration

Brier 0.1385 · ECE 0.0552 · MCE 0.1449

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.014–0.093] | 374 | 0.063 | 0.029 | +0.033 |
| 1 [0.093–0.212] | 374 | 0.138 | 0.075 | +0.063 |
| 2 [0.213–0.399] | 374 | 0.316 | 0.275 | +0.041 |
| 3 [0.400–0.564] | 374 | 0.477 | 0.396 | +0.081 |
| 4 [0.564–0.684] | 374 | 0.629 | 0.484 | +0.145 |
| 5 [0.684–0.805] | 374 | 0.744 | 0.644 | +0.099 |
| 6 [0.806–0.886] | 374 | 0.849 | 0.845 | +0.004 |
| 7 [0.886–0.936] | 374 | 0.914 | 0.848 | +0.066 |
| 8 [0.936–0.969] | 373 | 0.954 | 0.968 | -0.014 |
| 9 [0.969–0.999] | 373 | 0.981 | 0.976 | +0.005 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,692 | 705 | 8.23% | 0.41% | 0.7937 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9856 |
| lol | 131 | 131 | 13.74% | — | — |
| org | 126 | 15 | 6.67% | 0.00% | 0.6288 |
| shop | 126 | 126 | 17.46% | — | — |
| click | 125 | 125 | 39.20% | — | — |
| sbs | 120 | 120 | 23.33% | — | — |
| top | 90 | 90 | 31.11% | — | — |
| net | 82 | 19 | 0.00% | 3.17% | 0.5272 |
| cfd | 60 | 60 | 13.33% | — | — |
| gov | 53 | 0 | — | 0.00% | — |
| pro | 53 | 53 | 26.42% | — | — |
| edu ⚠︎ | 40 | 0 | — | 0.00% | — |
| ru ⚠︎ | 38 | 2 | 0.00% | 0.00% | 0.2083 |
| de ⚠︎ | 32 | 5 | 20.00% | 0.00% | 0.9667 |
| fr ⚠︎ | 29 | 13 | 15.38% | 0.00% | 0.9372 |
| xyz ⚠︎ | 28 | 28 | 35.71% | — | — |
| lat ⚠︎ | 27 | 27 | 18.52% | — | — |
| io ⚠︎ | 26 | 1 | 0.00% | 0.00% | 0.0769 |
| help ⚠︎ | 24 | 24 | 8.33% | — | — |
| it ⚠︎ | 20 | 0 | — | 0.00% | — |
| pl ⚠︎ | 20 | 5 | 20.00% | 0.00% | 0.6968 |
| site ⚠︎ | 17 | 17 | 52.94% | — | — |
| casa ⚠︎ | 15 | 15 | 13.33% | — | — |
| co.uk ⚠︎ | 15 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 2,097 | 1,116 | 18.10% | 0.10% | 0.9398 |
| <30 | 920 | 593 | 0.51% | 0.61% | 0.8204 |
| 60-99 | 503 | 262 | 25.57% | 0.83% | 0.9149 |
| 100-199 | 193 | 83 | 19.28% | 1.82% | 0.8406 |
| >=200 ⚠︎ | 25 | 16 | 56.25% | 11.11% | 0.8522 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,986 | 1,986 | 14.70% | — | — |
| tranco:top-1m-20260913 | 1,614 | 0 | — | 0.50% | — |
| openphish | 84 | 84 | 5.95% | — | — |
| tranco:VALIDATION-POOL | 54 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 37.3 ms · p90 38.2 ms · p99 40.0 ms · max 41.7 ms

Batch throughput: 3,464 URLs/s
