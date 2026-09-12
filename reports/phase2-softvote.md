# PhishNet eval — `soft_vote(models-v1)`

2026-09-12T20:44:37+00:00 · git `5eed0cf1` · dataset `test.csv` sha256 `385aa409c222`

2,538 URLs · 2,076 phishing (81.8%) · 1,700 registrable domains · 2026-08-22 → 2026-09-12

## Read this first

- Only 462 negatives, so an FPR of 0.500% is 2 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.
- Test base rate is 81.8%; PR-AUC is base-rate dependent and not comparable across datasets.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.9729 | [0.9646, 0.9807] | +0.0223 |
| ROC-AUC | 0.8984 | — | +0.0270 |
| Recall @ FPR≤0.50% | 11.66% | [10.03%, 38.03%] | +11.66% |
| Achieved FPR | 0.43% | — | +0.43% |
| Precision (test set) | 99.18% | — | — |
| Precision @ prevalence 0.0100% | 0.27% | — | — |

Operating threshold **0.977088**. At a deployment prevalence of 0.0100%, this fires **43.3 false warnings per 10,000 URLs browsed**.

TP 242 · FP 2 · FN 1,834 · TN 460

## Calibration

Brier 0.1023 · ECE 0.0999 · MCE 0.2696

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.020–0.220] | 254 | 0.106 | 0.161 | -0.056 |
| 1 [0.220–0.445] | 254 | 0.349 | 0.618 | -0.270 |
| 2 [0.447–0.638] | 254 | 0.551 | 0.728 | -0.177 |
| 3 [0.639–0.738] | 254 | 0.694 | 0.894 | -0.200 |
| 4 [0.739–0.844] | 254 | 0.799 | 0.882 | -0.083 |
| 5 [0.844–0.897] | 254 | 0.869 | 0.976 | -0.108 |
| 6 [0.897–0.933] | 254 | 0.916 | 0.941 | -0.025 |
| 7 [0.933–0.960] | 254 | 0.947 | 0.988 | -0.041 |
| 8 [0.960–0.977] | 253 | 0.968 | 1.000 | -0.032 |
| 9 [0.977–0.999] | 253 | 0.985 | 0.992 | -0.007 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,009 | 708 | 6.21% | 0.66% | 0.9255 |
| info | 250 | 250 | 0.00% | — | — |
| lol | 131 | 131 | 9.16% | — | — |
| shop | 126 | 126 | 17.46% | — | — |
| click | 125 | 125 | 36.00% | — | — |
| sbs | 120 | 120 | 15.83% | — | — |
| top | 91 | 91 | 24.18% | — | — |
| org | 65 | 15 | 0.00% | 0.00% | 0.7458 |
| cfd | 60 | 60 | 13.33% | — | — |
| pro | 53 | 53 | 22.64% | — | — |
| net ⚠︎ | 31 | 19 | 0.00% | 0.00% | 0.9046 |
| xyz ⚠︎ | 28 | 28 | 21.43% | — | — |
| lat ⚠︎ | 27 | 27 | 18.52% | — | — |
| help ⚠︎ | 24 | 24 | 8.33% | — | — |
| de ⚠︎ | 17 | 5 | 20.00% | 0.00% | 0.9667 |
| ru ⚠︎ | 17 | 2 | 0.00% | 0.00% | 0.4167 |
| site ⚠︎ | 17 | 17 | 41.18% | — | — |
| casa ⚠︎ | 15 | 15 | 6.67% | — | — |
| online ⚠︎ | 15 | 15 | 0.00% | — | — |
| pl ⚠︎ | 15 | 5 | 20.00% | 0.00% | 0.8767 |
| fr ⚠︎ | 14 | 13 | 15.38% | 0.00% | 0.9822 |
| co ⚠︎ | 12 | 12 | 8.33% | — | — |
| cn ⚠︎ | 9 | 8 | 0.00% | 0.00% | 0.9318 |
| com.br ⚠︎ | 9 | 9 | 11.11% | — | — |
| app ⚠︎ | 8 | 8 | 12.50% | — | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 1,392 | 1,121 | 14.45% | 0.37% | 0.9809 |
| <30 | 694 | 596 | 0.17% | 0.00% | 0.9503 |
| 60-99 | 321 | 258 | 21.32% | 1.59% | 0.9801 |
| 100-199 | 112 | 84 | 16.67% | 0.00% | 0.9641 |
| >=200 ⚠︎ | 19 | 17 | 58.82% | 0.00% | 0.9769 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,991 | 1,991 | 11.85% | — | — |
| tranco:top-1m-20260913 | 403 | 0 | — | 0.50% | — |
| openphish | 85 | 85 | 7.06% | — | — |
| tranco:VALIDATION-POOL | 59 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 44.3 ms · p90 48.4 ms · p99 76.8 ms · max 101.6 ms

Batch throughput: 1,991 URLs/s
