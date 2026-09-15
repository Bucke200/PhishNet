# PhishNet eval — `cc_retrained(hard-vote)`

2026-09-15T08:44:52+00:00 · git `7710c230` · dataset `test.csv` sha256 `385aa409c222`

2,538 URLs · 2,076 phishing (81.8%) · 1,700 registrable domains · 2026-08-22 → 2026-09-12

## Read this first

- Only 462 negatives, so an FPR of 0.500% is 2 false positives. The operating point is estimated from too few events; you need ~4,000 negatives for a stable estimate.
- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function — read the operating point only, and fix the predictor to emit probabilities.
- Test base rate is 81.8%; PR-AUC is base-rate dependent and not comparable across datasets.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.9638 | [0.9506, 0.9755] | +0.0131 |
| ROC-AUC | 0.9100 | — | +0.0386 |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 0.00%] | +0.00% |
| Achieved FPR | 0.00% | — | +0.00% |
| Recall @ FPR≤0.10% | 0.00% | — | — |
| Achieved FPR (0.10% budget) | 0.00% | — | — |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 2,076 · TN 462

Strict point: threshold **1.000000** — nearest attainable point at FPR 0.0000% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.0665 · ECE 0.0184 · MCE 0.0453

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.750] | 508 | 0.250 | 0.266 | -0.016 |
| 1 [0.750–1.000] | 508 | 0.901 | 0.892 | +0.009 |
| 2 [1.000–1.000] | 508 | 1.000 | 0.955 | +0.045 |
| 3 [1.000–1.000] | 507 | 1.000 | 0.982 | +0.018 |
| 4 [1.000–1.000] | 507 | 1.000 | 0.996 | +0.004 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 1,009 | 708 | 0.00% | 0.00% | 0.9305 |
| info | 250 | 250 | 0.00% | — | — |
| lol | 131 | 131 | 0.00% | — | — |
| shop | 126 | 126 | 0.00% | — | — |
| click | 125 | 125 | 0.00% | — | — |
| sbs | 120 | 120 | 0.00% | — | — |
| top | 91 | 91 | 0.00% | — | — |
| org | 65 | 15 | 0.00% | 0.00% | 0.4331 |
| cfd | 60 | 60 | 0.00% | — | — |
| pro | 53 | 53 | 0.00% | — | — |
| net ⚠︎ | 31 | 19 | 0.00% | 0.00% | 0.7162 |
| xyz ⚠︎ | 28 | 28 | 0.00% | — | — |
| lat ⚠︎ | 27 | 27 | 0.00% | — | — |
| help ⚠︎ | 24 | 24 | 0.00% | — | — |
| de ⚠︎ | 17 | 5 | 0.00% | 0.00% | 1.0000 |
| ru ⚠︎ | 17 | 2 | 0.00% | 0.00% | 0.5588 |
| site ⚠︎ | 17 | 17 | 0.00% | — | — |
| casa ⚠︎ | 15 | 15 | 0.00% | — | — |
| online ⚠︎ | 15 | 15 | 0.00% | — | — |
| pl ⚠︎ | 15 | 5 | 0.00% | 0.00% | 0.6250 |
| fr ⚠︎ | 14 | 13 | 0.00% | 0.00% | 1.0000 |
| co ⚠︎ | 12 | 12 | 0.00% | — | — |
| cn ⚠︎ | 9 | 8 | 0.00% | 0.00% | 1.0000 |
| com.br ⚠︎ | 9 | 9 | 0.00% | — | — |
| app ⚠︎ | 8 | 8 | 0.00% | — | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 1,392 | 1,121 | 0.00% | 0.00% | 0.9635 |
| <30 | 694 | 596 | 0.00% | 0.00% | 0.9515 |
| 60-99 | 321 | 258 | 0.00% | 0.00% | 0.9945 |
| 100-199 | 112 | 84 | 0.00% | 0.00% | 0.9787 |
| >=200 ⚠︎ | 19 | 17 | 0.00% | 0.00% | 0.9869 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| phishtank | 1,991 | 1,991 | 0.00% | — | — |
| tranco:top-1m-20260913 | 403 | 0 | — | 0.00% | — |
| openphish | 85 | 85 | 0.00% | — | — |
| tranco:VALIDATION-POOL | 59 | 0 | — | 0.00% | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 37.6 ms · p90 39.0 ms · p99 43.2 ms · max 45.8 ms

Batch throughput: 3,487 URLs/s
