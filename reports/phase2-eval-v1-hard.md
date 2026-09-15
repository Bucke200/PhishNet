# PhishNet eval — `legacy_ensemble(models-v1)`

2026-09-15T14:46:26+00:00 · git `5c3dd169` · dataset `test.csv` sha256 `a659fee6e3ab`

6,386 URLs · 2,065 phishing (32.3%) · 2,583 registrable domains · 2026-08-22 → 2026-09-12

## Read this first

- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function — read the operating point only, and fix the predictor to emit probabilities.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.6485 | [0.6217, 0.6767] | — |
| ROC-AUC | 0.8511 | — | — |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 0.00%] | — |
| Achieved FPR | 0.00% | — | — |
| Recall @ FPR≤0.10% | 0.00% | — | — |
| Achieved FPR (0.10% budget) | 0.00% | — | — |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 2,065 · TN 4,321

Strict point: threshold **1.000000** — nearest attainable point at FPR 0.0000% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.2016 · ECE 0.1921 · MCE 0.4013

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.000] | 1,278 | 0.000 | 0.008 | -0.008 |
| 1 [0.000–0.250] | 1,277 | 0.084 | 0.060 | +0.024 |
| 2 [0.250–0.750] | 1,277 | 0.556 | 0.205 | +0.351 |
| 3 [0.750–1.000] | 1,277 | 0.922 | 0.521 | +0.401 |
| 4 [1.000–1.000] | 1,277 | 1.000 | 0.824 | +0.176 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 3,237 | 700 | 0.00% | 0.00% | 0.4855 |
| org | 309 | 15 | 0.00% | 0.00% | 0.2180 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9955 |
| net | 166 | 19 | 0.00% | 0.00% | 0.2410 |
| ru | 158 | 3 | 0.00% | 0.00% | 0.0474 |
| lol | 130 | 130 | 0.00% | — | — |
| shop | 126 | 126 | 0.00% | — | — |
| click | 125 | 125 | 0.00% | — | — |
| sbs | 120 | 120 | 0.00% | — | — |
| edu | 101 | 0 | — | 0.00% | — |
| gov | 91 | 0 | — | 0.00% | — |
| top | 90 | 90 | 0.00% | — | — |
| de | 77 | 5 | 0.00% | 0.00% | 0.4167 |
| io | 75 | 1 | 0.00% | 0.00% | 0.0270 |
| cfd | 59 | 59 | 0.00% | — | — |
| pro | 54 | 53 | 0.00% | 0.00% | 0.9804 |
| jp | 50 | 0 | — | 0.00% | — |
| in ⚠︎ | 40 | 2 | 0.00% | 0.00% | 0.1667 |
| it ⚠︎ | 35 | 0 | — | 0.00% | — |
| fr ⚠︎ | 34 | 13 | 0.00% | 0.00% | 0.8809 |
| lat ⚠︎ | 32 | 27 | 0.00% | 0.00% | 0.9148 |
| nl ⚠︎ | 31 | 0 | — | 0.00% | — |
| co ⚠︎ | 29 | 12 | 0.00% | 0.00% | 0.7862 |
| xyz ⚠︎ | 28 | 28 | 0.00% | — | — |
| ai ⚠︎ | 27 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 3,568 | 1,114 | 0.00% | 0.00% | 0.7448 |
| <30 | 1,512 | 591 | 0.00% | 0.00% | 0.5495 |
| 60-99 | 924 | 262 | 0.00% | 0.00% | 0.7462 |
| 100-199 | 320 | 82 | 0.00% | 0.00% | 0.5177 |
| >=200 | 62 | 16 | 0.00% | 0.00% | 0.2404 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| tranco:top-1m-20260913 | 4,174 | 0 | — | 0.00% | — |
| phishtank | 1,978 | 1,978 | 0.00% | — | — |
| tranco:VALIDATION-POOL | 147 | 0 | — | 0.00% | — |
| openphish | 87 | 87 | 0.00% | — | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 41.2 ms · p90 46.9 ms · p99 55.4 ms · max 62.4 ms

Batch throughput: 3,477 URLs/s
