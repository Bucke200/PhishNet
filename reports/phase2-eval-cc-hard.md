# PhishNet eval — `cc_retrained(hard-vote)`

2026-09-15T17:48:03+00:00 · git `5c3dd169` · dataset `test.csv` sha256 `a659fee6e3ab`

6,386 URLs · 2,065 phishing (32.3%) · 2,583 registrable domains · 2026-08-22 → 2026-09-12

vs baseline `legacy_ensemble(models-v1)` on the same dataset — deltas mix every pipeline difference between the two contracts.

## Read this first

- Predictor emits only 5 distinct scores. PR-AUC, ROC-AUC and the FPR sweep are not meaningful for a step function - read the operating point only, and fix the predictor to emit probabilities.
- 2358 URLs tie at the top score 1.0000, exceeding the FPR<=0.50% budget of 21 false positives: no threshold inside the tie is reachable, so the operating point collapses above it (recall 0.0).

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.7225 | [0.6943, 0.7507] | +0.0740 |
| ROC-AUC | 0.8992 | — | +0.0481 |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 0.00%] | +0.00% |
| Achieved FPR | 0.00% | — | +0.00% |
| Recall @ FPR≤0.10% | 0.00% | — | +0.00% |
| Achieved FPR (0.10% budget) | 0.00% | — | +0.00% |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 2,065 · TN 4,321

Strict point: threshold **1.000000** — nearest attainable point at FPR 0.0000% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.1804 · ECE 0.2104 · MCE 0.4638

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.000] | 1,278 | 0.000 | 0.003 | -0.003 |
| 1 [0.000–0.250] | 1,277 | 0.114 | 0.016 | +0.098 |
| 2 [0.250–0.750] | 1,277 | 0.588 | 0.124 | +0.464 |
| 3 [0.750–1.000] | 1,277 | 0.962 | 0.593 | +0.369 |
| 4 [1.000–1.000] | 1,277 | 1.000 | 0.882 | +0.118 |
reliability (x = mean score, o = empirical rate):
|0.0                    0.5                    1.0|
|*................................................| bin 0
|.o...x...........................................| bin 1
|......o.....................x....................| bin 2
|............................o.................x..| bin 3
|..........................................o.....x| bin 4

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 3,237 | 700 | 0.00% | 0.00% | 0.5832 |
| org | 309 | 15 | 0.00% | 0.00% | 0.1216 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9960 |
| net | 166 | 19 | 0.00% | 0.00% | 0.2894 |
| ru | 158 | 3 | 0.00% | 0.00% | 0.0314 |
| lol | 130 | 130 | 0.00% | — | — |
| shop | 126 | 126 | 0.00% | — | — |
| click | 125 | 125 | 0.00% | — | — |
| sbs | 120 | 120 | 0.00% | — | — |
| edu | 101 | 0 | — | 0.00% | — |
| gov | 91 | 0 | — | 0.00% | — |
| top | 90 | 90 | 0.00% | — | — |
| de | 77 | 5 | 0.00% | 0.00% | 0.4545 |
| io | 75 | 1 | 0.00% | 0.00% | 0.0182 |
| cfd | 59 | 59 | 0.00% | — | — |
| pro | 54 | 53 | 0.00% | 0.00% | 0.9804 |
| jp | 50 | 0 | — | 0.00% | — |
| in ⚠︎ | 40 | 2 | 0.00% | 0.00% | 0.6667 |
| it ⚠︎ | 35 | 0 | — | 0.00% | — |
| fr ⚠︎ | 34 | 13 | 0.00% | 0.00% | 0.9945 |
| lat ⚠︎ | 32 | 27 | 0.00% | 0.00% | 0.9617 |
| nl ⚠︎ | 31 | 0 | — | 0.00% | — |
| co ⚠︎ | 29 | 12 | 0.00% | 0.00% | 0.4126 |
| xyz ⚠︎ | 28 | 28 | 0.00% | — | — |
| ai ⚠︎ | 27 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 3,568 | 1,114 | 0.00% | 0.00% | 0.7364 |
| <30 | 1,512 | 591 | 0.00% | 0.00% | 0.6373 |
| 60-99 | 924 | 262 | 0.00% | 0.00% | 0.8914 |
| 100-199 | 320 | 82 | 0.00% | 0.00% | 0.8861 |
| >=200 | 62 | 16 | 0.00% | 0.00% | 0.7538 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| tranco:top-1m-20260913 | 4,174 | 0 | — | 0.00% | — |
| phishtank | 1,978 | 1,978 | 0.00% | — | — |
| tranco:VALIDATION-POOL | 147 | 0 | — | 0.00% | — |
| openphish | 87 | 87 | 0.00% | — | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 31.1 ms · p90 33.9 ms · p99 43.7 ms · max 58.5 ms

Batch throughput: 3,465 URLs/s
