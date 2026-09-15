# PhishNet eval — `gbm_isotonic`

2026-09-15T17:47:37+00:00 · git `5c3dd169` · dataset `test.csv` sha256 `a659fee6e3ab`

6,386 URLs · 2,065 phishing (32.3%) · 2,583 registrable domains · 2026-08-22 → 2026-09-12

vs baseline `legacy_ensemble(models-v1)` on the same dataset — deltas mix every pipeline difference between the two contracts.

## Read this first

- 1096 URLs tie at the top score 1.0000, exceeding the FPR<=0.50% budget of 21 false positives: no threshold inside the tie is reachable, so the operating point collapses above it (recall 0.0).

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.9009 | [0.8837, 0.9167] | +0.2524 |
| ROC-AUC | 0.9478 | — | +0.0968 |
| Recall @ FPR≤0.50% | 0.00% | [0.00%, 60.74%] | +0.00% |
| Achieved FPR | 0.00% | — | +0.00% |
| Recall @ FPR≤0.10% | 0.00% | — | +0.00% |
| Achieved FPR (0.10% budget) | 0.00% | — | +0.00% |
| Precision (test set) | — | — | — |
| Precision @ prevalence 0.0100% | — | — | — |

Operating threshold **1.000000**. At a deployment prevalence of 0.0100%, this fires **0.0 false warnings per 10,000 URLs browsed**.

TP 0 · FP 0 · FN 2,065 · TN 4,321

Strict point: threshold **1.000000** — nearest attainable point at FPR 0.0000% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.1891 · ECE 0.2601 · MCE 0.5819

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.000–0.061] | 639 | 0.028 | 0.000 | +0.028 |
| 1 [0.061–0.087] | 639 | 0.079 | 0.006 | +0.073 |
| 2 [0.087–0.281] | 639 | 0.246 | 0.027 | +0.219 |
| 3 [0.281–0.524] | 639 | 0.383 | 0.025 | +0.358 |
| 4 [0.524–0.636] | 639 | 0.594 | 0.081 | +0.512 |
| 5 [0.636–0.742] | 639 | 0.705 | 0.213 | +0.493 |
| 6 [0.742–0.870] | 638 | 0.848 | 0.266 | +0.582 |
| 7 [0.870–0.973] | 638 | 0.959 | 0.663 | +0.296 |
| 8 [0.973–1.000] | 638 | 0.995 | 0.958 | +0.037 |
| 9 [1.000–1.000] | 638 | 1.000 | 0.997 | +0.003 |
reliability (x = mean score, o = empirical rate):
|0.0                    0.5                    1.0|
|ox...............................................| bin 0
|o...x............................................| bin 1
|.o..........x....................................| bin 2
|.o................x..............................| bin 3
|....o.......................x....................| bin 4
|..........o.......................x..............| bin 5
|.............o...........................x.......| bin 6
|................................o.............x..| bin 7
|..............................................o.x| bin 8
|................................................*| bin 9

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 3,237 | 700 | 0.00% | 0.00% | 0.7719 |
| org | 309 | 15 | 0.00% | 0.00% | 0.3245 |
| info | 251 | 250 | 0.00% | 0.00% | 0.9988 |
| net | 166 | 19 | 0.00% | 0.00% | 0.4086 |
| ru | 158 | 3 | 0.00% | 0.00% | 0.1181 |
| lol | 130 | 130 | 0.00% | — | — |
| shop | 126 | 126 | 0.00% | — | — |
| click | 125 | 125 | 0.00% | — | — |
| sbs | 120 | 120 | 0.00% | — | — |
| edu | 101 | 0 | — | 0.00% | — |
| gov | 91 | 0 | — | 0.00% | — |
| top | 90 | 90 | 0.00% | — | — |
| de | 77 | 5 | 0.00% | 0.00% | 0.8435 |
| io | 75 | 1 | 0.00% | 0.00% | 0.0312 |
| cfd | 59 | 59 | 0.00% | — | — |
| pro | 54 | 53 | 0.00% | 0.00% | 0.9989 |
| jp | 50 | 0 | — | 0.00% | — |
| in ⚠︎ | 40 | 2 | 0.00% | 0.00% | 1.0000 |
| it ⚠︎ | 35 | 0 | — | 0.00% | — |
| fr ⚠︎ | 34 | 13 | 0.00% | 0.00% | 0.9339 |
| lat ⚠︎ | 32 | 27 | 0.00% | 0.00% | 0.9915 |
| nl ⚠︎ | 31 | 0 | — | 0.00% | — |
| co ⚠︎ | 29 | 12 | 0.00% | 0.00% | 0.7608 |
| xyz ⚠︎ | 28 | 28 | 0.00% | — | — |
| ai ⚠︎ | 27 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 3,568 | 1,114 | 0.00% | 0.00% | 0.9042 |
| <30 | 1,512 | 591 | 0.00% | 0.00% | 0.8655 |
| 60-99 | 924 | 262 | 0.00% | 0.00% | 0.9656 |
| 100-199 | 320 | 82 | 0.00% | 0.00% | 0.9694 |
| >=200 | 62 | 16 | 0.00% | 0.00% | 0.9655 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| tranco:top-1m-20260913 | 4,174 | 0 | — | 0.00% | — |
| phishtank | 1,978 | 1,978 | 0.00% | — | — |
| tranco:VALIDATION-POOL | 147 | 0 | — | 0.00% | — |
| openphish | 87 | 87 | 0.00% | — | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 3.6 ms · p90 4.1 ms · p99 4.4 ms · max 4.9 ms

Batch throughput: 3,518 URLs/s
