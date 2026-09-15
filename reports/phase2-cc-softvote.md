# PhishNet eval — `cc_soft_vote`

2026-09-15T15:16:48+00:00 · git `5c3dd169` · dataset `test.csv` sha256 `a659fee6e3ab`

6,386 URLs · 2,065 phishing (32.3%) · 2,583 registrable domains · 2026-08-22 → 2026-09-12

vs baseline `legacy_ensemble(models-v1)` on the same dataset — deltas mix every pipeline difference between the two contracts.

## Headline

| Metric | Value | 95% CI (domain bootstrap) | vs baseline |
|---|---|---|---|
| PR-AUC (headline) | 0.9220 | [0.9094, 0.9335] | +0.2735 |
| ROC-AUC | 0.9497 | — | +0.0987 |
| Recall @ FPR≤0.50% | 56.80% | [48.58%, 64.87%] | +56.80% |
| Achieved FPR | 0.49% | — | +0.49% |
| Recall @ FPR≤0.10% | 32.40% | — | +32.40% |
| Achieved FPR (0.10% budget) | 0.09% | — | +0.09% |
| Precision (test set) | 98.24% | — | — |
| Precision @ prevalence 0.0100% | 1.16% | — | — |

Operating threshold **0.938628**. At a deployment prevalence of 0.0100%, this fires **48.6 false warnings per 10,000 URLs browsed**.

TP 1,173 · FP 21 · FN 892 · TN 4,300

Strict point: threshold **0.970916** — nearest attainable point at FPR 0.0926% (ROC is discrete; the curve was not interpolated).

## Calibration

Brier 0.1574 · ECE 0.2268 · MCE 0.5130

| bin | n | mean score | empirical rate | gap |
|---|---|---|---|---|
| 0 [0.008–0.101] | 639 | 0.059 | 0.000 | +0.059 |
| 1 [0.102–0.173] | 639 | 0.138 | 0.006 | +0.131 |
| 2 [0.173–0.259] | 639 | 0.211 | 0.014 | +0.197 |
| 3 [0.259–0.399] | 639 | 0.327 | 0.053 | +0.274 |
| 4 [0.399–0.553] | 639 | 0.475 | 0.094 | +0.381 |
| 5 [0.553–0.723] | 639 | 0.640 | 0.214 | +0.426 |
| 6 [0.723–0.835] | 638 | 0.783 | 0.270 | +0.513 |
| 7 [0.835–0.930] | 638 | 0.885 | 0.622 | +0.263 |
| 8 [0.930–0.972] | 638 | 0.956 | 0.966 | -0.010 |
| 9 [0.972–0.997] | 638 | 0.984 | 0.997 | -0.013 |

## Slices (at the global threshold)

### tld

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| com | 3,237 | 700 | 28.00% | 0.28% | 0.8222 |
| org | 309 | 15 | 6.67% | 0.00% | 0.2915 |
| info | 251 | 250 | 92.40% | 0.00% | 0.9999 |
| net | 166 | 19 | 10.53% | 0.68% | 0.4137 |
| ru | 158 | 3 | 0.00% | 0.65% | 0.0771 |
| lol | 130 | 130 | 86.92% | — | — |
| shop | 126 | 126 | 48.41% | — | — |
| click | 125 | 125 | 78.40% | — | — |
| sbs | 120 | 120 | 92.50% | — | — |
| edu | 101 | 0 | — | 0.00% | — |
| gov | 91 | 0 | — | 1.10% | — |
| top | 90 | 90 | 73.33% | — | — |
| de | 77 | 5 | 40.00% | 0.00% | 0.9000 |
| io | 75 | 1 | 0.00% | 0.00% | 0.0208 |
| cfd | 59 | 59 | 76.27% | — | — |
| pro | 54 | 53 | 86.79% | 0.00% | 0.9986 |
| jp | 50 | 0 | — | 2.00% | — |
| in ⚠︎ | 40 | 2 | 50.00% | 0.00% | 1.0000 |
| it ⚠︎ | 35 | 0 | — | 0.00% | — |
| fr ⚠︎ | 34 | 13 | 53.85% | 0.00% | 0.9945 |
| lat ⚠︎ | 32 | 27 | 48.15% | 0.00% | 0.9959 |
| nl ⚠︎ | 31 | 0 | — | 0.00% | — |
| co ⚠︎ | 29 | 12 | 50.00% | 0.00% | 0.7079 |
| xyz ⚠︎ | 28 | 28 | 82.14% | — | — |
| ai ⚠︎ | 27 | 0 | — | 0.00% | — |

### url_length

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 3,568 | 1,114 | 60.59% | 0.61% | 0.9329 |
| <30 | 1,512 | 591 | 51.61% | 0.65% | 0.8947 |
| 60-99 | 924 | 262 | 52.67% | 0.00% | 0.9585 |
| 100-199 | 320 | 82 | 54.88% | 0.00% | 0.9552 |
| >=200 | 62 | 16 | 62.50% | 0.00% | 0.9601 |

### source

| group | n | pos | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| tranco:top-1m-20260913 | 4,174 | 0 | — | 0.50% | — |
| phishtank | 1,978 | 1,978 | 58.59% | — | — |
| tranco:VALIDATION-POOL | 147 | 0 | — | 0.00% | — |
| openphish | 87 | 87 | 16.09% | — | — |

⚠︎ = fewer than 50 rows; treat as anecdote.

## Latency (single-URL calls)

p50 30.9 ms · p90 33.2 ms · p99 40.6 ms · max 67.2 ms

Batch throughput: 3,217 URLs/s
