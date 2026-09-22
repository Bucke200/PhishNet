# Phase 5 lexical-evasion arm — Tier-1 recall under URL transforms

Base: 200 test-split phishing URLs (`p5-lexical-sample:7`). Thresholds pinned in `phase5-E` (`t05=0.9269363298832987`, `t10=0.8780843789420926`), recomputation asserted. Production-mode scoring; eval==production max abs diff 0.0 (asserted 0.0). `paired_bootstrap_ci` on the recall difference vs clean (`n_boot=2000`, seed 7). Pooled multi-variant CIs treat judgments as the unit (lower bound on width, stated). Homoglyph-unicode is descriptive; ASCII (`xn--`) is primary. Homoglyph not applicable: 21 rows. Post-hoc controls (`phase5-F`, descriptive): host-swap on the 200 phish URLs; covered-shortener transform on 200 benign URLs (`p5-benign-sample:7`).

## Recall at t_0.5%

| arm | recall (k/n) | Wilson 95% | paired diff vs clean |
|---|---|---|---|
| clean | 0.5450 (109/200) | [0.4758, 0.6125] | [+0.0000, +0.0000] |
| xn-- (primary) | 0.4916 (88/179) | [0.4193, 0.5643] | [-0.1117, +0.0447] |
| homoglyph-unicode (descriptive) | 0.3687 (66/179) | [0.3015, 0.4415] | [-0.2179, -0.1006] |
| shortener covered (pooled) | 0.9870 (987/1000) | [0.9779, 0.9924] | [+0.4110, +0.4740] |
| shortener uncovered (pooled) | 0.0170 (17/1000) | [0.0106, 0.0271] | [-0.5600, -0.4950] |
| redirect pooled | 0.0000 (0/600) | [0.0000, 0.0064] | [-0.5850, -0.5050] |
| host-swap control (post-hoc) | 0.0300 (6/200) | [0.0138, 0.0639] | [-0.5850, -0.4450] |

## Recall at t_1.0%

| arm | recall (k/n) | Wilson 95% | paired diff vs clean |
|---|---|---|---|
| clean | 0.6450 (129/200) | [0.5765, 0.7080] | [+0.0000, +0.0000] |
| xn-- (primary) | 0.5866 (105/179) | [0.5134, 0.6562] | [-0.1173, +0.0279] |
| homoglyph-unicode (descriptive) | 0.4860 (87/179) | [0.4139, 0.5588] | [-0.2067, -0.0894] |
| shortener covered (pooled) | 0.9980 (998/1000) | [0.9927, 0.9995] | [+0.3230, +0.3820] |
| shortener uncovered (pooled) | 0.1090 (109/1000) | [0.0912, 0.1298] | [-0.5720, -0.5010] |
| redirect pooled | 0.0000 (0/600) | [0.0000, 0.0064] | [-0.6833, -0.6050] |
| host-swap control (post-hoc) | 0.0800 (16/200) | [0.0498, 0.1260] | [-0.6350, -0.4900] |

## Benign alert rate at fixed thresholds (post-hoc control)

Same covered-shortener transform on 200 test-split benign URLs (`p5-benign-sample:7`). Alert = score above threshold (FPR).

| arm | alert rate (k/n) | Wilson 95% | paired diff vs clean benign |
|---|---|---|---|
| benign clean @t05 | 0.0050 (1/200) | [0.0009, 0.0278] | [+0.0000, +0.0000] |
| benign shortened @t05 | 0.9920 (992/1000) | [0.9843, 0.9959] | [+0.9790, +0.9930] |
| benign clean @t10 | 0.0050 (1/200) | [0.0009, 0.0278] | [+0.0000, +0.0000] |
| benign shortened @t10 | 0.9960 (996/1000) | [0.9898, 0.9984] | [+0.9840, +0.9960] |

Per-host tables (descriptive) are in `reports/phase5-lexical.json` under `per_host_descriptive` (covered hosts all 0.97–1.00; uncovered synthetic hosts 0.00–0.085; all three redirect hosts exactly 0.00 at both thresholds).

## Reading the directions (interpretation, not protocol)

- Covered shorteners raise recall (+0.41–+0.47 paired): `is_shortened` is a learned phish indicator, so collapsing to the shortener host's score helps the defender here. The prereg expectation was direction-neutral and holds.
- Uncovered shorteners collapse recall (−0.50–−0.56) — with a caveat: the uncovered hosts are synthetic `.example` names, never seen in training, so this measures 'unknown short host', not a real-world uncovered service.
- Redirect pooled 0.000 carries the same caveat: its hosts are fixed `.example` names, not real redirectors. The host-swap control adjudicates: same URLs on random `.example` hosts (path kept, no redirect) recall 0.030/0.080 — collapsed just as hard. The TLD is the effect: Tier-1 scores unknown hosts as benign. Redirect wrapping adds nothing beyond the host swap.
- Benign shortener control: covered-shortened benign links alert at 0.992/0.996 against a clean-benign 0.005/0.005 baseline. The model flags link shorteners, not phishing — a Phase 6 production-gaps finding (`phase5-F`), never a Phase 5 headline. `is_shortened` is a source-composition artifact, same family as the takedown leak. The clean-benign 0.005 is consistent with the budget, not a calibration check: at N=200 the smallest nonzero FPR is 1/200, so both thresholds landing there means one alerting row. The real calibration evidence is Phase 3's full-test FPR (0.40% on 21,020 benign rows).
- `xn--` costs little (−0.11–+0.04, indistinguishable); the Unicode form costs more (−0.22–−0.10, descriptive only).
