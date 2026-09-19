# Phase 5 lexical-evasion arm — Tier-1 recall under URL transforms

Base: 200 test-split phishing URLs (`p5-lexical-sample:7`). Thresholds pinned in `phase5-E` (`t05=0.9269363298832987`, `t10=0.8780843789420926`), recomputation asserted. Production-mode scoring; eval==production max abs diff 0.0 (asserted 0.0). `paired_bootstrap_ci` on the recall difference vs clean (`n_boot=2000`, seed 7). Pooled multi-variant CIs treat judgments as the unit (lower bound on width, stated). Homoglyph-unicode is descriptive; ASCII (`xn--`) is primary. Homoglyph not applicable: 21 rows.

## Recall at t_0.5%

| arm | recall (k/n) | Wilson 95% | paired diff vs clean |
|---|---|---|---|
| clean | 0.5450 (109/200) | [0.4758, 0.6125] | [+0.0000, +0.0000] |
| xn-- (primary) | 0.4916 (88/179) | [0.4193, 0.5643] | [-0.1117, +0.0447] |
| homoglyph-unicode (descriptive) | 0.3687 (66/179) | [0.3015, 0.4415] | [-0.2179, -0.1006] |
| shortener covered (pooled) | 0.9870 (987/1000) | [0.9779, 0.9924] | [+0.4110, +0.4740] |
| shortener uncovered (pooled) | 0.0170 (17/1000) | [0.0106, 0.0271] | [-0.5600, -0.4950] |
| redirect pooled | 0.0000 (0/600) | [0.0000, 0.0064] | [-0.5850, -0.5050] |

## Recall at t_1.0%

| arm | recall (k/n) | Wilson 95% | paired diff vs clean |
|---|---|---|---|
| clean | 0.6450 (129/200) | [0.5765, 0.7080] | [+0.0000, +0.0000] |
| xn-- (primary) | 0.5866 (105/179) | [0.5134, 0.6562] | [-0.1173, +0.0279] |
| homoglyph-unicode (descriptive) | 0.4860 (87/179) | [0.4139, 0.5588] | [-0.2067, -0.0894] |
| shortener covered (pooled) | 0.9980 (998/1000) | [0.9927, 0.9995] | [+0.3230, +0.3820] |
| shortener uncovered (pooled) | 0.1090 (109/1000) | [0.0912, 0.1298] | [-0.5720, -0.5010] |
| redirect pooled | 0.0000 (0/600) | [0.0000, 0.0064] | [-0.6833, -0.6050] |

Per-host tables (descriptive) are in `reports/phase5-lexical.json`
under `per_host_descriptive` (covered hosts all 0.97–1.00; uncovered
synthetic hosts 0.00–0.085; all three redirect hosts exactly 0.00 at both
thresholds — the pattern is systematic, not a single-host artifact).

## Reading the directions (interpretation, not protocol)

- Covered shorteners raise recall (+0.41–0.47 paired): `is_shortened` is a
  learned phish indicator, so collapsing to the shortener host's score helps
  the defender here. The prereg expectation was direction-neutral and holds.
- Uncovered shorteners collapse recall (−0.50–0.56) — with a caveat: the
  uncovered hosts are synthetic `.example` names, never seen in training, so
  this measures "unknown short host", not a real-world uncovered service. A
  real obscure shortener could score anywhere between these poles.
- Open redirects evade completely (0/600): Tier-1 is domain-centric, and a
  benign host plus a long query carries no learned phish signal. No expansion
  step exists in serving (prereg §7.2, Phase 6 item).
- `xn--` costs little (−0.11–+0.04, indistinguishable); the Unicode form
  costs more (−0.22–−0.10, descriptive only).
