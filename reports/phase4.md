# Phase 4 report — PROVISIONAL (no recorded run yet)

Only a `recorded` run publishes (§4.1). This report is explicitly
provisional: verdicts cover a sealed subset of the in-band test
population; unjudged in-band rows retain Tier-1 per the §2 failure
policy. Coverage is stated beside every number (criterion 14).

## Cascade [default] (provisional)
- fpr0.5: cascade recall=0.5049 fpr=0.00414 (tp=1918 fp=87) vs tier1 recall=0.5041 fpr=0.00404 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
- fpr1.0: cascade recall=0.6070 fpr=0.00990 (tp=2306 fp=208) vs tier1 recall=0.6065 fpr=0.00980 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
## Cascade [alternative] (provisional)
- fpr0.5: cascade recall=0.5049 fpr=0.00414 (tp=1918 fp=87) vs tier1 recall=0.5041 fpr=0.00404 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.
- fpr1.0: cascade recall=0.6070 fpr=0.00990 (tp=2306 fp=208) vs tier1 recall=0.6065 fpr=0.00980 vs password baseline recall=0.0000 fpr=0.00010; paired PR-AUC lift [-0.0003, 0.0002] -> indistinguishable.

## Descriptive (provisional)
- agreement on sealed rows: 0.956 (n=68)
- verdict distribution: {'benign': 63, 'phishing': 5}
- rank metrics: artifactual per §2 (tie block at t_alert); fixed-threshold recall/FPR above is primary.
- cost (cold-cache, n=119): prompt 1476 / completion 243 (reasoning 90, visible 154) tokens per call; latency p50 1308ms p90 2069ms.
- coverage: verdicts sealed for 68/1106 test in-band fetched-ok rows; unjudged rows retain Tier-1 (§2 failure policy).

## Criteria (provisional reading)
- 1 met (registration 509ff11f before first snapshot/gate call).
- 2 met (threshold_at_fpr twice on calib, no loop).
- 3 UNMET by one discrete row: achieved benign band mass 406/8110 = 0.0501 (both edges individually at-most-target; stated, not rounded).
- 4 met (boundary tests pass).
- 5 met (cascade mapping §2 exact; no confidence gating).
- 6 met (both unfetchable policies reported).
- 7 met (trigger mechanical; phase4-B option-1).
- 8 met (no row holds two snapshots).
- 9 amended by phase4-C (model+prompt+seed identity; fingerprint distribution sealed, explicitly weaker).
- 9a met (5/5 gate pass, no json_object).
- 9b UNMET (no recorded run yet; free-tier TPD fits ~100 calls/day, full 1106-row sweep needs Developer tier).
- 10 met (disagreement 11/50 reported with decomposition).
- 10a n/a (no negative claim made on provisional data).
- 10b pending (over bar -> headline is a range over three full-population repeats, on the recorded sweep).
- 11 met (raw-HTML-never-sent test).
- 12 met (baseline sealed before any LLM read).
- 13 met for split + cold-cache counts; priced forecast pending per-token rate (re-check at report time).
- 14 met (coverage beside every number).
