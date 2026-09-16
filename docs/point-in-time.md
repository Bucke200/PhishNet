# Point-in-time semantics (Phase 3)

Every enrichment filter is relative to a per-row timestamp. The two
classes have different clocks — this file pins which is which so
age/CT lookups can't silently use future knowledge.

## Time basis by class

### Phishing rows (`label == 1`)

* `first_seen`: submission time for PhishTank (`time_basis: submitted`),
  observation stamp for OpenPhish (`time_basis: observed`). Earliest
  wins across snapshots (`build_splits.enrich`).
* `first_snapshot`: earliest daily raw file (`YYYY-MM-DD` in the
  filename) containing the URL. Recorded in `load_raw` before dedup;
  the minimum per URL survives dedup.
* Snapshot collection time follows a recorded per-file priority (each
  file's anchor and method land in the manifest's `snapshot_anchors`
  block, and each row carries its `snapshot_anchor`):
  1. `openphish-run-stamp` — the OpenPhish stamp of the same collection
     date IS that day's run moment (single-valued per file: 13:06:02 /
     08:16:04 / 08:51:35 / 08:38:27, each postdating its same-date dump
     max). Applies to every file of that date. The dump max always
     precedes it (09-15: 07:03 vs 08:38), so max-anchoring underestimates
     lag by minutes to hours while still reading ≥ 0.
  2. `file-max` — max `first_seen` within the file (no same-date
     OpenPhish file).
  3. `filename-eod` — end-of-day UTC of the filename date, an upper
     bound that overestimates lag; files with no parseable stamps only.
  All three are content-derived and reproducible, never file mtimes.
  `survival_lag_days = collection_ts − first_seen ≥ 0` (asserted; a
  negative lag is broken input and refuses rather than mis-stratifying).
* Strata (pre-committed): fresh ≤ 2d, short ≤ 30d, long > 30d.
  `time_basis: observed` rows (OpenPhish: first_seen IS the snapshot
  moment, so lag would read ~0 for every row) map to `unknown` — live
  phish of unknown age, i.e. the survivor case — and are kept out of the
  fresh slice. Benign rows are `na`.

### Benign rows (`label == 0`)

* CC rows: `first_seen` is the **capture timestamp** (`cc_timestamp`
  in the raw record, `time_basis: commoncrawl-index`). It is not a
  crawl date — the corpus was selected on 2026-09-15 from the
  CC-MAIN-2026-34/30 indexes.
* Crawl rows (Tranco deep links): `first_seen` is the crawl stamp
  (`time_basis: crawled`).
* `first_snapshot` is the file date as for phishing; benign strata
  are always `na` (no survival concept on the benign side).
* Age/CT filters use the same rule with the benign `first_seen`:
  creation dates and certificates must predate capture/crawl.

## Headline rule: unknown stratum stays in (pre-registered)

The ~1,063 OpenPhish (`unknown`) rows land in test because their
`first_seen` is the observation time — roughly a quarter of test-band
phishing, of unknown and possibly long-lived age. Pre-registered before
Step 5 (choosing after the numbers exist is not allowed):

**The headline includes them, with per-stratum slices beside it.**
Rationale: the test band must represent the collected deployment-era
population, and OpenPhish is the only currently-live feed — the most
deployment-relevant source. Excluding a quarter of test phishing by
basis would be post-hoc test-set filtering and would further starve an
already thin fresh slice. The `survival_stratum` slice in every report
(mean recall/FPR/PR-AUC per fresh/short/long/unknown) discloses the
age-mix instead of hiding it; any claim about "fresh" reads from the
fresh slice, never from the headline alone.

## Feature classification

| Group | Features | Status |
|---|---|---|
| Lexical (row a) | Scheme-canonicalized URL features (`canonicalize_scheme` then the extractor; `is_https` constant 0, dropped) | Baseline, retrained on the new train band |
| + Domain age | `domain_age_days` (= RDAP/WHOIS creation → `first_seen`), `age_known`; `age_na` on hosted tenants (platform records are not the tenant's) | Ablation row (b) |
| + CT history | `ct_age_days` (first_seen − earliest pre-cutoff issuance), `ct_cert_count_pre`, `ct_known`; `ct_na` on hosted tenants | Ablation row (c) |
| All safe | Union of the above | Ablation row (d) |
| Tranco vintage rank | Pinned-list rank at the row's `first_seen` | Diagnostic row ONLY (see below), never trained |
| DNS (A/AAAA/MX/NS…) | Resolver answers | Excluded from the ablation; forward-only collection in `collect.py` |

Provenance columns (`age_source`, `ct_provider`) are recorded per row and
never featurized: fallback coverage differs by TLD and phishing clusters
on particular TLDs, so the source would proxy the label.

## Survivorship

* "Fresh" is still survivor-filtered at the scale of hours: a phish taken
  down before the next daily run never enters the corpus at all. The
  fresh stratum measures "caught within 2 days of submission", not
  "all phishing born in the window".
* Measurements to fill in at rebuild (queries against the built split):
  stratum counts per source (`survival_strata` in the manifest, including
  `unknown`); per-stratum lexical-baseline recall (does the baseline
  already separate fresh from long-lived?); per-class/per-stratum
  unknown and na rates (the contamination gate — a phishing-skewed
  failure rate disqualifies the signal however good its AUC).
* Fresh-stratum precondition: `collect.yml` must be green across the band
  window — the fresh stratum cannot be backfilled.

## Tranco selection argument

Every benign row in the CC population was drawn from Tranco 46VQX, so any
Tranco vintage still encodes "part of the benign sampling frame". Pinning
the vintage fixes the timing leak but not the selection leak. Hence the
rank is collected for exactly one labeled diagnostic row (how big is the
leak?) and stays out of the headline table and the trained champion.
