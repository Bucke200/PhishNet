# Common-Crawl benign corpus — external acquisition runbook

Status: **pipeline implemented and offline-tested; final fetch NOT yet run.**
No `data/raw/benign-cc-*` artifact exists. Nothing below retrains any model
or touches the frozen baseline (`data/splits/`, `data/splits-large/`,
`reports/baseline.json` — hashes pinned by `tests/test_dataset_identity.py`).

Primary acquisition mechanism: the **columnar index on S3**
(`s3://commoncrawl/cc-index/table/cc-main/warc/`), queried per-domain with
Amazon Athena (`--source columnar`, the default). The CDX front-end
(`--source cdx`) is retained for small probes only — it throttles bulk
fetching. Same pinned crawls (CC-MAIN-2026-34, fallback CC-MAIN-2026-30),
same cache schema/provenance shape, same resume semantics, same seeds.

## What is already in place (verified this session)

* `build_cc_benign.py` — fetch (columnar Athena primary, CDX probe flag)
  + select (quota sampling) pipeline.
* `validate_cc_benign.py` — duplicate/malformed/type-drift/scheme-gap/
  hostname-stat/root-host-form/eTLD+1-overlap checks. Never trains.
* `tests/test_cc_benign_select.py` — 3 offline tests for the v2 root
  synthesis (observed-scheme-only, empirical-wins, provenance).
* `tests/test_cc_columnar.py` + `tests/fixtures/cc-columnar-athena-pages.json`
  — offline proof that columnar rows map into the exact slim-record schema
  the selection stage expects (header skip, NextToken paging, NULL cells,
  timestamp normalisation, collapse-to-earliest dedupe). The fixture caught
  one real pagination bug (NextToken read from inside ResultSet) before any
  live call.
* Frozen Tranco input: `data/raw/tranco-46VQX-top1000000-2026-09-13.csv`
  (sha256 `4fb2f1c0…81d2b`, provenance sidecar alongside it).
* Phishing URL-type targets, measured from the deduplicated feeds in
  `data/raw` (75,833 unique normalised URLs, zero malformed):
  root .3635 / path1 .3600 / pathN .1280 / query .1485.
  Phishing calibration: 91.0% https overall, 81.9% on roots;
  88% of phishing roots sit on non-www subdomains.
* Partial v1 cache `data/raw/cc-index-CC-MAIN-2026-34.json` (1000 entries,
  182 productive, ~792k records). **Superseded — do not select from it**
  (see "v1 query defect" below). Kept only as a fetch-behaviour record.

## v1 query defect (why the cache is superseded, not resumed)

The v1 query used a bare prefix form, `url=<domain>/*`. The CDX index
matches on the SURT key, so that prefix excludes `www.<domain>/` and every
other subdomain. Consequences observed in the v1 cache:

* `facebook.com`, `google.com`, `amazon.com`, `twitter.com`,
  `instagram.com` all returned HTTP 404 on both CC indexes, although they
  demonstrably have captures — the prefix form cannot see them.
* Only ~0.74 usable roots per productive domain; deep pages dominate, so
  the phishing-measured root quota (36%) is unfillable from v1 records.

The v2 query uses the documented domain form,
`url=<domain>&matchType=domain` (apex + www + subdomains, one call),
recorded as `cc_query_form` in every cache entry and in select provenance.
The v2 default cache is a new path,
`data/raw/cc-index-CC-MAIN-2026-34-matchdomain.json`, so v1 bytes are never
mixed in. Resume (`--phase fetch` without `--refetch`) only continues v2
entries: definitive outcomes are kept, stale transient failures are
retried, seeded candidate order is reproduced identically.

## Rate limiting (why the CDX fetch stopped, and why columnar is primary)

After ~4,000 CDX calls in ~2h (`--workers 2`), `index.commoncrawl.org`
began refusing even single-domain probes with `ConnectionError` — IP-based
throttling, not a code bug. Athena queries bill to our AWS account instead
of a shared front-end, so bulk fetching moves there. Rules for any CDX use:

* `--source cdx` is probe-only (single domains, `--workers 1`).
* If `ConnectionError`/`502`/`504` bursts recur, stop and resume hours
  later; resume-from-cache loses nothing (per-stratum incremental saves).
* Never run two fetchers from the same egress IP concurrently.

## External step: exact commands (AWS credentials + unthrottled network)

From the repo root. One-time Athena setup (Query Editor, region us-east-1):

```sql
CREATE DATABASE ccindex;
-- then the flat-schema DDL from commoncrawl/cc-index-table
-- (src/sql/athena/cc-index-create-table-flat.sql) pointed at
-- s3://commoncrawl/cc-index/table/cc-main/warc/
-- (includes warc_record_id BINARY in current upstream; re-check before
-- any future crawl pin — evolution is additive)
MSCK REPAIR TABLE ccindex;
-- verify the pinned partitions exist:
SHOW PARTITIONS ccindex;
```

Verify connectivity with both queries the pipeline splits per domain
(LIMIT-capped smoke only):

```sql
-- evidence query: NO status filter (redirects count as served scheme)
SELECT url, fetch_time, fetch_status, content_digest, content_mime_type
FROM ccindex
WHERE crawl = 'CC-MAIN-2026-34' AND subset = 'warc'
  AND url_host_registered_domain = 'example.com'
LIMIT 10;
-- selection check: the 200-only subset the dataset draws from
SELECT url, fetch_time, fetch_status, content_digest, content_mime_type
FROM ccindex
WHERE crawl = 'CC-MAIN-2026-34' AND subset = 'warc'
  AND url_host_registered_domain = 'example.com'
  AND fetch_status = 200
LIMIT 10;
```

These prove filters, columns, and partition pruning — and nothing else.
`LIMIT` terminates after the first matching row groups, so smoke latency
and bytes-scanned say **nothing** about a production query, which runs to
completion over every matching row group. Do not extrapolate cost or
wall-clock from them; that is what the pilot below is for.

Fetch (columnar default; each command is resumable by re-running it):

```bash
# 1. Fetch + cache the columnar index (record the SQL + table in provenance)
python build_cc_benign.py --phase fetch \
    --athena-output s3://<your-bucket>/phishnet-athena/
# 2. Build the corpus (refuses to overwrite an existing --out)
python build_cc_benign.py --phase select --target-n 12000
#    -> data/raw/benign-cc-CC-MAIN-2026-34-<today>.jsonl + .provenance.json
# 3. Trial split in a scratch dir (NEVER --out data/splits):
mkdir -p /tmp/cc-trial/raw /tmp/cc-trial/split
cp data/raw/openphish-*.jsonl data/raw/phishtank-*.jsonl \
   data/raw/benign-cc-CC-MAIN-2026-34-<today>.jsonl /tmp/cc-trial/raw/
python build_splits.py --raw /tmp/cc-trial/raw --out /tmp/cc-trial/split \
    --deterministic-manifest
# 4. Validate (exit 1 on any failure; JSON report always written)
python validate_cc_benign.py \
    --benign data/raw/benign-cc-CC-MAIN-2026-34-<today>.jsonl \
    --split-dir /tmp/cc-trial/split \
    --out /tmp/cc-trial/validation-report.json
```

CDX probe (same selection schema, same provenance shape, separate cache):

```bash
python build_cc_benign.py --phase fetch --source cdx --cache /tmp/probe.json \
    --productive-per-stratum 2 --workers 1
```

Promote the corpus into `data/raw/` + a `data/splits-cc/`-style split dir
only if every validation gate passes, and record the new files (with
sha256) in the follow-up report. Do not claim production-ready otherwise.

## v2 sampling design (implemented, offline-tested)

* Popularity: six log-spaced Tranco strata, equal productive domains per
  stratum (`--productive-per-stratum 700`), seed 0. Approximately
  log-uniform where the data permits. 700 (not 500) is the bootstrapped
  number: over the pilot yield distribution (median 2 capped roots/domain,
  heavy tail) N=2,100 meets the 4,362 root quota in expectation but fails
  it outright (P≈0.000, mean 3,383) with the top-3 infra giants removed,
  while N=2,900 (10+90+4×700) meets it in both cases (weak-tail mean
  4,672, p10 4,564). 700 also fits the s3 pool ceiling (~729 productive
  from 900 candidates at 81% yield). Cost delta vs 500: ≈+$0.26.
* URL type: quotas are the measured phishing shares above. Deep types are
  always empirical index records under identical per-domain (16),
  per-domain-type (6) and per-eTLD+1 (25) caps. Deduplication runs before
  quota counting on a canonical key (scheme + apex host + path + query;
  earliest capture wins), so www/apex variants and re-crawls never consume
  quota slots. Scheme and query are preserved in the key — collapsing
  across schemes would game the scheme gate by construction. Optional
  `--collapse-digest` drops same-content URLs (counted, off by default).
* Roots: empirical index records first; the residual quota is backfilled
  with synthesised apex roots, at most one per scheme **observed** for
  that domain, dated at the domain's earliest apex capture. Observed means
  the unfiltered per-domain scheme evidence (all fetch statuses — an http
  301 counts as http served); the 200-only selection rows are a different
  population and must never feed the evidence. Empirical duplicates win.
  Each synthesised row carries `synthesized_root: true`, `scheme_evidence`,
  and `time_basis: domain-inferred-root` (empirical rows:
  `commoncrawl-index`). Provenance records both rates per domain and the
  `scheme_evidence_vs_selection` aggregate (mean https share under each +
  delta), so the filter's effect stays visible.
* Host-form rule (committed): **apex, not www** — for synthesised roots
  and for every dedup key. The `www.` prefix is four characters of
  `netloc_len` on the feature that carried the original leak, varying
  arbitrarily by domain, so it is not noise. Consequences, all counted in
  provenance (`domains_no_apex_capture`, `domains_non_apex_seed_skipped`,
  `skipped_duplicate`): a domain with no apex capture at all is skipped
  for backfill rather than falling back to www (its 200 URLs still select
  normally); a www empirical root blocks synthesis of that apex root via
  the (apex host, scheme) dedup key; non-apex seeds are never synthesised.
* Splits stay eTLD+1-grouped: `build_splits.py` partitions benign by
  registrable-domain hash and drops straddling domains from test, so the
  isolation requirement carries over unchanged.

## Known watch items for the full run

* Scheme gate (`validate_cc_benign.py`): binary features gate on the rate
  gap, continuous features on ROC-AUC — an AUC threshold on a binary
  feature compresses the full gap range into [0.5, 1.0] and silently never
  fires on the high side. The gate is
  `|benign_https_rate - phishing_https_rate| <= SCHEME_RATE_GAP_MAX
  (0.04)`, two-sided, with gap + both rates in every report. The
  select provenance records `scheme_handling.post_hoc_scheme_filter:
  false`: if a corpus lands outside tolerance, fix the sampling design,
  never a post-hoc scheme filter (which would pass the gate while trading
  the distortion into depth or rank strata), and record the change there.
* Which phishing rate the gate is calibrated against: the reference is the
  full deduped feeds (91.05% https — the same population the URL-type
  quotas are measured from). The frozen *test* splits show ~77% phishing
  https (recent, campaign-capped positives; train is 83.9%), and
  OpenPhish alone is 69.8% (n=484). At the 91% reference the gate has 9pp
  of upward headroom and the 0.04 tolerance fires on the trial's 0.061
  gap. Note the evaluation-population gap is wider (trial benign 97.1%
  vs test phish ~77%) and pre-existing (frozen benign is 97–99% https
  everywhere) — the gate keeps a new corpus from drifting further, it
  does not remove that inherent signal.

* All root-availability figures so far (~0.7 usable roots/domain, 5%
  roots in the partial-cache trial) were measured under the **broken v1
  query form** and must not be quoted as properties of the v2 design. A
  26-domain stratified re-measurement under `matchType=domain` is scripted
  (`sample_v2_roots.py` pattern: seed-0 pick, per-domain unique-root
  counts) but blocked on the same throttling; run it from an unthrottled
  network before tuning quotas. The root backfill needs no logic change
  either way — it fills the residual quota gap by construction — but its
  expected share (hence the apex-host-form weight) depends on the outcome.

* Partial-cache trial (v2 select on 182 v1 domains — diagnostic only):
  benign 97.1% https overall / 90.5% on roots vs phishing 91.0% / 81.9%.
  Mild https bias; re-check on the full matchType corpus.
* Root host forms diverge structurally (legit apex/www vs phishing
  evil-subdomains). Inherent to the phenomenon, not samplable away;
  the trial-split shape audit is the judge.
* Coverage finding (CC, not us): **15 of 21 productive pilot domains
  have no apex capture at all** — their captures live entirely on
  www/subdomain hosts. That is why the benign root share is hard to hit
  and why the apex-only backfill rule skips most domains; it is a property
  of what CC crawls (deep links first, apex roots rarely), not of the
  sampling. Expect anyone reviewing the root share to ask about it; the
  per-domain answer is `has_apex_capture` in the pilot JSON and
  `domains_no_apex_capture` in every select provenance.
* No crawling, no link-following, no live-crawl dependency: the training
  pipeline never touches the network; generation reproduces from the
  committed cache + recorded seed.

## Wall-clock and cost model (~2,900 productive domains at P=700)

Measured by the 26-domain no-LIMIT pilot (seed-0 pick, 2/4/5/5/5/5 across
strata, pipeline-exact queries, CC-MAIN-2026-34; temp-only output, no
dataset artifact). Per-domain figures:

* Bytes scanned: mean ~52 MB per domain-query (partition pruning +
  pushdown work). × ~3,580 candidate queries ≈ 186 GB ≈ **~$0.93** at
  $5/TB. Cheap; the pilot totalled 1.36 GB for 26 domains / 31 queries.
* Wall time: **median 3.4 s** per domain, but the distribution has a
  mega-domain tail — wikipedia.org took 1,501 s (3.3M rows paginated),
  samsung.com 76 s (165k rows). Engine time is only ~2–3 s/query; the
  tail is pagination + client-side processing, not Athena.
* Extrapolation at `--workers 5`: bulk domains ≈ 3,580 × 3.4 s / 5 ≈
  40 min, plus a mega-domain tail parallelized across workers → **~1–2 h
  total**. Bounded and parallelizable, unlike the CDX throttle. The row
  cap does not shorten this (retrieval happens regardless); it bounds
  cache bytes and select memory.
* Records after dedup: median 110/domain, but head domains return
  30k–3.3M rows. Per-domain row cap committed: `ORDER BY` avoided (sort
  cost + earliest-first bias), instead client-side seeded-uniform sampling
  — sort rows by URL (deterministic content order, never Athena's
  unguaranteed page order), take a seeded-permutation prefix, cap 5,000,
  seed = run seed + 2, recorded in cache meta and select provenance.
  Micro-measured on 3 head domains (wikipedia/samsung/digicert):
  earliest-first skews URL-type mix by up to ~5pp (samsung query-ward;
  ≤2pp elsewhere) vs the seeded sample, so earliest-first was rejected
  per the pre-committed rule. The cap bounds cache bytes and select
  memory, not pagination wall time.
* Seeded-sample mix check (no new queries): the micro's seeded-uniform
  5,000 (permutation over retrieval order, seed 0) already measures what
  `sample_rows()` produces — both are uniform subsets, and the sort-by-URL
  step buys only determinism against page order (unit-tested: identical
  set regardless of input order). Sample vs full mix: wikipedia
  .861/.139 vs .854/.146, samsung .473/.525 vs .481/.518, digicert
  .990/.006 vs .990/.006 (pathN/query) — max Δ 0.8pp. Clean; the
  permutation washes out the URL-sort structure as expected.

| | CDX (`--source cdx`) | Columnar (`--source columnar`) |
|---|---|---|
| Per-query latency | ~1–2 s unthrottled (**measured** this session) | median 3.4 s, mega-domain tail to ~25 min (**measured**, pilot) |
| Queries needed | ~3,500 candidates at ~60% yield (**assumed**; v1 yield was 41% but inflated by the SURT defect) | ~3,580 candidates at ~81% yield (**measured** pilot: 21/26 productive; N=2,900 at P=700) |
| Concurrency | 1 (throttle-verified) | 5 (default; inside the Athena concurrent-query quota) |
| Expected fetch time | ~1.5 h if never throttled; **observed reality: throttled after ~2 h**, so 1–3 days elapsed across cooldowns | ~1–2 h at 5 workers (**extrapolated** from pilot) |
| Cost | free, paid in throttle risk | ~$0.93 total (**extrapolated**: 3,580 queries × 52 MB mean × $5/TB) |
| Failure mode | IP throttling (observed), silent truncation risk: none | query failures surface as `unproductive(columnar:query-failed)` and retry on resume; **proven** pagination handling via fixture test |

Verdict: columnar primary (failure mode is money — measured at ~$1 —
not a shared throttle), CDX retained for single-domain probes and smoke
tests only.

## Pilot root + scheme findings (go/no-go inputs)

* Roots: 89 unique roots over 21 productive pilot domains (mean 4.2,
  median ~1), but concentrated in three infra domains (googleapis 24,
  digicert 22, samsung 20); typical long-tail domains yield 0–2, and
  **15/21 productive domains have no apex capture** (backfill skips).
  Bootstrap over the pilot yield distribution (capped at 6/domain +1 per
  apex domain, 20k resamples, seed 42): N=2,100 meets the 4,362 quota in
  expectation (mean 4,700) but fails it outright without the top-3 giants
  (P≈0.000, mean 3,383); N=2,900 (P=700) meets it in both cases (weak-tail
  mean 4,672, p10 4,564). **Decision: P=700** — robust under both tail
  scenarios, fits the s3 pool ceiling (~729), costs ≈+$0.26. The earlier
  shortfall projection (median-1 reasoning without caps) is superseded by
  this calculation. Margin sanity (100k resamples): selection truncates
  overshoot at the quota, so the binding threshold is not 4,362 but the
  drift-breach floor — available roots below ~3,822 (at full non-root
  fill), which has probability 0.00000 even weak-tail (min observed
  4,322). The p10's 4.6% margin over quota is ~19% margin over what
  actually trips.
* Scheme: domain-mean https 0.901 unfiltered vs 0.902 filtered — the
  301-effect is negligible at corpus scale, and 0.90 vs the 0.9105 phishing
  reference leaves a ~0.01 gap against the 0.04 tolerance. **The scheme
  gate is reachable**; it is not the blocker. (Record-weighted means round
  to 0.999 only because wikipedia.org contributes 3.3M near-all-https
  URLs pre-cap — another reason per-domain caps precede any corpus
  composition claim.)
