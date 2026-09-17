# Enrichment coverage — Phase 3 age-only run (Amendment E)

Run: `data/enrichment-p3-2026-09-17.jsonl`, `run-1`, sealed
`058ee5838fdbce1502befc31514b2565673d6caba0790ba9edb190d60d29c426`
(41,739 records). Population manifest
`aef0a555697ab1115bf4ab90a9627034dcde76f18213d98e69b80c3b144f457a`.
Only the age signal was requested; CT payloads are absent by design
(Amendment E), not by failure.

## Unknown rates per class (na-excluded)

| band | benign unknown | phish unknown | gap | verdict |
|---|---|---|---|---|
| train (n=44,285) | 17.6% (11,778 eligible) | 16.0% (20,988 eligible) | 0.016 | pass |
| test (n=24,819) | 17.0% (20,300 eligible) | 11.1% (2,918 eligible) | 0.059 | **fail** |

The test-band failure is benign-heavy: long-tail CC domains fail
lookups (unparseable WHOIS, resets, timeouts) more often than phishing
domains do. The gate is absolute-gap, so direction doesn't matter —
age is ineligible for the headline (Amendment E.2 applies).

## Unknown rates per stratum (test band, na-excluded)

| stratum | n | na rate | unknown rate |
|---|---|---|---|
| fresh (phish) | 579 | 16.8% | 9.3% |
| short (phish) | 2,236 | 14.8% | 10.0% |
| unknown (phish) | 984 | 46.0% | 16.4% |
| na (benign) | 21,020 | 3.4% | 17.0% |

The `unknown` stratum (OpenPhish, no submission time) is near-half
hosted (na 46.0%) — expected: live-feed rows skew to tenant platforms.

## Failure composition (all bands, 4,909 unknown of 27,969 queried)

| error | n | reading |
|---|---|---|
| whois-no-creation-date | 2,769 | fallback WHOIS servers with no parseable creation line (long-tail TLDs) |
| connection reset | 965 | server-side resets on port 43 / RDAP |
| rdap-404 | 678 | dead-after-takedown domains (see re-registration) |
| timeout (connect/read) | ~250 | slow hosts, bounded by the 10 s timeouts |
| dns-gaierror | 191 | unresolvable hosts (dead domains) |
| ip-literal | 32 | no registration exists by construction |

## Re-registration count (creation after first_seen, fail-closed unknown)

| band | phish | benign |
|---|---|---|
| train | 1,943 | 5 |
| test | 407 | 6 |

Re-registration is near-exclusively phishing: domains (re-)created
after the phish was observed — direct evidence of takedown during the
collection window. These rows read unknown by rule and contribute to
the phishing unknown rate. (Model-card item either way.)

## Age distribution per stratum (test, age-known rows)

| group | n | median | p25–p75 (days) |
|---|---|---|---|
| benign (na) | 16,852 | 7,977 (~21.9 y) | 4,852–10,286 |
| phish fresh | 437 | 5.2 | 0.3–355 |
| phish short | 1,714 | 4.5 | 0.7–300 |
| phish unknown | 444 | 51.4 | 10.6–502 |

Fresh/short phishing is days-old attack infrastructure against
decades-old benign domains — the signal is real where it resolves.

## Enrichment lag (enriched_at − first_seen, median days)

Benign 35.4 (CC capture → enrichment), phishing 8.9. The phishing lag
is the takedown exposure window: every day of delay after the
September snapshots raises the lapsed-domain unknown rate, which is
why the age pass ran the day the population was pinned.

## CT (dropped, not measured)

No CT coverage exists or is claimed. crt.sh JSON failed ~97% of
checkpointed lookups under a shared pool of 10 (documented ~5/min/IP
limit); no validated alternative was provisioned; the feature is
unservable at request time. See Amendment E and the Phase 3 report.
