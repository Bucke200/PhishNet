TRANCO_ID   ?= NONE
DOMAINS     ?= 1200
PER_DOMAIN  ?= 12
TEST_DAYS   ?= 21
PRED        ?= predictors:LegacyEnsemble
TEST        := data/splits/test.csv
BASELINE    := reports/baseline.json
# Belt and braces behind the snapshot-pinned builder: any tldextract call
# site without an explicit cache_dir resolves its cache here instead of a
# machine-local directory.
TLDEXTRACT_CACHE ?= .tld_cache
export TLDEXTRACT_CACHE
# Successor evaluation population: pinned cutoff (never now-minus-days),
# eval-heavy benign allocation, own output dir (override with OUT= for the
# repro job). Frozen dirs stay untouched.
SPLIT_DATE  ?= 2026-08-22 22:18:46.738237+00:00
EVAL_FRAC   ?= 0.5
OUT         ?= data/splits-eval
# Exact enlarged deep-link input set (data/raw also holds other snapshots).
EVAL_RAW_FILES := benign-2026-09-13.jsonl \
	benign-ranks1201-3900-2026-09-13.jsonl \
	openphish-2026-09-12.jsonl \
	phishtank-2026-09-12.jsonl

## Phase 3 enlarged corpus: select (quotas measured fresh and pinned) then
## validate. Gates run BEFORE any model scores the corpus — cc-gates is the
## single entry point; never train or evaluate on an unvalidated corpus.
CC_OUT    ?= data/raw/benign-cc-CC-MAIN-2026-34-enlarged.jsonl
CC_TARGET ?= 40000
CC_SPLIT  ?= data/splits-cc-trial
CC_REPORT ?= /tmp/cc-trial/validation-report.json
CC_BENIGN ?= $(CC_OUT)
# Hosted-benign stratum (Amendment C): per-platform suffix queries over the
# same pinned crawls, target 2,000 rows equally per suffix. Fetch once
# (journaled, resumable), then select appends it on top of the main quotas.
CC_HOSTED_CACHE  ?= data/raw/cc-hosted-CC-MAIN-2026-34.json
CC_HOSTED_TARGET ?= 2000
ATHENA_OUTPUT    ?= s3://phishnet-athena/hosted/

## Phase 3 three-band population (recorded decision, not yet run: needs the
## pinned enlarged corpus in data/raw first). T2 is the splits-eval cutoff
## so the threshold-transfer verdict differs from Phase 2 only in its
## calibration slice; 30/20/50 benign buckets is option-1 power sizing.
## Always with --phase3 (provenance columns) and --deterministic-manifest.
P3_T1    ?= 2026-07-25T00:00:00+00:00
P3_T2    ?= 2026-08-22T00:00:00+00:00
P3_TFRAC ?= 0.5
P3_CFRAC ?= 0.2
P3_OUT   ?= data/splits-p3

.PHONY: report collect split eval-split baseline eval canary test clean cc-select cc-fetch-hosted cc-validate cc-gates p3-split

p3-split:
	uv run python build_splits.py --phase3 --deterministic-manifest \
		--calib-date "$(P3_T1)" --split-date "$(P3_T2)" \
		--benign-test-fraction $(P3_TFRAC) --benign-calib-fraction $(P3_CFRAC) \
		--raw data/raw --out $(P3_OUT)

cc-select:
	python build_cc_benign.py --phase select --target-n $(CC_TARGET) --measure-quotas-from data/raw --exclude-phishing-tenants-from data/raw --require-multi-crawl-hosted --hosted-cache $(CC_HOSTED_CACHE) --hosted-target-n $(CC_HOSTED_TARGET) --out $(CC_OUT)

cc-fetch-hosted:
	python build_cc_benign.py --phase fetch-hosted --athena-output $(ATHENA_OUTPUT) --hosted-cache $(CC_HOSTED_CACHE)

cc-validate:
	python validate_cc_benign.py --benign $(CC_BENIGN) --split-dir $(CC_SPLIT) --out $(CC_REPORT)

cc-gates: cc-select cc-validate

## the deliverable: rebuild splits from the raw log and re-run the frozen baseline
report: split baseline

collect:
	python collect.py --phish --benign --tranco-id $(TRANCO_ID) \
		--benign-domains $(DOMAINS) --per-domain $(PER_DOMAIN)

split:
	python build_splits.py --test-days $(TEST_DAYS)

## successor evaluation population (frozen splits are never overwritten).
## Stages the manifest-exact input set, builds deterministically
## (--deterministic-manifest moves the run timestamp to run-meta.json, so
## train.csv/test.csv/manifest.json are byte-stable), then verifies the
## pinned hashes in repro/hashes.json.
# uv run (not bare python): CI runners have no project packages on PATH,
# so only the synced environment resolves the locked dependencies.
eval-split:
	rm -rf "$(OUT)-staging" && mkdir -p "$(OUT)-staging" \
		&& cp $(addprefix data/raw/,$(EVAL_RAW_FILES)) "$(OUT)-staging"/
	uv run python build_splits.py --split-date "$(SPLIT_DATE)" \
		--benign-test-fraction $(EVAL_FRAC) --raw "$(OUT)-staging" \
		--out $(OUT) --deterministic-manifest
	rm -rf "$(OUT)-staging"
	uv run python repro/verify.py --hashes repro/hashes.json --dir $(OUT)

## freeze the current model's number on the new test set
baseline: $(TEST)
	python eval.py --predictor predictors:LegacyEnsemble --dataset $(TEST) --tag baseline

## evaluate anything else against that frozen number
eval: $(TEST)
	python eval.py --predictor $(PRED) --dataset $(TEST) --compare $(BASELINE)

## dataset sanity: a model that only knows URL length and path depth
canary: $(TEST)
	python eval.py --predictor predictors:UrlShapeHeuristic --dataset $(TEST) --tag canary
	python eval.py --predictor predictors:RandomScorer --dataset $(TEST) --tag random

test:
	python -m pytest test_eval.py -q

clean:
	rm -rf reports/*.json reports/*.md data/splits
