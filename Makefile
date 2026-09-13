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

.PHONY: report collect split eval-split baseline eval canary test clean

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
eval-split:
	rm -rf "$(OUT)-staging" && mkdir -p "$(OUT)-staging" \
		&& cp $(addprefix data/raw/,$(EVAL_RAW_FILES)) "$(OUT)-staging"/
	python build_splits.py --split-date "$(SPLIT_DATE)" \
		--benign-test-fraction $(EVAL_FRAC) --raw "$(OUT)-staging" \
		--out $(OUT) --deterministic-manifest
	rm -rf "$(OUT)-staging"
	python repro/verify.py --hashes repro/hashes.json --dir $(OUT)

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
