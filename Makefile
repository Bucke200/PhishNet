TRANCO_ID   ?= NONE
DOMAINS     ?= 400
PER_DOMAIN  ?= 10
TEST_DAYS   ?= 21
PRED        ?= predictors:LegacyEnsemble
TEST        := data/splits/test.csv
BASELINE    := reports/baseline.json

.PHONY: report collect split baseline eval canary test clean

## the deliverable: rebuild splits from the raw log and re-run the frozen baseline
report: split baseline

collect:
	python collect.py --phish --benign --tranco-id $(TRANCO_ID) \
		--benign-domains $(DOMAINS) --per-domain $(PER_DOMAIN)

split:
	python build_splits.py --test-days $(TEST_DAYS)

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
