#!/usr/bin/env bash
# PhishNet Cloud Run smoke test (deployment-plan §8).
# Verifies /health, /predict, /explain against a live serving URL.
#
#   SERVING_URL=https://phishnet-serving-xyz-uc.a.run.app ./deploy/smoke.sh
set -euo pipefail

SERVING_URL="${SERVING_URL:?set SERVING_URL to the phishnet-serving Cloud Run URL}"
BASE="${SERVING_URL%/}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "==> GET ${BASE}/health"
HEALTH="$(curl -fsS "${BASE}/health")" || fail "health unreachable"
echo "${HEALTH}"
echo "${HEALTH}" | python3 -c "
import json,sys
b=json.load(sys.stdin)
assert b['status']=='ok', b
assert len(b['model_hash'])==64 and len(b['columns_hash'])==64, b
assert b['n_columns']==79, b
assert b['tier2_mode']=='live', b
assert b['tier2_failure_policy']=='mechanism', b
print('health ok:', b['model_hash'][:12], b['thresholds_source'])
" || fail "health contract"

echo "==> POST ${BASE}/predict (benign)"
PREDICT="$(curl -fsS -X POST "${BASE}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.google.com"}')" || fail "predict unreachable"
echo "${PREDICT}"
echo "${PREDICT}" | python3 -c "
import json,sys
b=json.load(sys.stdin)
assert b['disposition'] in ('allow','alert',\"can't assess\"), b
assert isinstance(b['tier1_score'],(int,float)), b
print('predict ok:', b['disposition'], b['reason'])
" || fail "predict contract"

echo "==> POST ${BASE}/explain"
EXPLAIN="$(curl -fsS -X POST "${BASE}/explain" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.google.com","top_k":3}')" || fail "explain unreachable"
echo "${EXPLAIN}"
echo "${EXPLAIN}" | python3 -c "
import json,sys
b=json.load(sys.stdin)
feats=b['attribution']['features']
assert len(feats)==3, b
assert all(set(f)=={'feature','contribution'} for f in feats), b
print('explain ok:', [(f['feature'], round(f['contribution'],3)) for f in feats])
" || fail "explain contract"

echo "SMOKE PASS: ${BASE}"
