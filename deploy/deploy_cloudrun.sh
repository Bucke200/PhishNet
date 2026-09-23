#!/usr/bin/env bash
# PhishNet Cloud Run deploy (deployment-plan §7 runbook, executable form).
#
# Deploys the two-service split — phishnet-fetcher then phishnet-serving —
# to Google Cloud Run on the permanent free tier ($0.00/mo modelled at
# <1% of quota). Idempotent: safe to re-run; Artifact Registry, secrets,
# and services are created-or-updated, never duplicated.
#
# Prerequisites: gcloud CLI (authenticated), docker, GROQ_API_KEY in env.
#   export PROJECT_ID=phishnet-prod REGION=us-central1 GROQ_API_KEY=gsk_...
#   ./deploy/deploy_cloudrun.sh
#
# GROQ_API_KEY is stored in Secret Manager (secret `groq-api-key`) and
# mounted via --set-secrets — never via plaintext --set-env-vars.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-phishnet-prod}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-phishnet-repo}"
TAG="${TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUDRUN_DIR="${SCRIPT_DIR}/cloudrun"

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is required in the environment (never committed)." >&2
  exit 1
fi

echo "==> project=${PROJECT_ID} region=${REGION} repo=${REPO} tag=${TAG}"
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com >/dev/null

# --- Artifact Registry (create-or-keep) -------------------------------------
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="PhishNet container images"
fi
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

SERVING_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/serving:${TAG}"
FETCHER_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/fetcher:${TAG}"

# --- Build & push (model weights baked at build time, §4.2) -----------------
docker build -f backend/fetcher/Dockerfile -t "${FETCHER_IMAGE}" .
docker push "${FETCHER_IMAGE}"

docker build -f backend/Dockerfile -t "${SERVING_IMAGE}" .
docker push "${SERVING_IMAGE}"

# --- Secret Manager (create-or-new-version, no plaintext in YAML/CLI) -------
if ! gcloud secrets describe groq-api-key >/dev/null 2>&1; then
  printf '%s' "${GROQ_API_KEY}" | gcloud secrets create groq-api-key --data-file=-
else
  printf '%s' "${GROQ_API_KEY}" | gcloud secrets versions add groq-api-key --data-file=-
fi

# --- Service 2: fetcher (§5.1: 1.5Gi/1vCPU, concurrency 10, 0..2) ------------
gcloud run deploy phishnet-fetcher \
  --image="${FETCHER_IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --memory=1.5Gi \
  --cpu=1 \
  --concurrency=10 \
  --min-instances=0 \
  --max-instances=2 \
  --timeout=60 \
  --allow-unauthenticated \
  --set-env-vars="PHISHNET_FETCH_TIMEOUT=8,PHISHNET_FETCHER_RENDER=1,PYTHONUNBUFFERED=1"

FETCHER_URL="$(gcloud run services describe phishnet-fetcher \
  --region="${REGION}" --format='value(status.url)')/fetch"
echo "==> fetcher: ${FETCHER_URL}"

# --- Service 1: serving (§4.1: 512Mi/1vCPU, concurrency 80, 0..2) -------------
gcloud run deploy phishnet-serving \
  --image="${SERVING_IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --memory=512Mi \
  --cpu=1 \
  --concurrency=80 \
  --min-instances=0 \
  --max-instances=2 \
  --timeout=30 \
  --allow-unauthenticated \
  --set-env-vars="PHISHNET_TIER2_MODE=live,PHISHNET_TIER2_FAILURE_POLICY=mechanism,PHISHNET_FETCHER_URL=${FETCHER_URL},PHISHNET_EXTENSION_ID=cphacgebncakdmjbpoibajnihhbbcjec,PHISHNET_ML_ASSETS_DIR=/app/models,PHISHNET_THRESHOLDS_FILE=/app/reports/phase4.json,PYTHONUNBUFFERED=1" \
  --set-secrets="GROQ_API_KEY=groq-api-key:latest"

SERVING_URL="$(gcloud run services describe phishnet-serving \
  --region="${REGION}" --format='value(status.url)')"
echo "==> serving: ${SERVING_URL}"
echo "==> next: SERVING_URL=${SERVING_URL} ./deploy/smoke.sh"
echo "==> then paste ${SERVING_URL} into the extension options page."
