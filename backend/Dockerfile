# PhishNet Tier-1 serving image (Phase 6): the Phase 3 row (a) LightGBM.
#
# Build from the repository root so pyproject.toml + uv.lock are in context:
#   docker build -f backend/Dockerfile -t phishnet-serving .
#
# This image is deliberately slim: pandas is used only by the eval-mode
# reference path, never at serving, and Playwright lives in the separate
# fetcher image (backend/fetcher/Dockerfile), so the latency number is not
# measured next to a browser.

FROM python:3.10-slim

WORKDIR /app

# LightGBM loads libgomp.so.1 (OpenMP runtime); slim images omit it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv without depending on pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependency definition first for layer caching. README.md is required to
# build the phishnet project itself (hatchling).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# Serving runtime inputs: the frozen thresholds (C1 verifies them at startup)
# and the sealed Phase 5 verdicts behind the offline demo Tier 2.
COPY reports/phase4.json /app/reports/phase4.json
COPY reports/adversarial-manifest-p5.json /app/reports/adversarial-manifest-p5.json
COPY runs/phase5 /app/runs/phase5

ENV PATH="/app/.venv/bin:$PATH"
# Row (a) artifacts are SHA256-verified into this dir at container start.
ENV PHISHNET_ML_ASSETS_DIR="/app/models"
ENV PHISHNET_THRESHOLDS_FILE="/app/reports/phase4.json"
ENV PHISHNET_TIER2_MODE="sealed"
# CORS allowlist: the extension manifest key pins this ID (phase6-E).
ENV PHISHNET_EXTENSION_ID="cphacgebncakdmjbpoibajnihhbbcjec"

EXPOSE 8000

# Fetch only the two row (a) artifacts (not the 93 MB legacy urlset
# ensemble), then serve phishnet.serving.app. Binds to $PORT when the host
# provides one (Render, Heroku, Cloud Run), else 8000.
CMD ["sh", "-c", "python -m phishnet.verified_download --only ablation_lexical_gbm_model.pkl --only ablation_lexical_feature_columns.pkl && uvicorn phishnet.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
