# PhishNet Tier-2 fetcher image (Phase 6 decision 2, opt-in live mode).
#
# Build from the repository root:
#   docker build -f backend/fetcher/Dockerfile -t phishnet-fetcher .
#
# Kept separate from the Tier-1 image so serving latency is measured without
# a browser in the process, and so the pinned Tier-1 runtime stays minimal.
# Production (docs/deployment-plan.md §5): 1.5 GiB / 1 vCPU on Cloud Run,
# concurrency 10, Chromium tuned via CHROMIUM_ARGS in phishnet.fetcher.app.

FROM python:3.10-slim

WORKDIR /app

# The fetcher needs only the extractor and HTTP; it does not install the
# Tier-1 runtime (pandas/lightgbm).
# Pinned to pyproject.toml versions (single source of truth for the shared
# stack); playwright is fetcher-only and floats on its own release line.
RUN pip install --no-cache-dir \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.24.0" \
    "pydantic[email]==2.11.7" \
    "beautifulsoup4==4.15.0" \
    "requests==2.32.3" \
    "playwright" \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY src ./src
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    PHISHNET_FETCH_TIMEOUT="8" \
    PHISHNET_FETCHER_RENDER="1"

EXPOSE 8100

# Bind to $PORT when the host provides one (Cloud Run injects PORT=8080),
# else 8100. Single worker: each request launches one short-lived Chromium
# (CHROMIUM_ARGS); Cloud Run scales via concurrency, not forked workers.
# Runs as root: Chromium's --no-sandbox + --single-process path expects
# root-owned browser deps; Cloud Run's gVisor sandbox provides isolation.
CMD ["sh", "-c", "uvicorn phishnet.fetcher.app:app --host 0.0.0.0 --port ${PORT:-8100} --workers 1"]
