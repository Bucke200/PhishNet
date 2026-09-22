# PhishNet Tier-2 fetcher image (Phase 6 decision 2, opt-in live mode).
#
# Build from the repository root:
#   docker build -f backend/fetcher/Dockerfile -t phishnet-fetcher .
#
# Kept separate from the Tier-1 image so serving latency is measured without
# a browser in the process, and so the pinned Tier-1 runtime stays minimal.

FROM python:3.10-slim

WORKDIR /app

# The fetcher needs only the extractor and HTTP; it does not install the
# Tier-1 runtime (pandas/lightgbm).
RUN pip install --no-cache-dir \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.24.0" \
    "pydantic[email]==2.11.7" \
    "beautifulsoup4==4.15.0" \
    "requests==2.32.3" \
    "playwright"
RUN playwright install --with-deps chromium

COPY src ./src
ENV PYTHONPATH="/app/src"

EXPOSE 8100

# Bind to $PORT when the host provides one (Render), else 8100.
CMD ["sh", "-c", "uvicorn phishnet.fetcher.app:app --host 0.0.0.0 --port ${PORT:-8100}"]
