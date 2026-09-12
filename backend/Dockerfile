# PhishNet backend image (uv-based dependencies).
# Build from the repository root so pyproject.toml + uv.lock are in context:
#   docker build -f backend/Dockerfile -t phishnet-backend .
# (A backend/-only context cannot see the root dependency files.)

# Use official Python image (matches requires-python >= 3.10)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv without depending on pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition first for layer caching.
# README.md is required to build the phishnet project itself (hatchling).
COPY pyproject.toml uv.lock README.md ./

# Copy the packaged source (needed to install the phishnet project)
COPY src ./src

# Install the locked runtime environment (no dev tooling in the image)
RUN uv sync --locked --no-dev

# Copy the backend runtime files
COPY backend ./backend

# Use the environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Model assets are fetched at container start by the verified downloader
# (`python -m phishnet.verified_download`: GitHub Release -> SHA256 verify
# -> atomic install). Point the packaged app at them so `phishnet.api`
# does not depend on the current working directory.
ENV PHISHNET_ML_ASSETS_DIR="/app/backend/urlset_ml_assets"

# Serve from backend/ (historic runtime layout; the verified downloader
# resolves its destination via $PHISHNET_ML_ASSETS_DIR, not via cwd).
WORKDIR /app/backend

# Expose the FastAPI port
EXPOSE 8000

# Start the packaged FastAPI app with Uvicorn
CMD ["sh", "-c", "python -m phishnet.verified_download && uvicorn phishnet.api:app --host 0.0.0.0 --port 8000"]
