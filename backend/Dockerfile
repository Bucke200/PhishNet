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

# Copy the backend runtime files (legacy entrypoint, model downloader)
COPY backend ./backend

# Use the environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Serve from backend/ so the existing entrypoint keeps working:
# download_models.py writes ./urlset_ml_assets and `main:app` imports locally.
WORKDIR /app/backend

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app with Uvicorn (unchanged entrypoint)
CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
