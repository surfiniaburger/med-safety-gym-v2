# Use the official Python image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the project into /app
COPY . /app
WORKDIR /app

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=1

# Install git, which is required for installing packages from git repositories
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN uv sync --frozen

EXPOSE 8080

# Run the FastAPI server (using shell form to expand $PORT)
CMD uv run uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8080}