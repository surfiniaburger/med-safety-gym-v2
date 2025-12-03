#!/usr/bin/env bash
# Run the Ministral cloud models benchmark using the project's virtual environment.
# This wrapper ensures the correct Python environment and required packages are active.

set -euo pipefail

# Locate the project root (directory of this script's parent)
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Activate the virtual environment (or create it if missing)
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "Virtual environment not found at $PROJECT_ROOT/.venv. Creating one..."
  python3 -m venv "$PROJECT_ROOT/.venv"
  source "$PROJECT_ROOT/.venv/bin/activate"
  # Ensure pip is available in the new venv
  "$PROJECT_ROOT/.venv/bin/python" -m ensurepip --upgrade
  # Install core dependencies from pyproject.toml (frozen lockfile)
  uv sync --frozen
fi

# Use the venv's pip explicitly (fallback to ensurepip if missing)
PIP_CMD="$PROJECT_ROOT/.venv/bin/pip"
if ! command -v "$PIP_CMD" >/dev/null 2>&1; then
  echo "pip not found in venv, installing via ensurepip..."
  "$PROJECT_ROOT/.venv/bin/python" -m ensurepip --upgrade
fi

# Install benchmark-specific dependencies (quiet mode)
"$PROJECT_ROOT/.venv/bin/python" -m pip install -q a2a-sdk matplotlib litellm httpx

# Run the benchmark script using the venv's python
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/benchmark_ministral_models.py" "$@"
