#!/bin/bash
# Start SafeClaw Observability Hub (Governor)
# Phase 43: Secrets loaded from macOS Keychain via load_secrets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_secrets.sh"

PORT=${SAFECLAW_HUB_PORT:-8000}

echo "🏦 Starting SafeClaw Observability Hub on port $PORT..."
uv run uvicorn med_safety_eval.observability_hub:app --port "$PORT" --host 0.0.0.0
