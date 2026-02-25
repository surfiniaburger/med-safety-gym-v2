#!/bin/bash
# Start SafeClaw Observability Hub (Governor)

set -a
source .env
set +a

PORT=${SAFECLAW_HUB_PORT:-8000}

echo "🏦 Starting SafeClaw Observability Hub on port $PORT..."
uv run uvicorn med_safety_eval.observability_hub:app --port "$PORT" --host 0.0.0.0
