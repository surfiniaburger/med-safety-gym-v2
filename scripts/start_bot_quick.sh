#!/bin/bash
# Start SafeClaw Telegram Bot
# Phase 43: Secrets loaded from macOS Keychain via load_secrets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_secrets.sh"

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not found in keychain or .env"
    exit 1
fi

echo "🤖 Starting SafeClaw Telegram Bot..."
echo "   Token: ${TELEGRAM_BOT_TOKEN:0:10}..."

# 1. Ensure Dependencies
uv sync --extra agent --extra mcp --extra telegram --extra dev --quiet

# 2. Check if Hub is running, start if not
HUB_PORT=${SAFECLAW_HUB_PORT:-8000}
if ! curl -s "http://localhost:$HUB_PORT/health" > /dev/null; then
    echo "🏦 SafeClaw Hub not found. Starting in background..."
    nohup uv run uvicorn med_safety_eval.observability_hub:app --port "$HUB_PORT" > hub.log 2>&1 &
    # Wait for hub to be ready
    for i in {1..10}; do
        if curl -s "http://localhost:$HUB_PORT/health" > /dev/null; then
            echo "✅ Hub is responsive."
            break
        fi
        sleep 1
    done
else
    echo "✅ SafeClaw Hub is already running."
fi

# 3. Run the Bot
echo "🚀 Launching Telegram Bridge..."
uv run python -m med_safety_gym.telegram_bridge
