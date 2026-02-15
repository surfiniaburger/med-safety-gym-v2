#!/bin/bash
# Quick fix: Run Telegram bot directly without tests

set -a
source .env
set +a

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not found in .env"
    exit 1
fi

echo "🤖 Starting SafeClaw Telegram Bot..."
echo "   Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo ""

# Install dependencies first
uv sync --extra agent --extra mcp --extra telegram --extra dev

# Run the bot
uv run python -m med_safety_gym.telegram_bridge
