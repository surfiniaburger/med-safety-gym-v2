#!/bin/bash
# Setup and run SafeClaw Telegram Bot

set -e

echo "🤖 SafeClaw Telegram Bot - Setup"
echo "====================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Load environment variables properly (handle quotes and keychain)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_secrets.sh"

export $(cat .env | grep -v '^#' | grep -v 'TELEGRAM_BOT_TOKEN' | xargs)

# Check for Telegram token
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not set in .env!"
    echo ""
    echo "Please add to .env:"
    echo "  TELEGRAM_BOT_TOKEN=\"your-bot-token-from-botfather\""
    echo ""
    echo "To get a token:"
    echo "  1. Open Telegram and search for @BotFather"
    echo "  2. Send /newbot and follow instructions"
    echo "  3. Copy the token to your .env file"
    exit 1
fi

echo "✅ Bot token found: ${TELEGRAM_BOT_TOKEN:0:10}..."

# Check API keys
if [ "$NEBIUS_API_KEY" = "your-nebius-api-key-here" ] || [ -z "$NEBIUS_API_KEY" ]; then
    echo "⚠️  Warning: NEBIUS_API_KEY not properly configured"
fi

echo "📦 Installing dependencies (this may take a minute)..."
uv sync --extra agent --extra mcp --extra telegram --extra dev --quiet

echo "🧪 Quick safety check..."
uv run pytest tests/test_mcp_entity_parity.py tests/test_claw_agent_brain.py -q || true

echo ""
echo "✅ Setup complete!"
echo "🚀 Starting SafeClaw Telegram Bot..."
echo "   Bot username: Search for your bot on Telegram!"
echo "   Press Ctrl+C to stop"
echo ""

uv run python -m med_safety_gym.telegram_bridge
