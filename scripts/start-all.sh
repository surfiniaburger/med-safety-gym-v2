#!/bin/bash
# Start SafeClaw Ecosystem (Hub, Agent, Bot) via Docker Compose
# Phase 43: Secrets loaded from macOS Keychain — never from a file on disk

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_secrets.sh"

# Validate required secrets are present
if [ -z "$NEBIUS_API_KEY" ]; then
    echo "❌ NEBIUS_API_KEY not found in keychain. Run: bash scripts/store_secrets.sh"
    exit 1
fi
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not found in keychain. Run: bash scripts/store_secrets.sh"
    exit 1
fi

echo "🚀 Starting SafeClaw Ecosystem (Hub, Agent, Bot) via Docker Compose..."

# Inject secrets from keychain at runtime — they are NEVER written to disk
NEBIUS_API_KEY="$NEBIUS_API_KEY" \
TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
docker-compose up -d --build

echo "✅ All core services started."
echo "👉 Governor Hub running on port 8080 (http://localhost:8080)"
echo "🔐 Runners (Agent, Bot) are executing in zero-trust isolated network mode."
echo "📄 Run 'docker-compose logs -f' to monitor the system."
