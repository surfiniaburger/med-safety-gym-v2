#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
  source .env
else
  echo "⚠️  .env file not found! Docker Compose might miss NEBIUS_API_KEY or TELEGRAM_BOT_TOKEN."
fi

echo "🚀 Starting SafeClaw Ecosystem (Hub, Agent, Bot) via Docker Compose..."

# Build and start in detached mode
docker-compose up -d --build

echo "✅ All core services started."
echo "👉 Governor Hub running on port 8080 (http://localhost:8080)"
echo "🔐 Runners (Agent, Bot) are executing in zero-trust isolated network mode."
echo "📄 Run 'docker-compose logs -f' to monitor the system."
