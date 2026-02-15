#!/bin/bash
# Restart the bot (kills old process and starts fresh)

echo "🔄 Restarting SafeClaw Bot..."

# Kill any existing bot processes
pkill -f "med_safety_gym.telegram_bridge" || true
sleep 1

# Start fresh
./scripts/start_bot_quick.sh
