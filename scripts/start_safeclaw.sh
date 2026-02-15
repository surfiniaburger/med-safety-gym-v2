#!/bin/bash
# SafeClaw Startup Script

set -e

echo "🤖 SafeClaw Agent - Startup"
echo "============================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📋 Creating template .env..."
    cat > .env << EOF
# LLM Configuration
USER_LLM_MODEL="nebius/Qwen/Qwen3-235B-A22B-Thinking-2507"
NEBIUS_API_KEY="your-nebius-api-key-here"
MAX_TOKENS=4096

# SafeClaw Configuration
SAFECLAW_PORT=8003
EOF
    echo "✅ .env created. Please edit it with your API key."
    exit 1
fi

# Load environment
export $(cat .env | grep -v '^#' | xargs)

# Check API key
if [ "$NEBIUS_API_KEY" = "your-nebius-api-key-here" ]; then
    echo "❌ Please set your NEBIUS_API_KEY in .env"
    exit 1
fi

echo "✅ Environment loaded"
echo "📦 Installing dependencies..."
uv sync --extra dev --extra mcp --extra agent --quiet

echo "🧪 Running safety verification tests..."
uv run pytest tests/test_mcp_entity_parity.py tests/test_claw_agent_brain.py -q

echo "🚀 Starting SafeClaw Agent on port ${SAFECLAW_PORT:-8003}..."
uv run python -m med_safety_gym.claw_server
