# SafeClaw Quickstart Guide

## Prerequisites

- Python 3.11+
- `uv` package manager ([install](https://docs.astral.sh/uv/))
- Nebius API Key (for Qwen model)
- Telegram Bot Token (optional, for production)

---

## Step 1: Environment Setup

Create a `.env` file in the project root:

```bash
# LLM Configuration
USER_LLM_MODEL="nebius/Qwen/Qwen3-235B-A22B-Thinking-2507"
NEBIUS_API_KEY="your-nebius-api-key-here"
MAX_TOKENS=4096

# Optional: Telegram Bot Token (for production deployment)
TELEGRAM_BOT_TOKEN="your-bot-token-here"

# SafeClaw Configuration
MCP_SERVER_COMMAND="uv"
MCP_SERVER_ARGS='["run", "python", "-m", "med_safety_gym.mcp_server"]'
SAFECLAW_PORT=8003
```

---

## Step 2: Install Dependencies

```bash
cd <project_root>
uv sync --extra dev --extra mcp --extra agent
```

---

## Step 3: Test the Tool Layer (MCP Server)

Verify the Entity Parity safety check works:

```bash
uv run pytest tests/test_mcp_entity_parity.py -v
```

Expected output: **3 passed** (Unknown Entity blocked, Known Entity allowed, Subset allowed)

---

## Step 4: Test the Integration Layer

Verify the A2A Agent can talk to the MCP Server:

```bash
uv run pytest tests/test_claw_agent_integration.py -v
```

Expected output: **1 passed** (Client spawns server, calls `check_entity_parity`)

---

## Step 5: Start SafeClaw Agent (Local HTTP Server)

Run the A2A server on `http://localhost:8003`:

```bash
uv run python -m med_safety_gym.claw_server
```

You should see:
```
INFO:     Started server process [xxxxx]
INFO:     Uvicorn running on http://0.0.0.0:8003 (Press CTRL+C to quit)
```

---

## Step 6: Test with a Mock A2A Request

In a separate terminal, send a test message:

```bash
curl -X POST http://localhost:8003/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "messageId": "test-1",
    "role": "user",
    "parts": [{"kind": "text", "text": "Verify safety of: Prescribe ScillyCure"}]
  }'
```

### Expected Behavior:
- Agent sends request to MCP server
- MCP server detects "ScillyCure" is **not** in context
- Agent **refuses** with: `🚫 Action BLOCKED by Guardian.`

Try a **safe** action:
```bash
curl -X POST http://localhost:8003/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "messageId": "test-2",
    "role": "user",
    "parts": [{"kind": "text", "text": "Check: Prescribe Panobinostat for DIPG patient"}]
  }'
```

This should **pass** if "Panobinostat" is in the context.

---

## Step 7: (Optional) Connect to Telegram

### 7.1 Register Your Bot with BotFather

1. Open Telegram, search for **@BotFather**
2. Send `/newbot` and follow prompts
3. Copy the **Bot Token** to `.env` as `TELEGRAM_BOT_TOKEN`

### 7.2 Update `claw_server.py` for Telegram

We'll need to add a Telegram adapter. For now, the HTTP endpoint works for testing.

**TODO**: Implement `TelegramMessageBridge` to convert Telegram messages → A2A protocol.

---

## Next Steps

### For Production Deployment:
1. **Add Telegram Bridge**: Listen to Telegram updates, convert to A2A `Message` format
2. **Deploy to Cloud**: Use Docker + Cloud Run or similar
3. **Add LLM Planning**: Currently the agent only validates actions. Add an LLM layer for autonomous planning.

### For Local Testing:
- Use the HTTP endpoint (Step 6) to simulate messages
- Mock different "contexts" to test Entity Parity edge cases
