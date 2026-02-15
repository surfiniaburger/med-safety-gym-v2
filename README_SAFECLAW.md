# SafeClaw - Quick Start (Simplified)

You have **3 ways** to test SafeClaw locally:

## Option 1: Direct Agent Test (Unit Level) ✅ VERIFIED

Test the agent logic without starting a server:

```bash
cd /Users/surfiniaburger/Desktop/med-safety-gym-v2

# Test Entity Parity (Tool Layer)
uv run pytest tests/test_mcp_entity_parity.py -v

# Test Integration (Bridge Layer)
uv run pytest tests/test_claw_agent_integration.py -v

# Test Agent Logic (Brain Layer)  
uv run pytest tests/test_claw_agent_brain.py -v
```

**Status**: ✅ All 6 tests passing

---

## Option 2: Start the A2A Server (For HTTP Testing)

Start SafeClaw as an HTTP server on port 8003:

```bash
uv run python -m med_safety_gym.claw_server
```

You should see:
```
🤖 SafeClaw Agent serving on 0.0.0.0:8003
🛡️  Guardian safety checks: ENABLED
INFO:     Uvicorn running on http://0.0.0.0:8003
```

### Test with curl:

```bash
# Get Agent Card
curl http://localhost:8003/

# Send a task (A2A protocol requires a specific JSON-RPC format)
# The exact format depends on A2A SDK version - check green_agent examples
```

---

## Option 3: Telegram Integration ✅ READY

SafeClaw can now run as a Telegram bot!

### Setup (5 minutes)

1. **Create your bot with BotFather**:
   ```
   Open Telegram → Search @BotFather → Send /newbot
   ```

2. **Add token to .env**:
   ```bash
   echo 'TELEGRAM_BOT_TOKEN="your-token-here"' >> .env
   ```

3. **Run the bot**:
   ```bash
   ./scripts/run_telegram_bot.sh
   ```

4. **Test on Telegram**:
   - Search for your bot
   - Send `/start`
   - Try: `Check: Prescribe Panobinostat for DIPG`

### What It Does

The Telegram bridge:
- ✅ Converts Telegram messages → A2A protocol
- ✅ Routes through SafeClawAgent
- ✅ Enforces Entity Parity via MCP
- ✅ Sends Guardian responses back to Telegram

See [`TELEGRAM_SETUP.md`](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/TELEGRAM_SETUP.md) for detailed instructions.

---

## Current Status

| Component | Status | Test Command |
|-----------|--------|--------------|
| **Tool Layer** (Entity Parity) | ✅ Working | `uv run pytest tests/test_mcp_entity_parity.py` |
| **Bridge Layer** (MCP Client) | ✅ Working | `uv run pytest tests/test_claw_agent_integration.py` |
| **Brain Layer** (Agent Logic) | ✅ Working | `uv run pytest tests/test_claw_agent_brain.py` |
| **A2A HTTP Server** | ⚠️ Runs, needs A2A client | `uv run python -m med_safety_gym.claw_server` |
| **Telegram Integration** | ✅ Working | `./scripts/run_telegram_bot.sh` |

---

## Next Steps

**Current Status**: ✅ **Fully Functional**

SafeClaw now supports:
- ✅ Local testing (demo script)
- ✅ Unit testing (all layers)
- ✅ Telegram bot deployment

**Optional Enhancements**:
1. **Richer Medical Context**: Load a medical knowledge base for better entity validation
2. **Conversation History**: Track context across Telegram message history
3. **Webhook Mode**: Switch from polling to webhooks for production
4. **Multi-Agent**: Connect SafeClaw to other A2A agents for complex workflows

**Ready to use**: Just run `./scripts/run_telegram_bot.sh` and start chatting! 🎉
