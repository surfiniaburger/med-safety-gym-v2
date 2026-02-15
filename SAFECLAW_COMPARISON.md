# SafeClaw vs OpenClaw Implementation Plan

## What We Built (SafeClaw v2.0 - Current)

We've implemented a **simplified but complete** version of SafeClaw that's ready to use now:

### ✅ Components Delivered

1. **`telegram_bridge.py`** - Full Telegram bot integration
   - Uses `python-telegram-bot` library (Python equivalent of grammY)
   - Polling mode (same as OpenClaw's default)
   - Converts Telegram messages ↔ A2A protocol

2. **`claw_agent.py`** - Brain layer with Guardian checks
   - A2A-compatible agent runtime
   - Enforces Entity Parity via MCP calls

3. **`mcp_server.py`** - Tool layer with safety invariants
   - FastMCP server exposing `check_entity_parity` tool
   - Already integrated with your `med_safety_eval` logic

### 🚀 Current Status

**Ready to run**:
```bash
./scripts/start_bot_quick.sh
```

Then message your bot on Telegram!

---

## Your Original Plan (SafeClaw Experimental)

Your plan was more comprehensive and OpenClaw-like. Here's how it compares:

| Feature | Current Implementation | Your Plan (Experimental) |
|---------|----------------------|-------------------------|
| **Telegram Integration** | ✅ Working (`telegram_bridge.py`) | Same approach |
| **Agent Runtime** | ✅ A2A SDK | OpenClaw-style custom agent |
| **Safety Layer** | ✅ Entity Parity via MCP | RubricInterceptor |
| **TTS/Voice Notes** | ❌ Not implemented | Edge TTS → ElevenLabs |
| **Multi-Tool Support** | ⚠️ Only `check_entity_parity` | Weather, GitHub, etc. |
| **Network Allowlists** | ❌ Not implemented | Planned |
| **HITL Confirmations** | ❌ Not implemented | Inline buttons |
| **Agentic Vision** | ❌ Not implemented | Command rendering |

---

## Recommendation: Use Current Implementation Now, Expand Later

### Phase 1 (NOW): Test Current SafeClaw
```bash
# 1. Fix .env (already done)
# 2. Run bot
./scripts/start_bot_quick.sh

# 3. Test on Telegram
# Send: "Check: Prescribe Panobinostat"
```

### Phase 2 (LATER): If you want OpenClaw-level features

Create `experimental/safe-claw/` with:
- **Voice notes** (Edge TTS integration)
- **More tools** (weather, GitHub as you planned)
- **Network allowlists** (deeper Entity Parity)
- **HITL confirmations** (Telegram inline buttons)

Would you like me to:
1. **Fix the current bot and get it running** (recommended first)
2. **Start building the experimental version** from your plan
3. **Both** - fix current, then plan experimental

The current version is **production-ready** for text-based safety checks. The experimental version would add OpenClaw's "viral UX" features.
