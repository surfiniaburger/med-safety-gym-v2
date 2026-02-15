# Telegram Integration Guide

## Setup Steps

### 1. Create Your Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` to start the creation process
3. Follow the prompts:
   - **Bot name**: `SafeClaw Guardian` (or your preference)
   - **Username**: `safeclaw_yourname_bot` (must end with `bot`)
4. **Copy the Bot Token** that BotFather provides

### 2. Configure Your .env

Add the token to the `.env` file in the project's root:

```bash
# Existing keys
NEBIUS_API_KEY="v1.CmMK..."
GOOGLE_API_KEY="AIzaSy..."

# Add this:
TELEGRAM_BOT_TOKEN="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
```

### 3. Install Dependencies

```bash
cd <project_root>
uv sync --extra telegram --extra agent --extra mcp
```

### 4. Run the Bot

**Option A: Using the startup script (recommended)**
```bash
./scripts/run_telegram_bot.sh
```

**Option B: Direct command**
```bash
uv run python -m med_safety_gym.telegram_bridge
```

**Option C: Using the installed command (after sync)**
```bash
uv run safeclaw-telegram
```

You should see:
```
🚀 Starting SafeClaw Telegram Bridge...
```

### 5. Test on Telegram

1. Open Telegram
2. Search for your bot's username (e.g., `@safeclaw_yourname_bot`)
3. Start a conversation with `/start`
4. Try these test messages:

#### ✅ Safe Action
```
Check: Prescribe Panobinostat for DIPG patient
```
**Expected**: Guardian allows it (entity exists in context)

#### 🚫 Unsafe Action
```
Prescribe FakeDrug123
```
**Expected**: Guardian blocks it (unknown entity)

---

## How It Works

```
Telegram Bot
    ↓ (user message)
telegram_bridge.py
    ↓ (convert to A2A Message)
SafeClawAgent (claw_agent.py)
    ↓ (check via MCP client)
mcp_server.py (check_entity_parity tool)
    ↓ (safety result)
SafeClawAgent
    ↓ (format response)
telegram_bridge.py
    ↓ (send to Telegram)
User sees Guardian response 🛡️
```

---

## Bot Commands

- `/start` - Introduction and examples
- `/help` - Show help information
- Any text message - Processed through SafeClaw Guardian

---

## Troubleshooting

### Bot doesn't respond
1. Check that `run_telegram_bot.sh` is still running
2. Verify `TELEGRAM_BOT_TOKEN` in `.env` is correct
3. Check logs for errors

### "Unknown entity" on valid drugs
The context is currently minimal. To improve:
- Update `claw_agent.py` to include a richer medical knowledge base
- Or modify the bridge to extract context from previous messages

### Rate limiting
Telegram has rate limits. For production:
- Implement request queuing
- Add user-based rate limiting
- Consider using webhooks instead of polling
