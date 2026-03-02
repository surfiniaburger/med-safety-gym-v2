# SafeClaw Startup Guide

This guide documents the exact procedure for starting the SafeClaw ecosystem (Server and Telegram Bot) using the macOS Keychain for secrets.

## Prerequisites
- **macOS Keychain**: Ensure `nebius_api_key` and `telegram_bot_token` are stored under the `safeclaw` service.
  - Run `bash scripts/store_secrets.sh` if they are missing.

## Quick Start Procedure

### 1. Clear Existing Processes
Ensure port `8003` is free and stop any running bot instances:
```bash
lsof -i :${SAFECLAW_PORT:-8003} -t | xargs kill -9 || true
pkill -f "med_safety_gym.telegram_bridge" || true
```

### 2. Start the SafeClaw Agent Server
Run this in a background terminal. It sources the keychain secrets and starts the FastAPI server:
```bash
bash -c "source scripts/load_secrets.sh && bash scripts/start_safeclaw.sh"
```
*Port: 8003*

### 3. Start the Telegram Bot Bridge
Run this in another background terminal:
```bash
bash -c "source scripts/load_secrets.sh && bash scripts/run_telegram_bot.sh"
```

## Why this works
- **`load_secrets.sh`**: Exports the verified keys from your Keychain into the environment without writing them to disk.
- **`start_safeclaw.sh`**: Runs safety verification tests before binding the server.
- **`run_telegram_bot.sh`**: Initializes the bridge between Telegram and the Agent Server.

---
> [!TIP]
> If you encounter `bad substitution` errors, ensure you are running these commands in a `bash` shell (the environment loader uses bash-specific syntax).
