#!/bin/bash
# =============================================================================
# Phase 43: SafeClaw Keychain → Shell Environment Loader
#
# SOURCE this file (don't run it directly):
#   source scripts/load_secrets.sh
#
# It exports all API keys from macOS Keychain into the current shell session.
# Falls back gracefully to .env values for CI / non-Mac environments.
# Non-secret config (HUB_URL, PORT) still comes from .env.
# =============================================================================

SERVICE="safeclaw"

# Pull one key from keychain, with graceful fallback to .env
_load_key() {
    local keychain_key="$1"
    local env_var="$2"

    # Skip if already set (e.g. explicitly exported before sourcing this file)
    if [ -n "${!env_var}" ]; then
        return
    fi

    local val
    val=$(security find-generic-password -a "$keychain_key" -s "$SERVICE" -w 2>/dev/null || echo "")

    if [ -z "$val" ]; then
        # Fallback: read from .env for CI / Docker / non-Mac
        val=$(grep "^${env_var}=" .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        if [ -n "$val" ]; then
            echo "   ⚠️  $env_var: keychain miss — loaded from .env (CI/fallback mode)" >&2
        else
            echo "   ❌ $env_var: not found in keychain or .env" >&2
        fi
    fi

    export "$env_var"="$val"
}

echo "🔑 Loading SafeClaw secrets from macOS Keychain..." >&2

_load_key "nebius_api_key"     "NEBIUS_API_KEY"
_load_key "telegram_bot_token" "TELEGRAM_BOT_TOKEN"
_load_key "github_token"       "GITHUB_TOKEN"
_load_key "google_api_key"     "GOOGLE_API_KEY"

# Non-secret config still sourced from .env
if [ -f .env ]; then
    # Only pull non-secret vars (lines NOT matching key patterns)
    while IFS='=' read -r key value ; do
        case "$key" in
            SAFECLAW_HUB_URL|SAFECLAW_HUB_PORT)
                # Value already extracted — strip both single and double quotes consistently
                val=$(echo "$value" | tr -d '"' | tr -d "'")
                export "$key"="$val"
                ;;
        esac
    done < .env
fi

# Defaults for config vars if still unset
export SAFECLAW_HUB_URL="${SAFECLAW_HUB_URL:-http://localhost:8000}"
export SAFECLAW_HUB_PORT="${SAFECLAW_HUB_PORT:-8000}"

echo "   ✅ Secrets loaded." >&2
