#!/bin/bash
# =============================================================================
# Phase 43: SafeClaw Secret Seeding Tool
# One-time setup: stores API keys in macOS Keychain (never written to disk).
#
# Usage:
#   bash scripts/store_secrets.sh
#
# To verify a stored key afterwards:
#   security find-generic-password -a nebius_api_key -s safeclaw -w
# =============================================================================

set -e

SERVICE="safeclaw"

echo "🔐 SafeClaw Phase 43 — Seeding API Keys into macOS Keychain"
echo "   Service name : $SERVICE"
echo "   Each key will be stored securely. Input is hidden (not echoed)."
echo ""

store_secret() {
    local key="$1"
    local label="$2"

    echo -n "Enter $label (hidden, Enter to skip): "
    read -r -s value
    echo  # newline after hidden input

    if [ -z "$value" ]; then
        echo "   ⏭️  Skipping $label (empty input)"
        return
    fi

    # -U = update if exists, -a = account, -s = service, -w = password
    security add-generic-password -U \
        -a "$key" \
        -s "$SERVICE" \
        -w "$value" \
        -l "SafeClaw: $label" \
        -j "Stored by store_secrets.sh on $(date)"

    echo "   ✅ Stored $label in keychain"
}

store_secret "nebius_api_key"     "NEBIUS_API_KEY"
store_secret "telegram_bot_token" "TELEGRAM_BOT_TOKEN"
store_secret "github_token"       "GITHUB_TOKEN"
store_secret "google_api_key"     "GOOGLE_API_KEY"

echo ""
echo "✅ Done! Keys are secured in macOS Keychain under service '$SERVICE'."
echo ""
echo "📋 Verify with:"
echo "   security find-generic-password -a nebius_api_key     -s $SERVICE -w"
echo "   security find-generic-password -a telegram_bot_token -s $SERVICE -w"
echo "   security find-generic-password -a github_token        -s $SERVICE -w"
echo "   security find-generic-password -a google_api_key      -s $SERVICE -w"
echo ""
echo "🔒 You can now remove the secret values from your .env file."
echo "   Keep only: SAFECLAW_HUB_URL and SAFECLAW_HUB_PORT"
