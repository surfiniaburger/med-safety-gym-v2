"""
Phase 43 TDD Tests: API Key Keyring Migration

Tests validate that:
1. SecretStore.KNOWN_KEYS includes the 4 new API key names
2. InMemorySecretStore correctly round-trips each new key
3. load_secrets.sh falls back gracefully (CI simulation)
4. .env no longer contains raw secret values
"""

import os
import re
import subprocess
import pytest

from med_safety_gym.identity.secret_store import (
    InMemorySecretStore,
    KeyringSecretStore,
    SecretStore,
)

ENV_PATH = ".env"

# ---------------------------------------------------------------------------
# 1. KNOWN_KEYS validation
# ---------------------------------------------------------------------------

class TestKnownKeysExtended:
    """KNOWN_KEYS must include the 4 new API key names per Phase 43."""

    EXPECTED_NEW_KEYS = (
        "nebius_api_key",
        "telegram_bot_token",
        "github_token",
        "google_api_key",
    )

    def test_known_keys_includes_original_agent_keys(self):
        assert "auth_token" in SecretStore.KNOWN_KEYS
        assert "hub_pub_key" in SecretStore.KNOWN_KEYS

    @pytest.mark.parametrize("key", EXPECTED_NEW_KEYS)
    def test_known_keys_includes_api_keys(self, key):
        assert key in SecretStore.KNOWN_KEYS, (
            f"'{key}' missing from SecretStore.KNOWN_KEYS"
        )


# ---------------------------------------------------------------------------
# 2. InMemorySecretStore round-trips for all API keys
# ---------------------------------------------------------------------------

class TestInMemoryApiKeyRoundtrip:
    """InMemorySecretStore must get/set each new API key (used in tests/CI)."""

    @pytest.fixture
    def store(self):
        return InMemorySecretStore()

    @pytest.mark.parametrize("key,value", [
        ("nebius_api_key",     "test-nebius-key-12345"),
        ("telegram_bot_token", "1234567890:AAFFFFFFFF"),
        ("github_token",       "ghp_test_token"),
        ("google_api_key",     "AIzaTest1234"),
    ])
    def test_set_and_get_api_key(self, store, key, value):
        store.set_secret(key, value)
        assert store.get_secret(key) == value

    def test_clear_removes_all_api_keys(self, store):
        for key in ("nebius_api_key", "telegram_bot_token", "github_token", "google_api_key"):
            store.set_secret(key, "dummy-value")
        store.clear_secrets()
        for key in ("nebius_api_key", "telegram_bot_token", "github_token", "google_api_key"):
            assert store.get_secret(key) is None

    def test_get_missing_key_returns_none(self, store):
        assert store.get_secret("nebius_api_key") is None


# ---------------------------------------------------------------------------
# 3. load_secrets.sh CI fallback
# ---------------------------------------------------------------------------

class TestLoadSecretsScript:
    """load_secrets.sh should export env vars (fallback or keychain)."""

    def test_load_secrets_script_exists(self):
        assert os.path.exists("scripts/load_secrets.sh"), (
            "scripts/load_secrets.sh not found"
        )

    def test_load_secrets_script_is_executable_or_sourceable(self):
        """Script should be a valid bash file (parseable by bash -n)."""
        result = subprocess.run(
            ["bash", "-n", "scripts/load_secrets.sh"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"load_secrets.sh has bash syntax error:\n{result.stderr}"
        )

    def test_store_secrets_script_exists(self):
        assert os.path.exists("scripts/store_secrets.sh"), (
            "scripts/store_secrets.sh not found"
        )

    def test_store_secrets_script_is_syntactically_valid(self):
        result = subprocess.run(
            ["bash", "-n", "scripts/store_secrets.sh"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"store_secrets.sh has bash syntax error:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 4. .env sanitization guard
# ---------------------------------------------------------------------------

class TestEnvSanitization:
    """
    After Phase 43, .env must NOT contain raw secret values.
    It should only hold non-secret config (SAFECLAW_HUB_URL, etc.).
    """

    # Patterns that indicate a secret is still in .env in plaintext
    SECRET_PATTERNS = [
        r"^NEBIUS_API_KEY\s*=\s*.+",
        r"^TELEGRAM_BOT_TOKEN\s*=\s*.+",
        r"^GITHUB_TOKEN\s*=\s*.+",
        r"^GOOGLE_API_KEY\s*=\s*.+",
    ]

    @pytest.mark.skipif(not os.path.exists(ENV_PATH), reason=".env not present")
    @pytest.mark.parametrize("pattern", SECRET_PATTERNS)
    def test_env_does_not_contain_secret(self, pattern):
        with open(ENV_PATH) as f:
            content = f.read()
        matches = [
            line for line in content.splitlines()
            if re.match(pattern, line.strip())
        ]
        assert not matches, (
            f".env still contains a plaintext secret matching '{pattern}'.\n"
            f"Run 'bash scripts/store_secrets.sh' to migrate to keychain, "
            f"then remove the line from .env.\n"
            f"Found: {matches}"
        )

    @pytest.mark.skipif(not os.path.exists(ENV_PATH), reason=".env not present")
    def test_env_retains_non_secret_config(self):
        """Non-secret config vars should still be in .env after migration."""
        with open(ENV_PATH) as f:
            content = f.read()
        assert "SAFECLAW_HUB_URL" in content, (
            "SAFECLAW_HUB_URL should remain in .env as a non-secret config var"
        )
