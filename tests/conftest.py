# tests/conftest.py
"""
Pytest configuration for the test suite.
"""
import pytest


import os
import sys
from unittest.mock import patch, MagicMock
from med_safety_gym.identity.secret_store import InMemorySecretStore

# Set default JWT secret for tests
if "JWT_SECRET" not in os.environ:
    os.environ["JWT_SECRET"] = "super_secret_test_key"

@pytest.fixture(scope="session")
def anyio_backend():
    """Force anyio to only use asyncio backend."""
    return "asyncio"

@pytest.fixture(autouse=True)
def mock_biometric_auth():
    """
    Automatically mock biometric/local auth for the entire test suite 
    if running in CI (GitHub Actions) or on non-macOS systems.
    """
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    is_not_mac = sys.platform != "darwin"
    
    if is_ci or is_not_mac:
        with patch("med_safety_gym.auth_guard.require_local_auth", return_value=True):
            yield
    else:
        yield

@pytest.fixture(autouse=True)
def mock_keyring():
    """
    Globally mock the 'keyring' library to prevent ANY test from triggering
    macOS Keychain password prompts. This covers all SecretStore implementations.
    """
    mock_kr = MagicMock()
    # Mock common keyring methods to return None (miss) by default
    mock_kr.get_password.return_value = None
    mock_kr.set_password.return_value = None
    mock_kr.delete_password.return_value = None
    
    with patch("keyring.get_password", mock_kr.get_password), \
         patch("keyring.set_password", mock_kr.set_password), \
         patch("keyring.delete_password", mock_kr.delete_password), \
         patch("keyring.get_keyring", return_value=mock_kr):
        yield

@pytest.fixture(autouse=True)
def force_in_memory_store():
    """
    Ensure SafeClawAgent uses InMemorySecretStore by default during tests.
    """
    with patch("med_safety_gym.claw_agent.KeyringSecretStore", return_value=InMemorySecretStore()):
        yield
