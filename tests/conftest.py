# tests/conftest.py
"""
Pytest configuration for the test suite.
"""
import pytest


import os
import sys
from unittest.mock import patch

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
