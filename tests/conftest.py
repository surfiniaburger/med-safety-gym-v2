# tests/conftest.py
"""
Pytest configuration for the test suite.
"""
import pytest


import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
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
    mock_kr.get_password.return_value = None
    mock_kr.set_password.return_value = None
    mock_kr.delete_password.return_value = None
    
    with (
        patch("keyring.get_password", mock_kr.get_password),
        patch("keyring.set_password", mock_kr.set_password),
        patch("keyring.delete_password", mock_kr.delete_password),
        patch("keyring.get_keyring", return_value=mock_kr),
    ):
        yield

@pytest.fixture(autouse=True)
def force_in_memory_store():
    """
    Ensure SafeClawAgent uses InMemorySecretStore by default during tests.
    Applied Gemini's suggestion to patch with the class directly.
    """
    with patch("med_safety_gym.claw_agent.KeyringSecretStore", InMemorySecretStore):
        yield

@pytest.fixture(autouse=True)
def mock_litellm():
    """
    Globally mock LiteLLM's acompletion to prevent tests from hitting
    real APIs (and failing in CI due to missing keys).
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "SafeClaw: Action approved by LLM mock."
    
    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=mock_response)):
        yield

@pytest.fixture(autouse=True)
def mock_mcp_adapter():
    """
    Globally mock MCPClientAdapter to prevent tests from spawning real subprocesses.
    This fixes the 'Hanging at 39%' issue while keeping unit tests fast.
    """
    mock_instance = MagicMock()
    from med_safety_gym.intent_classifier import IntentClassifier
    classifier = IntentClassifier()
    
    async def mock_call_tool(name, arguments):
        if name == "classify_intent":
            text = arguments.get("text", "")
            res = classifier.classify(text)
            return {"category": res.category.name, "is_correction": res.is_correction}
        if name in ["check_entity_parity", "check_grounding_parity", "verify_synthesis_match", "check_trace_support"]:
            return {"is_safe": True, "reason": "Mocked safety pass"}
        if name == "extract_clinical_entities":
            # Return some default entities if needed, or empty list
            return []
        return {}
        
    mock_instance.call_tool = AsyncMock(side_effect=mock_call_tool)
    mock_instance.list_tools = AsyncMock(return_value=MagicMock())
    
    with patch("med_safety_gym.mcp_client_adapter.MCPClientAdapter.__aenter__", AsyncMock(return_value=mock_instance)), \
         patch("med_safety_gym.mcp_client_adapter.MCPClientAdapter.__aexit__", AsyncMock()):
        yield

@pytest.fixture(autouse=True)
def mock_guidelines_loading(request):
    """
    Prevent the agent from attempting to fetch distilled guidelines during every test.
    """
    if request.node.get_closest_marker("allow_guidelines_loading"):
        yield
        return

    with patch("med_safety_gym.claw_agent.SafeClawAgent._load_pragmatic_guidelines", AsyncMock()):
        yield
