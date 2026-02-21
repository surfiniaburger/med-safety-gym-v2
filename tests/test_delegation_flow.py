import pytest
import os
import jwt
from unittest.mock import AsyncMock, patch
from a2a.types import TaskState
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.identity.scoped_identity import issue_delegation_token

@pytest.fixture
def test_secret():
    return os.environ.get("JWT_SECRET", "super_secret_test_key")

@pytest.fixture
def mock_updater():
    updater = AsyncMock()
    return updater

@pytest.mark.asyncio
async def test_interceptor_fails_on_invalid_token(mock_updater):
    agent = SafeClawAgent()
    agent.auth_token = "invalid.token.here"
    
    # We bypass _ensure_governor_interceptor slightly just to hit the gating logic
    # normally `interceptor` is initialized, but here we just need a mock
    from unittest.mock import MagicMock, ANY
    agent.interceptor = MagicMock()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is None
    mock_updater.update_status.assert_called_with(
        TaskState.failed,
        ANY # Don't care about exact Match, just that it was called with failed
    )
    # Check the error message contains 'Invalid delegation token'
    call_args = mock_updater.update_status.call_args[0]
    assert "Invalid delegation token" in call_args[1].parts[0].root.text

@pytest.mark.asyncio
async def test_interceptor_fails_on_expired_token(mock_updater, test_secret):
    agent = SafeClawAgent()
    # Issue expired token
    claims = {"sub": "test", "scope": ["read_file"], "profile": "read_only"}
    agent.auth_token = issue_delegation_token(claims, -3600, test_secret)
    
    agent.interceptor = AsyncMock()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is None
    call_args = mock_updater.update_status.call_args[0]
    assert "Delegation token has expired" in call_args[1].parts[0].root.text

@pytest.mark.asyncio
async def test_interceptor_gating_success(mock_updater, test_secret):
    agent = SafeClawAgent()
    claims = {"sub": "test", "scope": ["read_file"], "profile": "read_only"}
    agent.auth_token = issue_delegation_token(claims, 3600, test_secret)
    
    # Mock the interceptor to allow it
    from unittest.mock import MagicMock
    agent.interceptor = MagicMock()
    agent.interceptor.intercept.return_value = type('obj', (object,), {'allowed': True, 'tier': 'user', 'reason': ''})()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is not None
    assert result.allowed is True
