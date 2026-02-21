import pytest
import os
import jwt
from unittest.mock import AsyncMock, patch
from a2a.types import TaskState
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.identity.scoped_identity import issue_delegation_token

@pytest.fixture
def test_eddsa_keys():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    
    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()
    
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()
    
    pub_pem = pub_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    
    return {"private": priv_pem, "public": pub_pem}

@pytest.fixture
def mock_updater():
    return AsyncMock()

@pytest.mark.asyncio
async def test_interceptor_fails_on_invalid_token(mock_updater, test_eddsa_keys):
    agent = SafeClawAgent()
    agent.auth_token = "invalid.token.here"
    agent.hub_pub_key = test_eddsa_keys["public"]
    
    from unittest.mock import MagicMock, ANY
    agent.interceptor = MagicMock()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is None
    mock_updater.update_status.assert_called()
    call_args = mock_updater.update_status.call_args[0]
    assert "Invalid delegation token" in str(call_args[1])

@pytest.mark.asyncio
async def test_interceptor_fails_on_expired_token(mock_updater, test_eddsa_keys):
    agent = SafeClawAgent()
    agent.hub_pub_key = test_eddsa_keys["public"]

    # Issue expired token
    claims = {"sub": "test", "scope": ["read_file"], "profile": "read_only"}
    agent.auth_token = issue_delegation_token(claims, -3600, test_eddsa_keys["private"])
    
    agent.interceptor = AsyncMock()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is None
    call_args = mock_updater.update_status.call_args[0]
    assert "Delegation token has expired" in str(call_args[1])

@pytest.mark.asyncio
async def test_interceptor_gating_success(mock_updater, test_eddsa_keys):
    agent = SafeClawAgent()
    agent.hub_pub_key = test_eddsa_keys["public"]

    claims = {"sub": "test", "scope": ["read_file"], "profile": "read_only"}
    agent.auth_token = issue_delegation_token(claims, 3600, test_eddsa_keys["private"])
    
    # Mock the interceptor to allow it
    from unittest.mock import MagicMock
    agent.interceptor = MagicMock()
    agent.interceptor.intercept.return_value = type('obj', (object,), {'allowed': True, 'tier': 'user', 'reason': ''})()
    
    result = await agent._verify_and_gate_tool_call("read_file", {}, mock_updater, None)
    
    assert result is not None
    assert result.allowed is True
