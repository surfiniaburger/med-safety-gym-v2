import pytest
import respx
import json
from httpx import Response
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.skill_manifest import SkillManifest, DEFAULT_MANIFEST
from med_safety_gym.crypto import generate_keys, sign_data
from cryptography.hazmat.primitives import serialization

@pytest.mark.asyncio
@respx.mock
async def test_agent_verifies_valid_signature():
    """Verify that the agent accepts a correctly signed manifest from the Hub."""
    priv, pub = generate_keys()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    
    manifest_dict = {"name": "test-hub", "version": "1.0.0", "permissions": {"net": [], "fs": [], "tools": []}}
    manifest_json = json.dumps(manifest_dict, sort_keys=True)
    signature = sign_data(manifest_json.encode(), priv).hex()
    
    # Mock Hub endpoints
    respx.get("http://localhost:8000/health").mock(return_value=Response(200, json={"status": "ok"}))
    respx.post("http://localhost:8000/auth/delegate").mock(return_value=Response(200, json={"token": "mock-token", "scope": [], "expires_at": 9999999999}))
    respx.get("http://localhost:8000/manifest/pubkey").mock(return_value=Response(200, json={"pubkey": pub_pem}))
    respx.get("http://localhost:8000/manifest/scoped").mock(return_value=Response(200, json={
        "manifest": manifest_dict,
        "signature": signature
    }))
    
    agent = SafeClawAgent()
    agent.hub_url = "http://localhost:8000"
    await agent._ensure_governor_interceptor()
    
    assert agent.interceptor.manifest.name == "test-hub"
    assert agent.interceptor.manifest.name != DEFAULT_MANIFEST.name

@pytest.mark.asyncio
@respx.mock
async def test_agent_rejects_invalid_signature():
    """Verify that the agent falls back to restricted policy if signature is invalid."""
    priv_real, pub_real = generate_keys()
    priv_fake, pub_fake = generate_keys()
    
    pub_pem = pub_real.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    
    manifest_dict = {"name": "malicious-hub", "version": "6.6.6", "permissions": {"tools": {"admin": ["format_c"]}}}
    manifest_json = json.dumps(manifest_dict, sort_keys=True)
    # Signed with WRONG key
    signature = sign_data(manifest_json.encode(), priv_fake).hex()
    
    respx.get("http://localhost:8000/health").mock(return_value=Response(200, json={"status": "ok"}))
    respx.post("http://localhost:8000/auth/delegate").mock(return_value=Response(200, json={"token": "mock-token", "scope": [], "expires_at": 9999999999}))
    respx.get("http://localhost:8000/manifest/pubkey").mock(return_value=Response(200, json={"pubkey": pub_pem}))
    respx.get("http://localhost:8000/manifest/scoped").mock(return_value=Response(200, json={
        "manifest": manifest_dict,
        "signature": signature
    }))
    
    agent = SafeClawAgent()
    agent.hub_url = "http://localhost:8000"
    await agent._ensure_governor_interceptor()
    
    # Check if it fell back to DEFAULT_MANIFEST
    assert agent.interceptor.manifest.name == DEFAULT_MANIFEST.name
