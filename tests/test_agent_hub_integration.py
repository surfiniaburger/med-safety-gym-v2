import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from med_safety_eval.observability_hub import app
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, TextPart, Part
from a2a.types import TaskState

@pytest.mark.asyncio
async def test_agent_fetches_manifest_from_hub():
    # 1. Setup Agent
    agent = SafeClawAgent()
    agent.hub_url = "http://localhost:8000"

    # 2. Mock manifest data
    mock_manifest = {
        "name": "test-manifest",
        "version": "1.0.0",
        "permissions": {
            "net": ["test.com"],
            "fs": ["/tmp"],
            "tools": {
                "user": ["test_tool"],
                "write": [],
                "admin": []
            }
        }
    }

    # 3. Use patch for more reliable mocking of httpx
    with patch("httpx.AsyncClient.get") as mock_get:
        # Mock responses for Health, Pubkey, and Manifest
        async def mock_router(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.raise_for_status = MagicMock()
            if "/health" in str(url):
                m.json.return_value = {"status": "ok"}
            elif "/manifest/pubkey" in str(url):
                # Return a valid-looking public key (PEM)
                from med_safety_gym.crypto import generate_keys
                from cryptography.hazmat.primitives import serialization
                _, pub = generate_keys()
                pub_pem = pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode()
                m.json.return_value = {"pubkey": pub_pem}
            elif "/manifest" in url:
                # Return signed manifest
                from med_safety_gym.crypto import generate_keys, sign_data
                priv, _ = generate_keys()
                manifest_json = json.dumps(mock_manifest, sort_keys=True)
                signature = sign_data(manifest_json.encode(), priv).hex()
                # Important: We need the test to pass, so we mock _verify_and_parse_manifest 
                # OR we ensure the pubkey we returned above matches this priv key.
                # Let's ensure they match for a real verification.
                return m
            return m

        # Simpler approach: Mock the high-level _fetch_signed_manifest or its components
        # to avoid complex httpx routing mocks in this specific test.
        # But let's try to keep it as an integration test.
        
        # Actually, the agent's _fetch_signed_manifest creates its OWN AsyncClient.
        # So patching httpx.AsyncClient.get at the class level is better.
        
        from med_safety_gym.crypto import generate_keys, sign_data
        from cryptography.hazmat.primitives import serialization
        import json
        
        priv, pub = generate_keys()
        pub_pem = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        manifest_json = json.dumps(mock_manifest, sort_keys=True)
        signature = sign_data(manifest_json.encode(), priv).hex()

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/health" in str(url):
                resp.json.return_value = {"status": "ok"}
            elif "/manifest/pubkey" in str(url):
                resp.json.return_value = {"pubkey": pub_pem}
            elif "/manifest" in str(url):
                resp.json.return_value = {"manifest": mock_manifest, "signature": signature}
            return resp

        mock_get.side_effect = side_effect

        # Trigger _ensure_governor_interceptor
        await agent._ensure_governor_interceptor()

    # 4. Verify interceptor
    assert agent.interceptor is not None
    assert agent.interceptor.manifest.name == "test-manifest"
    assert agent.interceptor.manifest.permissions.tools.user == ["test_tool"]

@pytest.mark.asyncio
async def test_agent_run_triggers_governor_fetch():
    agent = SafeClawAgent()
    agent.hub_url = "http://localhost:8000"
    
    # Mock _ensure_governor_interceptor
    agent._ensure_governor_interceptor = AsyncMock()
    
    updater = MagicMock()
    updater.update_status = AsyncMock()
    
    # Fix Message validation (adding messageId and role)
    message = Message(
        messageId="test-msg-id",
        role="user",
        parts=[Part(root=TextPart(text="gh: list issues"))]
    )
    
    try:
        await agent.run(message, updater)
    except Exception as e:
        print(f"Captured expected run failure: {e}")
        pass
    
    agent._ensure_governor_interceptor.assert_called_once()
