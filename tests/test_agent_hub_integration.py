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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_manifest
        mock_resp.raise_for_status = MagicMock()
        
        mock_get.return_value = mock_resp

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
