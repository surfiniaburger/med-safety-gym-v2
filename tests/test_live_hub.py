import pytest
import os
import asyncio
import httpx
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, TextPart, Part
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_live_handshake():
    """
    Smoke test: Verify the local agent can fetch a manifest 
    from the production Render Hub.
    """
    pytest.skip("Skipping until production Hub is updated to EdDSA/Asymmetric signing.")
    
    hub_url = "https://med-safety-hub-zqx8.onrender.com"
    
    # Skip if we can't reach the live hub quickly
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{hub_url}/health", timeout=2.0)
            if resp.status_code != 200:
                pytest.skip("Production Hub is currently unreachable.")
    except Exception:
        pytest.skip("Production Hub is offline or network restricted.")

    # Set the env var for the agent
    os.environ["SAFECLAW_HUB_URL"] = hub_url
    
    agent = SafeClawAgent()
    
    # 1. Trigger the boot handshake
    await agent._ensure_governor_interceptor()
    
    assert agent.interceptor is not None, "Interceptor should be initialized"
    assert agent.interceptor.manifest is not None, "Manifest should be loaded"
    
    manifest = agent.interceptor.manifest
    assert "safeclaw-core" in manifest.name
    
    # 2. Check a tool tier (must be in scope to see its tier)
    tool = "list_issues"
    tier = manifest.permissions.tools.tier_for(tool)
    assert tier == "user"

if __name__ == "__main__":
    asyncio.run(test_live_handshake())
