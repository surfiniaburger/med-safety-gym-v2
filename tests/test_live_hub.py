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
    hub_url = "https://med-safety-hub-zqx8.onrender.com"
    print(f"🚀 Testing live handshake with Hub: {hub_url}")
    
    # Set the env var for the agent
    os.environ["SAFECLAW_HUB_URL"] = hub_url
    
    agent = SafeClawAgent()
    
    # 1. Trigger the boot handshake
    print("📡 Fetching manifest...")
    await agent._ensure_governor_interceptor()
    
    if agent.interceptor and agent.interceptor.manifest:
        manifest = agent.interceptor.manifest
        print(f"✅ SUCCESS: Connected to Governor!")
        print(f"📦 Manifest Name: {manifest.name}")
        print(f"🏷️ Version: {manifest.version}")
        print(f"🛠️ Tools declared: {manifest.permissions.tools.all_tools}")
        
        # 2. Check a tool tier
        tool = "delete_repo"
        tier = manifest.permissions.tools.tier_for(tool)
        print(f"🔒 Security Check: Tier for '{tool}' is '{tier}'")
    else:
        print("❌ FAILED: Could not initialize interceptor from remote Hub.")

if __name__ == "__main__":
    asyncio.run(test_live_handshake())
