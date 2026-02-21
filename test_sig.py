import asyncio
import httpx
import json
from med_safety_gym.claw_agent import SafeClawAgent

async def main():
    agent = SafeClawAgent()
    manifest = await agent._fetch_signed_manifest()
    print("Agent loaded manifest name:", manifest.name)
    print("Permissions:", manifest.permissions.tools.user)

if __name__ == "__main__":
    asyncio.run(main())
