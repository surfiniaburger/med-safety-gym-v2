"""
Verification Script for Docker Deployment

This script verifies that the Dockerized A2A+ADK+MCP architecture is working correctly.
It acts as an external client connecting to the exposed ports of the containers.

Usage:
    # 1. Start containers
    docker-compose up --build

    # 2. Run verification (in another terminal)
    uv run python server/verify_docker.py
"""

import asyncio
import httpx
import sys
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, SendMessageSuccessResponse, Task
from uuid import uuid4

# Configuration
AGENT_URL = "http://localhost:10000"
MCP_URL = "http://localhost:8081/mcp"

async def check_connectivity():
    """Check if services are reachable."""
    print("🔍 Checking connectivity...")
    
    # Check MCP Server
    try:
        async with httpx.AsyncClient() as client:
            # MCP uses SSE, so we just check if the endpoint is reachable
            # It returns 406 Not Acceptable for GET requests without Accept header, which means it's running
            resp = await client.get(MCP_URL)
            if resp.status_code in [200, 406]:
                print(f"✅ FastMCP Server reachable at {MCP_URL}")
            else:
                print(f"⚠️  FastMCP Server returned unexpected status: {resp.status_code}")
    except Exception as e:
        print(f"❌ FastMCP Server unreachable: {e}")
        return False

    # Check Agent Server
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{AGENT_URL}/.well-known/agent-card.json")
            if resp.status_code == 200:
                print(f"✅ ADK Agent reachable at {AGENT_URL}")
            else:
                print(f"❌ ADK Agent returned unexpected status: {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ ADK Agent unreachable: {e}")
        return False
        
    return True

async def run_evaluation_test():
    """Run a simple evaluation workflow test."""
    print("\n🧪 Running evaluation workflow test...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client, AGENT_URL)
            agent_card = await resolver.get_agent_card()
            print(f"✅ Agent card retrieved: {agent_card.name}")
            
            client = A2AClient(httpx_client, agent_card)
            
            # Request tasks
            print("📤 Requesting evaluation tasks...")
            payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Get me 1 evaluation task"}],
                    "messageId": uuid4().hex,
                }
            }
            request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
            response = await client.send_message(request)
            
            if isinstance(response.root, SendMessageSuccessResponse) and isinstance(response.root.result, Task):
                print(f"✅ Task created successfully: {response.root.result.id}")
                return True
            else:
                print(f"❌ Failed to create task. Response: {response}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    print("=" * 60)
    print("🐳 Docker Deployment Verification")
    print("=" * 60)
    
    if not await check_connectivity():
        print("\n❌ Connectivity check failed. Are containers running?")
        sys.exit(1)
        
    if await run_evaluation_test():
        print("\n✨ Verification Successful! Docker deployment is working.")
    else:
        print("\n❌ Verification Failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
