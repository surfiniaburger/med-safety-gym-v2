"""
Test script for FastMCP server

This script tests the FastMCP server tools using the MCP client.
"""

import asyncio
import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def test_fastmcp_server():
    """Test the FastMCP server tools."""
    
    server_url = "http://localhost:8081/mcp"
    
    print(f"🔗 Connecting to FastMCP server at {server_url}...")
    
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            print("✅ Connected to FastMCP server")
            
            # List available tools
            print("\n📋 Listing available tools...")
            tools_result = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools_result.tools]}")
            
            # Test get_eval_tasks
            print("\n🧪 Testing get_eval_tasks tool...")
            result = await session.call_tool(
                "get_eval_tasks",
                arguments={"max_samples": 2, "shuffle": False}
            )
            print(f"Result: {result.content[0].text[:200]}...")  # First 200 chars
            
            print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_fastmcp_server())
