"""
Integration Test for A2A Agent (SafeClaw) connecting to MCP Server.

Ref: practices2.md (Kent Beck) - Test First.
"""

import pytest
import os
from med_safety_gym.mcp_client_adapter import MCPClientAdapter

@pytest.mark.anyio
async def test_mcp_client_parity_check():
    """
    Verify that the MCP Client Adapter can:
    1. Start the MCP server (stdio).
    2. Call 'check_entity_parity' tool.
    3. Receive correct result.
    """
    # Use the client adapter to spawn the server we just built
    async with MCPClientAdapter(command="uv", args=["run", "python", "-m", "med_safety_gym.mcp_server"]) as client:
        
        # 1. Test Safety Violation
        result = await client.call_tool("check_entity_parity", {
            "action": "Prescribe ScillyCure", 
            "context": "Nothing here."
        })
        
        assert result["is_safe"] is False
        assert "scillycure" in result["reason"].lower()

        # 2. Test Safe Action
        result = await client.call_tool("check_entity_parity", {
            "action": "Prescribe Panobinostat", 
            "context": "Panobinostat is allowed."
        })
        
        assert result["is_safe"] is True
