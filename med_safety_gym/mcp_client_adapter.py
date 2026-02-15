
import asyncio
import json
import os
from typing import Optional, List, Dict, Any, Union
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientAdapter:
    """
    Adapts the MCP Client Protocol for use by A2A Agents.
    Manages the subprocess lifecycle and session handling.
    """
    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None

    async def __aenter__(self):
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )
        
        self.exit_stack = AsyncExitStack()
        
        # Start the stdio client
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        # Initialize the session
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.exit_stack:
            await self.exit_stack.aclose()

    async def list_tools(self):
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
            
        result = await self.session.call_tool(name, arguments)
        
        if not result.content:
            return {}
            
        # Prioritize parsing JSON from the first TextContent
        first_content = result.content[0]
        if first_content.type == "text":
            try:
                return json.loads(first_content.text)
            except json.JSONDecodeError:
                return first_content.text
        
        return str(result.content)
