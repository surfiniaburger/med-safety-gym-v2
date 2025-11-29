import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.mcp_server"],
        env=os.environ.copy() # Pass environment variables (important for API keys/config)
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # List tools
            print("📋 Listing tools...")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test get_eval_tasks
            print("\n🧪 Testing get_eval_tasks...")
            result = await session.call_tool("get_eval_tasks", arguments={"max_samples": 2})
            print(f"  Result: {result.content[0].text[:200]}...") # Print first 200 chars

            # Test evaluate_batch (using a mock response)
            print("\n🧪 Testing evaluate_batch...")
            mock_evaluations = [{
                "response": '{"analysis": "test", "proof": "test", "final": "test"}',
                "ground_truth": {
                    "context": "Context",
                    "question": "Question",
                    "expected_answer": {"final": "Answer"}
                }
            }]
            
            result = await session.call_tool("evaluate_batch", arguments={
                "evaluations": mock_evaluations,
                "format": "json"
            })
            print(f"  Result: {result.content[0].text[:200]}...")

if __name__ == "__main__":
    asyncio.run(run())
