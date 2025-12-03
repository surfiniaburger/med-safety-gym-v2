"""
Integration Test Script for A2A + ADK + MCP

This script starts all three components and runs integration tests:
1. FastMCP Server (port 8081)
2. A2A Agent Server (port 10000)
3. A2A Test Client

Usage:
    # Start servers (in separate terminals):
    uv run python server/fastmcp_server.py
    uv run uvicorn server.dipg_agent:a2a_app --host localhost --port 10000
    
    # Run tests:
    uv run python server/test_a2a_client.py
"""

import subprocess
import time
import sys

def start_server(command: str, name: str, wait_time: int = 3):
    """Start a server in the background."""
    print(f"🚀 Starting {name}...")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"   Waiting {wait_time}s for {name} to start...")
    time.sleep(wait_time)
    return process

def main():
    """Run the full integration test."""
    print("=" * 70)
    print("A2A + ADK + MCP Integration Test")
    print("=" * 70)
    
    fastmcp_process = None
    a2a_process = None
    
    try:
        # Start FastMCP server
        fastmcp_process = start_server(
            "PORT=8081 uv run python server/fastmcp_server.py",
            "FastMCP Server",
            wait_time=5
        )
        
        # Start A2A agent server
        a2a_process = start_server(
            "MCP_SERVER_URL=http://localhost:8081/mcp uv run uvicorn server.dipg_agent:a2a_app --host localhost --port 10000",
            "A2A Agent Server",
            wait_time=5
        )
        
        # Run test client
        print("\n🧪 Running A2A test client...")
        result = subprocess.run(
            "uv run python server/test_a2a_client.py",
            shell=True
        )
        
        if result.returncode == 0:
            print("\n✅ Integration test passed!")
        else:
            print("\n❌ Integration test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if fastmcp_process:
            fastmcp_process.terminate()
            print("   Stopped FastMCP server")
        if a2a_process:
            a2a_process.terminate()
            print("   Stopped A2A agent server")

if __name__ == "__main__":
    main()
