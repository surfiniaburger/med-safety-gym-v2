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
import requests
import os

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll a server URL until it responds or timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code in [200, 406]:  # 406 acceptable for MCP
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
    return False

def start_server(command: list, name: str, health_url: str, wait_time: int = 30, env: dict | None = None):
    """Start a server in the background and wait for it to be ready."""
    
    print(f"🚀 Starting {name}...")
    
    process_env = None
    if env:
        process_env = os.environ.copy()
        process_env.update({k: str(v) for k, v in env.items()})
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=process_env,
    )
    print(f"   Waiting for {name} to be ready (timeout: {wait_time}s)...")
    if wait_for_server(health_url, timeout=wait_time):
        print(f"✅ {name} is ready!")
        return process
    else:
        print(f"❌ {name} failed to start in {wait_time}s.")
        process.terminate()
        raise RuntimeError(f"{name} failed to start")

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
            ["uv", "run", "python", "server/fastmcp_server.py"],
            "FastMCP Server",
            "http://localhost:8081/mcp",
            wait_time=30,
            env={"PORT": "8081"},
        )
        
        # Start A2A agent server
        a2a_process = start_server(
            ["uv", "run", "uvicorn", "server.dipg_agent:a2a_app", "--host", "localhost", "--port", "10000"],
            "A2A Agent Server",
            "http://localhost:10000/.well-known/agent-card.json",
            wait_time=30,
            env={"MCP_SERVER_URL": "http://localhost:8081/mcp"},
        )
        
        # Run test client
        print("\n🧪 Running A2A test client...")
        result = subprocess.run(
            ["uv", "run", "python", "server/test_a2a_client.py"]
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
