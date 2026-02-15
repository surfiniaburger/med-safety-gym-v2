"""
Quick Test: Verify SafeClaw server can start.
"""
import subprocess
import time
import requests
import pytest
from pathlib import Path

def test_server_starts():
    """Start the server and verify it responds."""
    # Start server in background
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "med_safety_gym.claw_server", "--port", "8888"],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server to start
        time.sleep(3)
        
        # Check if it's responding
        response = requests.get("http://localhost:8888/")
        # Accept 405 as A2A servers might only allow POST
        assert response.status_code in [200, 405]
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    test_server_starts()
    print("✅ Server smoke test passed!")
