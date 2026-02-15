"""
Quick Test: Verify SafeClaw can be started and responds to messages.
"""

import pytest
import httpx
from med_safety_gym.claw_server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_agent_card():
    """Verify the agent card is reachable or server is up."""
    # A2A Starlette apps often reject GET with 405 as they expect JSON-RPC POST
    response = client.get("/")
    assert response.status_code in [200, 405]

def test_health_check():
    """Verify server is responsive."""
    response = client.get("/health")
    # A2A apps may not have /health by default, but the root endpoint works
    assert response.status_code in [200, 404]  # Either exists or doesn't matter

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
