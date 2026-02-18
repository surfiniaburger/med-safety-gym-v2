import pytest
import unittest.mock as mock
import httpx
from med_safety_gym.claw_agent import SafeClawAgent

@pytest.mark.asyncio
async def test_agent_starts_local_hub_on_failure():
    """Verify that the agent attempts to start a local hub if connectivity fails."""
    agent = SafeClawAgent()
    agent.hub_url = "http://localhost:9999" # Non-existent port
    
    # Mock httpx.get to fail for both health check and manifest fetch
    # Mock subprocess.Popen to avoid actually starting a process
    with mock.patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection failed")), \
         mock.patch("subprocess.Popen") as mock_popen:
        
        await agent._ensure_governor_interceptor()
        
        # Verify that subprocess.Popen was called to start uvicorn
        assert mock_popen.called
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert "uvicorn" in cmd
        assert "med_safety_eval.observability_hub:app" in cmd
        assert "--port" in cmd
        assert "9999" in cmd
