
import pytest
import logging
import asyncio
import time
from unittest.mock import patch, MagicMock
from med_safety_gym.executor import Executor
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, Part, TextPart
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task
from uuid import uuid4

class MockAgent:
    def __init__(self):
        self.shutdown_called = False
    async def shutdown(self):
        self.shutdown_called = True
    async def run(self, msg, updater):
        pass

@pytest.mark.asyncio
async def test_executor_logs_reaping(caplog):
    """Executor must log at INFO level when an agent is reaped due to inactivity."""
    caplog.set_level(logging.INFO)
    
    # Initialize executor with 0.1s TTL for fast testing
    executor = Executor(MockAgent, idle_ttl=0.1)
    
    # "Cold Start" an agent with messageId
    msg = Message(
        role="user", 
        parts=[Part(root=TextPart(kind="text", text="hello"))],
        messageId=uuid4().hex
    )
    from a2a.types import MessageSendParams
    params = MessageSendParams(message=msg)
    task = new_task(msg)
    context = RequestContext(request=params, task=task)
    event_queue = EventQueue()
    
    await executor.execute(context, event_queue)
    
    assert len(executor.agents) == 1
    ctx_id = task.context_id
    
    # Wait for TTL to expire
    await asyncio.sleep(0.2)
    
    now = time.time()
    to_reap = [
        c_id for c_id, last_seen in executor.last_activity.items()
        if now - last_seen > executor.idle_ttl
    ]
    
    for c_id in to_reap:
        await executor._shutdown_single_agent(c_id)
        
    assert ctx_id not in executor.agents
    assert "Reaping idle agent" in caplog.text

@pytest.mark.asyncio
async def test_agent_logs_shutdown(caplog):
    """SafeClawAgent must log at INFO level when shutdown() is called."""
    caplog.set_level(logging.INFO)
    
    agent = SafeClawAgent()
    await agent.shutdown()
    
    assert "SafeClaw Agent shutting down" in caplog.text

@pytest.mark.asyncio
async def test_agent_logs_token_expiry(caplog):
    """SafeClawAgent must log at WARNING level when jwt.ExpiredSignatureError occurs."""
    caplog.set_level(logging.WARNING)
    
    agent = SafeClawAgent()
    agent.auth_token = "expired-token"
    agent.hub_pub_key = "dummy-pub-key"
    
    import respx
    import httpx
    import jwt
    
    with patch("med_safety_gym.claw_agent.SafeClawAgent._ensure_hub_running", return_value=None):
        with patch("med_safety_gym.claw_agent.SafeClawAgent._verify_and_parse_manifest", 
                   side_effect=jwt.ExpiredSignatureError("Token expired")):
            async with respx.mock:
                respx.get(f"{agent.hub_url}/manifest/scoped").respond(200, json={"manifest": {}, "signature": "..."})
                respx.post(f"{agent.hub_url}/auth/delegate").respond(200, json={"token": "new-token"})
                respx.get(f"{agent.hub_url}/manifest/pubkey").respond(200, json={"pubkey": "..."})
                
                await agent._init_scoped_session()
                
    assert "Delegation token has expired" in caplog.text
