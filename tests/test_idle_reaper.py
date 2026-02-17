import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from med_safety_gym.executor import Executor
from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext
from a2a.types import Message, TextPart, Part, Task, TaskState, TaskStatus

class MockAgent:
    def __init__(self):
        self.run_called = False
        self.shutdown_called = False
        
    async def run(self, msg, updater):
        self.run_called = True
        
    async def shutdown(self):
        self.shutdown_called = True

@pytest.mark.asyncio
async def test_executor_reaps_idle_agent():
    # 1. Setup Executor with 1s TTL
    executor = Executor(MockAgent, idle_ttl=1)
    
    # 2. Spawn an agent
    msg = Message(messageId="m1", role="user", parts=[Part(root=TextPart(text="hello"))])
    status = TaskStatus(state=TaskState.working)
    task = Task(id="t1", context_id="ctx1", status=status)
    
    # Use MagicMock for RequestContext to avoid instantiation errors
    context = MagicMock()
    context.message = msg
    context.current_task = task
    
    event_queue = MagicMock(spec=EventQueue)
    event_queue.enqueue_event = AsyncMock()

    await executor.execute(context, event_queue)
    
    assert "ctx1" in executor.agents
    agent = executor.agents["ctx1"]
    
    # 3. Wait for TTL to expire
    await asyncio.sleep(1.5)
    
    # 4. Trigger one iteration of the reaper logic
    now = time.time()
    to_reap = [
        ctx_id for ctx_id, last_seen in executor.last_activity.items()
        if now - last_seen > executor.idle_ttl
    ]
    for ctx_id in to_reap:
        await executor._shutdown_single_agent(ctx_id)
        
    # 5. Verify reaper actions
    assert "ctx1" not in executor.agents
    assert agent.shutdown_called is True

@pytest.mark.asyncio
async def test_reaper_loop_integration():
    # This test actually runs the background task
    executor = Executor(MockAgent, idle_ttl=1)
    
    # Monkeypatch sleep for the reaper loop to be fast
    original_sleep = asyncio.sleep
    async def fast_sleep(delay):
        if delay == 60: # The reaper interval
            await original_sleep(0.1)
        else:
            await original_sleep(delay)
            
    with patch("asyncio.sleep", side_effect=fast_sleep):
        await executor.start()
        
        # Spawn agent
        msg = Message(messageId="m2", role="user", parts=[Part(root=TextPart(text="hello"))])
        status = TaskStatus(state=TaskState.working)
        task = Task(id="t2", context_id="ctx2", status=status)
        
        # Mock RequestContext
        context = MagicMock()
        context.message = msg
        context.current_task = task
        
        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()
        
        await executor.execute(context, event_queue)
        assert "ctx2" in executor.agents
        
        # Wait for reaper to run (TTL is 1s, loop is 0.1s)
        await asyncio.sleep(1.5)
        
        assert "ctx2" not in executor.agents
        
        await executor.shutdown()
