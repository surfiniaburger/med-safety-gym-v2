import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, Part, TextPart
from med_safety_gym.benchmark_agent import BenchmarkAgent, BenchmarkRequest

@pytest.mark.asyncio
async def test_benchmark_agent_orchestration():
    """
    Test that BenchmarkAgent correctly orchestrates a conversation.
    Mocks the Messenger to simulate a target agent.
    """
    agent = BenchmarkAgent()
    
    # Mock Messenger
    mock_messenger = AsyncMock()
    mock_messenger.talk_to_agent.return_value = "I am the subject and I am safe."
    agent.messenger = mock_messenger
    
    # Setup TaskUpdater
    updater = MagicMock(spec=TaskUpdater)
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()
    updater.complete = AsyncMock()
    updater.reject = AsyncMock()
    updater.failed = AsyncMock()
    
    # Create request
    request = BenchmarkRequest(
        target_agent_url="http://localhost:8003",
        scenario="recollection",
        num_turns=2
    )
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=request.model_dump_json()))],
        message_id="test_msg",
        context_id="test_ctx"
    )
    
    # Run Agent
    await agent.run(message, updater)
    
    # Verify Orchestration
    assert mock_messenger.talk_to_agent.call_count == 2
    updater.add_artifact.assert_called_once()
    updater.complete.assert_called_once()
    
    # Verify scenario-specific calls
    calls = mock_messenger.talk_to_agent.call_args_list
    assert "Panobinostat" in calls[0].kwargs["message"]
    assert "liquid formulation" in calls[1].kwargs["message"]
