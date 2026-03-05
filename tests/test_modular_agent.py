
import pytest
from unittest.mock import AsyncMock, MagicMock
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, TaskState
from med_safety_gym.intent_classifier import IntentCategory

@pytest.mark.asyncio
async def test_agent_uses_modular_mcp():
    # Mocks for MCP client factories
    mock_safety_client = AsyncMock()
    mock_safety_client.call_tool.return_value = {"is_safe": True}
    
    mock_intent_client = AsyncMock()
    mock_intent_client.call_tool.return_value = {"category": "NEW_TOPIC", "is_correction": False}
    
    mock_exp_client = AsyncMock()
    mock_exp_client.call_tool.return_value = "Guideline result"
    
    # Factory wrappers
    def safety_factory():
        m = MagicMock()
        m.__aenter__.return_value = mock_safety_client
        return m

    def intent_factory():
        m = MagicMock()
        m.__aenter__.return_value = mock_intent_client
        return m

    def exp_factory():
        m = MagicMock()
        m.__aenter__.return_value = mock_exp_client
        return m

    agent = SafeClawAgent(
        client_factory=safety_factory,
        intent_client_factory=intent_factory,
        experience_client_factory=exp_factory
    )
    
    # Mock updater
    updater = AsyncMock()
    
    # Mock message
    class DummyMessage:
        def __init__(self, text):
            self.parts = [MagicMock()]
            self.parts[0].root.text = text

    message = DummyMessage("What is DIPG?")
    
    # Patch _ensure_governor_interceptor to avoid Hub interaction
    agent._ensure_governor_interceptor = AsyncMock()
    
    await agent.run(message, updater)
    
    # Check if intent tool was called
    mock_intent_client.call_tool.assert_called_with("classify_intent", {"text": "What is DIPG?"})
    
    # Check if safety gate was called via safety_client
    mock_safety_client.call_tool.assert_called()
    
    # Check if experience logging was attempted (it should be in context_aware_action)
    # Wait, in run() it calls context_aware_action
    # We need to wait for the task to "complete"
    assert updater.update_status.called
