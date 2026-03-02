"""
Unit Test for SafeClaw Agent (Brain Layer).
Verifies that the agent correctly uses the MCP Client Adapter and respects safety results.
"""

import pytest
from unittest.mock import AsyncMock, patch
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, TextPart, Part

@pytest.mark.anyio
async def test_agent_uses_intent_classifier():
    """
    Scenario: User sends a multi-turn refinement message.
    The agent's run() method should classify it and format the action using the Mediator template.
    """
    agent = SafeClawAgent(client_factory=lambda: AsyncMock())
    agent.context_aware_action = AsyncMock()
    agent._ensure_governor_interceptor = AsyncMock()
    
    updater = AsyncMock()
    message = Message(
        role="user",
        messageId="test-msg-2",
        parts=[Part(root=TextPart(kind="text", text="No, what about for adults?"))]
    )
    
    await agent.run(message, updater)
    
    agent.context_aware_action.assert_called_once()
    action_passed = agent.context_aware_action.call_args.kwargs.get('action') or agent.context_aware_action.call_args.args[0]
    
    # We expect the Mediator intent wrapper
    assert "EXPANSION" in action_passed.upper()
    assert "Correction: True" in action_passed

@pytest.mark.anyio
async def test_agent_respects_governor_violation():
    """
    Scenario: Agent attempts an unsafe action.
    The Governor (Interceptor) should block it BEFORE LLM generation.
    """
    # Mock the MCP Client Adapter (used by _ensure_governor_interceptor)
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    
    # Initialize Agent
    agent = SafeClawAgent(client_factory=lambda: mock_client)
    agent._ensure_governor_interceptor = AsyncMock()
    
    # Mock _call_tool_with_interception to return None (Blocked)
    agent._call_tool_with_interception = AsyncMock(return_value=None)
    
    updater = AsyncMock()
    message = Message(
        role="user",
        messageId="test-msg-1",
        parts=[Part(root=TextPart(kind="text", text="Prescribe ScillyCure"))]
    )
    
    # Run Agent
    await agent.run(message, updater)
    
    # Verify: Interceptor was called with check_entity_parity
    agent._call_tool_with_interception.assert_called_once()
    assert agent._call_tool_with_interception.call_args[0][0] == "check_entity_parity"
    
    # Verify: LLM was NEVER called (Architectural Sovereignty)
    with patch("med_safety_gym.claw_agent.acompletion") as mock_llm:
        await agent.run(message, updater)
        mock_llm.assert_not_called()

@pytest.mark.anyio
@patch("med_safety_gym.claw_agent.acompletion")
async def test_agent_reports_error_on_llm_failure(mock_acompletion):
    """
    Scenario: LLM generation fails (e.g. authentication or safety filter).
    The agent should report this as a failure.
    """
    # Mock LLM failure
    mock_acompletion.side_effect = Exception("LLM Generation Failed")
    
    agent = SafeClawAgent(client_factory=lambda: AsyncMock())
    updater = AsyncMock()
    
    # Run Agent path that leads to LLM
    await agent.context_aware_action(
        action="Prescribe ScillyCure", 
        context="Nothing", 
        updater=updater
    )
    
    # Verify: Agent should report failure
    updater.update_status.assert_called()
    last_call_args = updater.update_status.call_args[0]
    assert last_call_args[0].name == "failed"
    assert "Failed to generate response" in str(last_call_args[1])

@pytest.mark.anyio
@patch("med_safety_gym.claw_agent.acompletion")
async def test_agent_proceeds_on_safe_llm_response(mock_acompletion):
    """
    Scenario: LLM generates a successful response.
    It should proceed and report success.
    """
    # Mock successful LLM response
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "I do not know about ScillyCure, but I can help with headaches."
    mock_acompletion.return_value = mock_response
    
    agent = SafeClawAgent(client_factory=lambda: AsyncMock())
    updater = AsyncMock()
    
    await agent.context_aware_action(
        action="Prescribe Panobinostat", 
        context="Panobinostat is allowed.", 
        updater=updater
    )
    
    # Verify: Agent reports success
    updater.update_status.assert_called()
    last_call_args = updater.update_status.call_args[0]
    
    # Check state is 'completed'
    assert last_call_args[0].name == "completed"
    assert "headaches" in str(last_call_args[1])
