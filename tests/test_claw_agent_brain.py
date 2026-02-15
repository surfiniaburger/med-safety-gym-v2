"""
Unit Test for SafeClaw Agent (Brain Layer).
Verifies that the agent correctly uses the MCP Client Adapter and respects safety results.
"""

import pytest
from unittest.mock import AsyncMock, patch
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, TextPart, Part

@pytest.mark.anyio
async def test_agent_respects_safety_violation():
    """
    Scenario: Agent attempts an unsafe action (blocked by MCP).
    It should report the violation to the user.
    """
    # Mock the MCP Client Adapter
    mock_client = AsyncMock()
    # Mock context manager behavior
    mock_client.__aenter__.return_value = mock_client
    
    # Mock call_tool to return a Safety Violation
    mock_client.call_tool.return_value = {
        "is_safe": False,
        "reason": "Entity Parity Violation: Found entities {'scillycure'} in action not found in context."
    }
    
    # Initialize Agent with mocked client factory
    agent = SafeClawAgent(client_factory=lambda: mock_client)
    
    # Mock the TaskUpdater
    updater = AsyncMock()
    
    # Simulate User Message
    message = Message(
        role="user",
        messageId="test-msg-1",
        parts=[Part(root=TextPart(kind="text", text="Run safety check on ScillyCure"))]
    )
    
    # Run Agent
    await agent.context_aware_action(
        action="Prescribe ScillyCure", 
        context="Nothing", 
        updater=updater
    )
    
    # Verify: Agent should NOT proceed (or should report failure)
    updater.update_status.assert_called()
    last_call_args = updater.update_status.call_args[0]
    
    # Check state is 'failed'
    # Note: TaskState is an enum, so we compare the member or name
    assert str(last_call_args[0]) == "TaskState.failed" or last_call_args[0].name == "failed"
    
    assert "Safety Violation" in str(last_call_args[1]) or "blocked" in str(last_call_args[1]).lower()

@pytest.mark.anyio
async def test_agent_proceeds_on_safe():
    """
    Scenario: Agent attempts a safe action.
    It should proceed and report success.
    """
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.call_tool.return_value = {"is_safe": True, "reason": "OK"}
    
    agent = SafeClawAgent(client_factory=lambda: mock_client)
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
    assert str(last_call_args[0]) == "TaskState.completed" or last_call_args[0].name == "completed"
