import pytest
from unittest.mock import AsyncMock, MagicMock, ANY
from a2a.types import Message, TextPart, Part
from med_safety_gym.claw_agent import SafeClawAgent

@pytest.mark.asyncio
async def test_agent_routes_to_github():
    """Verify that 'gh:' messages are routed to the GitHub handler."""
    # Setup mocks
    github_client = MagicMock()
    github_client.__aenter__.return_value = github_client
    github_client.call_tool = AsyncMock(return_value="Issues list...")
    
    github_factory = lambda: github_client
    
    agent = SafeClawAgent(client_factory=lambda: MagicMock(), github_client_factory=github_factory)
    
    # Mock the interceptor to be permissive for this test
    agent.interceptor = MagicMock()
    agent.interceptor.intercept.return_value = MagicMock(allowed=True, tier="user")
    agent._ensure_governor_interceptor = AsyncMock()
    
    updater = AsyncMock()
    message = Message(
        messageId="test-id",
        role="user",
        parts=[Part(root=TextPart(text="gh: list issues"))]
    )
    
    # Run agent
    await agent.run(message, updater)
    
    # Assertions
    github_client.call_tool.assert_called_with("list_issues", {})
    updater.update_status.assert_any_call(ANY, ANY) # Check for status updates

@pytest.mark.asyncio
async def test_agent_configures_repo():
    """Verify that 'gh: set repo' calls the correct tool."""
    github_client = MagicMock()
    github_client.__aenter__.return_value = github_client
    github_client.call_tool = AsyncMock(return_value="Repo updated")
    
    agent = SafeClawAgent(github_client_factory=lambda: github_client)
    
    # Mock the interceptor to be permissive for this test
    agent.interceptor = MagicMock()
    agent.interceptor.intercept.return_value = MagicMock(allowed=True, tier="user")
    agent._ensure_governor_interceptor = AsyncMock()
    
    updater = AsyncMock()
    message = Message(
        messageId="test-id",
        role="user",
        parts=[Part(root=TextPart(text="gh: set repo my/repo"))]
    )
    
    await agent.run(message, updater)
    
    github_client.call_tool.assert_called_with("configure_repo", {"repo_name": "my/repo"})

@pytest.mark.asyncio
async def test_agent_handles_medical_default():
    """Verify that non-github messages still call safety tools."""
    safety_client = MagicMock()
    safety_client.__aenter__.return_value = safety_client
    safety_client.call_tool = AsyncMock(return_value={"is_safe": True, "reason": "OK"})
    
    # We need a medical mock
    agent = SafeClawAgent(client_factory=lambda: safety_client)
    
    updater = AsyncMock()
    message = Message(
        messageId="test-id",
        role="user",
        parts=[Part(root=TextPart(text="Prescribe Panobinostat"))]
    )
    
    await agent.run(message, updater)
    
    safety_client.call_tool.assert_called_with("check_entity_parity", ANY)
