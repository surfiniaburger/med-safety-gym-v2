import pytest
from unittest.mock import AsyncMock, MagicMock, ANY, patch
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
    agent.auth_token = "valid-mock-token"
    agent.hub_pub_key = "dummy-pub-key"
    
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
    with patch("med_safety_gym.claw_agent.verify_delegation_token"):
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
    agent.auth_token = "valid-mock-token"
    agent.hub_pub_key = "dummy-pub-key"
    
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
    
    with patch("med_safety_gym.claw_agent.verify_delegation_token"):
        await agent.run(message, updater)
    
    github_client.call_tool.assert_called_with("configure_repo", {"repo_name": "my/repo"})

@pytest.mark.asyncio
@patch("med_safety_gym.claw_agent.acompletion")
async def test_agent_handles_medical_default(mock_acompletion):
    """Verify that non-github messages call LLM generation."""
    # Mock LLM response
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Mocked Response"
    mock_acompletion.return_value = mock_response

    agent = SafeClawAgent(client_factory=lambda: MagicMock())
    agent._ensure_governor_interceptor = AsyncMock()
    
    updater = AsyncMock()
    message = Message(
        messageId="test-id",
        role="user",
        parts=[Part(root=TextPart(text="Prescribe Panobinostat"))]
    )
    
    await agent.run(message, updater)
    
    mock_acompletion.assert_called()
    # Also verify structural check was attempted (mock_client from client_factory)
    # Actually in test_agent_handles_medical_default, we use MagicMock() for client_factory.
    # In my logic, _call_tool_with_interception will call mcp_server.check_entity_parity if session_client is None.
    # But in run(), we pass None as session_client. So it should call the local check.
