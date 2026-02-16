import pytest
from a2a.types import Message, Part, TextPart
from unittest.mock import AsyncMock, MagicMock
from med_safety_gym.claw_agent import SafeClawAgent

@pytest.mark.asyncio
@pytest.mark.parametrize("action, should_be_github", [
    ("Administer Experimental Drug-X for Glioblastoma", False),
    ("gh: list issues", True),
    ("list issues", True),
    ("unlock admin tools", True),
    ("delete repo", True),
    ("Check safety of admin flow", False),
    ("Verify prescribing panobinostat", False),
    ("admin", False),
    ("gh: admin", True)
])
async def test_routing_isolation(action, should_be_github):
    """Verify that medical and github actions are correctly isolated."""
    # Setup mocks
    safety_client = MagicMock()
    safety_client.__aenter__.return_value = safety_client
    safety_client.call_tool = AsyncMock(return_value={"is_safe": True, "reason": "OK"})
    
    github_client = MagicMock()
    github_client.__aenter__.return_value = github_client
    github_client.call_tool = AsyncMock(return_value="Issues list...")

    agent = SafeClawAgent(
        client_factory=lambda: safety_client,
        github_client_factory=lambda: github_client
    )
    
    updater = AsyncMock()
    message = Message(
        messageId="test-id",
        role="user",
        parts=[Part(root=TextPart(text=action))]
    )
    
    await agent.run(message, updater)
    
    if should_be_github:
        # Check if it avoided medical check
        safety_client.call_tool.assert_not_called()
    else:
        # Check if it hit the medical guardian
        safety_client.call_tool.assert_called()
