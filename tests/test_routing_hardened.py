import pytest
from a2a.types import Message, Part, TextPart
from unittest.mock import AsyncMock, MagicMock, patch
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
@patch("med_safety_gym.claw_agent.acompletion", new_callable=AsyncMock)
async def test_routing_isolation(mock_acompletion, action, should_be_github):
    """Verify that medical and github actions are correctly isolated."""
    # Setup mocks
    safety_client = MagicMock()
    safety_client.__aenter__.return_value = safety_client
    safety_client.call_tool = AsyncMock(return_value={"is_safe": True, "reason": "OK"})
    
    github_client = MagicMock()
    github_client.__aenter__.return_value = github_client
    github_client.call_tool = AsyncMock(return_value="Issues list...")

    # Mock litellm response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked LLM response."
    mock_acompletion.return_value = mock_response

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
    
    # Identify commands that trigger a hardcoded fallback message (Zero-Trust or Info)
    # in claw_agent.py handlers, which skip the tool execution layer entirely.
    clean_action = action.lower().replace("gh:", "").strip()
    is_admin_fallback = any(kw in clean_action for kw in ["unlock", "delete repo"])
    is_info_fallback = "admin" == clean_action

    if should_be_github:
        # Check if it hit GitHub tools or returned a fallback message
        if not is_admin_fallback and not is_info_fallback:
            github_client.call_tool.assert_called()
        mock_acompletion.assert_not_called()
    else:
        # Check if it hit the LLM (Guardian check now removed from this path)
        mock_acompletion.assert_called()
        safety_client.call_tool.assert_not_called()
