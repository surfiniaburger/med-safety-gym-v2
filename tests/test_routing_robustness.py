import pytest
import asyncio
import re
from unittest.mock import MagicMock
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, Part, TextPart

@pytest.mark.asyncio
async def test_routing_robustness():
    """
    Verify that medical phrases containing github keywords as substrings 
    (but not as word-boundary-tokens) do not trigger GitHub routing.
    """
    agent = SafeClawAgent()
    updater = MagicMock()
    updater.update_status = MagicMock(return_value=asyncio.Future())
    updater.update_status.return_value.set_result(None)

    # Mock context_aware_action to prevent actual MCP calls
    agent.context_aware_action = MagicMock(return_value=asyncio.Future())
    agent.context_aware_action.return_value.set_result(None)
    
    # Mock github_action to detect if it's called
    agent.github_action = MagicMock(return_value=asyncio.Future())
    agent.github_action.return_value.set_result(None)

    # 1. "pull the records" -> Should NOT be GitHub (even though it contains 'pull')
    # Wait, 'pull' IS a GitHub keyword. But if it's "pull records", it might be a false positive.
    # The fix used word boundaries for 'pull' in the detection logic?
    # Actually, current logic is:
    # github_keywords = {..., 'pull', ...}
    # is_github = any(k in action.lower() for k in github_keywords) or re.search(r'\b(pull|pr)\b', ...)
    
    # Wait, if 'pull' is in github_keywords, 'pull records' will STILL be github.
    # The review suggested: any(k in action.lower() for k in github_keywords) or re.search(r'\b(pull|pr)\b', ...)
    # BUT 'pull' was ALREADY in github_keywords. 
    # To truly fix it, we should remove the brittle ones from the set and ONLY use regex for them.
    
    # Test: "pull the patient records"
    action = "pull the patient records"
    message = Message(
        role="user",
        messageId="test-id",
        parts=[Part(root=TextPart(kind="text", text=action))]
    )
    
    await agent.run(message, updater)
    
    # If it routed to GitHub, github_action was called.
    # If it routed to Medical, context_aware_action was called.
    assert not agent.github_action.called, "Should not route 'pull the patient records' to GitHub"
    assert agent.context_aware_action.called
    
    # Reset mocks
    agent.github_action.reset_mock()
    agent.context_aware_action.reset_mock()
    
    # 2. "gh: pull" -> SHOULD be GitHub
    action = "gh: pull"
    message = Message(
        role="user",
        messageId="test-id",
        parts=[Part(root=TextPart(kind="text", text=action))]
    )
    await agent.run(message, updater)
    assert agent.github_action.called

if __name__ == "__main__":
    asyncio.run(test_routing_robustness())
