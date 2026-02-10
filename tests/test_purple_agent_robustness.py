import pytest
from unittest.mock import MagicMock, patch
from med_safety_gym.purple_agent import PurpleAgent
from a2a.types import Message, TaskState
from a2a.server.tasks import TaskUpdater

from med_safety_gym.messenger import create_message

@pytest.mark.anyio
async def test_purple_agent_none_content():
    agent = PurpleAgent()
    message = create_message(text="Test Question")
    updater = MagicMock(spec=TaskUpdater)
    
    # Mock response with None content and None reasoning
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = None
    mock_message.reasoning_content = None
    mock_response.choices = [MagicMock(message=mock_message)]
    
    with patch("med_safety_gym.purple_agent.acompletion", return_value=mock_response):
        await agent.run(message, updater)
    
    # Check if complete was called with a string
    updater.complete.assert_called_once()
    args, _ = updater.complete.call_args
    # The argument to complete should be a Message object
    # We can't easily check the content of the message without knowing a2a internals
    # but we can check if it failed
    updater.failed.assert_not_called()

@pytest.mark.anyio
async def test_purple_agent_exception_handling():
    agent = PurpleAgent()
    message = create_message(text="Test Question")
    updater = MagicMock(spec=TaskUpdater)
    
    # Mock acompletion raising an exception
    with patch("med_safety_gym.purple_agent.acompletion", side_effect=Exception("Test Error")):
        await agent.run(message, updater)
    
    updater.failed.assert_called_once()
    args, _ = updater.failed.call_args
    # The error message should contain "Test Error"
