
import pytest
import asyncio
from unittest.mock import MagicMock
from med_safety_gym.telegram_bridge import TelegramUpdater, TelegramBridge
from med_safety_gym.session_memory import SessionMemory
from a2a.types import Message, Part, TextPart, TaskState
from a2a.utils import new_agent_text_message

@pytest.mark.asyncio
async def test_session_history_pollution():
    # Setup
    user_id = "test_user"
    session = SessionMemory(user_id)
    updater = TelegramUpdater()
    
    # Simulate agent status updates
    await updater.update_status(TaskState.working, new_agent_text_message("🛡️ Running safety check..."))
    await updater.update_status(TaskState.working, new_agent_text_message("✅ Analysis complete."))
    await updater.update_status(TaskState.completed, new_agent_text_message("The typical dose is 20mg."))
    
    # Simulate the fixed logic in handle_message:
    # Filter out internal logs/icons before saving to session history
    session_responses = [r for r in updater.responses if not any(marker in r for marker in ["🛡️", "✅", "Checking"])]
    if session_responses:
        session.add_message("assistant", "\n\n".join(session_responses))
    
    messages = session.get_messages()
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    
    print(f"\nFiltered Assistant Message:\n{assistant_msg['content']}")
    
    # Assertions for the FIXED state
    assert "🛡️" not in assistant_msg["content"]
    assert "✅" not in assistant_msg["content"]
    assert assistant_msg["content"] == "The typical dose is 20mg."

if __name__ == "__main__":
    asyncio.run(test_session_history_pollution())
