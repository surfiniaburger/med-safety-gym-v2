import pytest
from unittest.mock import AsyncMock, MagicMock
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.session_memory import SessionMemory

@pytest.mark.asyncio
async def test_multi_turn_history_parity():
    agent = SafeClawAgent()
    agent.model = 'test_model'
    # Mock LLM to return an answer containing a drug mentioned in history
    agent._apply_safety_gate = AsyncMock(return_value=(True, ""))
    
    # Mock liteLLM
    import med_safety_gym.claw_agent
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "I remember we discussed Pembrolizumab."
    med_safety_gym.claw_agent.acompletion = AsyncMock(return_value=mock_resp)

    # Setup session with history containing the drug
    session = SessionMemory(user_id="123")
    session._messages.append({"role": "user", "content": "What about Pembrolizumab?"})
    session._messages.append({"role": "assistant", "content": "It is an immunotherapy drug."})

    updater = AsyncMock()
    
    # Current action and context don't have the drug
    static_context = "This is a generic DIPG context without specific drug names."
    current_action = "Can you summarize our conversation?"
    
    await agent.context_aware_action(current_action, current_action, static_context, updater, session=session, intent=None)
    
    # The safety gate SHOULD be called with a context string that includes the history!
    # Otherwise, "Pembrolizumab" in the response will trigger a false positive.
    args, _ = agent._apply_safety_gate.call_args
    passed_response, passed_output_context, _ = args
    
    assert "Pembrolizumab" in passed_output_context, "Session history was NOT included in the safety gate context!"
