import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.session_memory import SessionMemory
from a2a.types import Message, Part, TextPart

@pytest.mark.asyncio
async def test_output_parity_uses_verified_context_only():
    agent = SafeClawAgent()
    agent.model = 'test_model'
    # Mock LLM to return an answer containing a drug mentioned in history
    agent._apply_safety_gate = AsyncMock(return_value=(True, ""))
    
    # Mock liteLLM
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "I remember we discussed Pembrolizumab."

    # Setup session with history containing the drug
    session = SessionMemory(user_id="123")
    session._messages.append({"role": "user", "content": "What about Pembrolizumab?"})
    session._messages.append({"role": "assistant", "content": "It is an immunotherapy drug."})

    updater = AsyncMock()
    
    # Current action and context don't have the drug
    static_context = "This is a generic DIPG context without specific drug names."
    current_action = "Can you summarize our conversation?"
    
    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=mock_resp)):
        await agent.context_aware_action(current_action, current_action, static_context, updater, session=session, intent=None)
    
    # Security invariant: post-generation parity must use only verified context.
    # Unverified chat history/user prompt must not widen allowed entities.
    args, _ = agent._apply_safety_gate.call_args
    passed_response, passed_output_context, _ = args
    
    assert passed_output_context == static_context
    assert "Pembrolizumab" not in passed_output_context


@pytest.mark.asyncio
async def test_run_prevents_history_taint_in_parity_context():
    agent = SafeClawAgent()
    agent.model = "test_model"
    session = SessionMemory(user_id="poison")
    # Unverified user-introduced entity that should never become parity allowlist context.
    session._messages.append({"role": "user", "content": "ScillyCure fixed my symptoms."})

    # Mock LLM response and intercept safety-gate calls.
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "I can discuss Panobinostat for DIPG."
    original_gate = agent._apply_safety_gate
    agent._apply_safety_gate = AsyncMock(side_effect=original_gate)

    updater = AsyncMock()
    msg = Message(
        role="user",
        messageId="history-taint-test",
        parts=[Part(root=TextPart(kind="text", text="What is DIPG?"))],
    )

    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=mock_resp)):
        await agent.run(msg, updater, session=session)

    # NEW_TOPIC path calls gate twice: input and output.
    first_ctx = agent._apply_safety_gate.call_args_list[0].args[1]
    second_ctx = agent._apply_safety_gate.call_args_list[1].args[1]
    assert "ScillyCure" not in first_ctx
    assert "ScillyCure" not in second_ctx
