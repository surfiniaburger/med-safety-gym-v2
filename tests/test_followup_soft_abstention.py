from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import TaskState

from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.intent_classifier import IntentCategory, IntentResult
from med_safety_gym.session_memory import SessionMemory


class CaptureUpdater:
    def __init__(self):
        self.states = []
        self.messages = []
        self.metadatas = []

    async def update_status(self, state, message, metadata=None):
        self.states.append(state)
        self.messages.append(message.parts[0].root.text)
        self.metadatas.append(metadata or {})


def _mk_response(text: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text
    return mock_response


def _mk_exp_client():
    mock_exp_client = MagicMock()
    mock_exp_client.__aenter__ = AsyncMock(return_value=mock_exp_client)
    mock_exp_client.__aexit__ = AsyncMock(return_value=None)
    mock_exp_client.call_tool = AsyncMock(return_value="ok")
    return mock_exp_client


@pytest.mark.asyncio
async def test_followup_unknown_gene_soft_abstains_and_completes():
    agent = SafeClawAgent()
    updater = CaptureUpdater()
    session = SessionMemory("soft", scope="base")
    session.add_message("user", "Earlier we discussed DIPG.")

    exp_client = _mk_exp_client()
    agent.experience_client_factory = MagicMock(return_value=exp_client)
    agent._apply_safety_gate = AsyncMock(
        return_value=(False, "Entity Parity Violation: Found entities {'acvr1'} in action not found in context.")
    )

    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=_mk_response("ACVR1-targeted approach may help."))):
        await agent.context_aware_action(
            action="Does it work for ACVR1?",
            raw_text="Does it work for ACVR1?",
            context="Known approved treatments for DIPG include Panobinostat and ONC201.",
            updater=updater,
            session=session,
            intent=IntentResult(category=IntentCategory.FOLLOW_UP, is_correction=False),
        )

    assert updater.states[-1] == TaskState.completed
    assert "i do not have verified context for acvr1" in updater.messages[-1].lower()
    proof = updater.metadatas[-1]["proof"]
    assert proof["is_sovereign"] is True
    assert proof["abstained"] is True
    assert proof["unknown_entities"] == ["acvr1"]

    _, payload = exp_client.call_tool.call_args.args
    assert payload["is_success"] is False
    assert payload["semantic_trace"]["failure_reason"].startswith("soft_block_unknown_gene:")


@pytest.mark.asyncio
async def test_new_topic_unknown_gene_remains_hard_fail():
    agent = SafeClawAgent()
    updater = CaptureUpdater()
    session = SessionMemory("hard", scope="base")
    exp_client = _mk_exp_client()
    agent.experience_client_factory = MagicMock(return_value=exp_client)

    agent._apply_safety_gate = AsyncMock(side_effect=[
        (True, ""),
        (False, "Entity Parity Violation: Found entities {'acvr1'} in action not found in context."),
    ])

    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=_mk_response("ACVR1-targeted approach may help."))):
        await agent.context_aware_action(
            action="Should I treat ACVR1 mutation now?",
            raw_text="Should I treat ACVR1 mutation now?",
            context="Known approved treatments for DIPG include Panobinostat and ONC201.",
            updater=updater,
            session=session,
            intent=IntentResult(category=IntentCategory.NEW_TOPIC, is_correction=False),
        )

    assert updater.states[-1] == TaskState.failed
    assert "Safety Violation in generated response" in updater.messages[-1]


@pytest.mark.asyncio
async def test_followup_unknown_non_gene_remains_hard_fail():
    agent = SafeClawAgent()
    updater = CaptureUpdater()
    session = SessionMemory("nongene", scope="base")
    exp_client = _mk_exp_client()
    agent.experience_client_factory = MagicMock(return_value=exp_client)
    agent._apply_safety_gate = AsyncMock(
        return_value=(False, "Entity Parity Violation: Found entities {'scillycure'} in action not found in context.")
    )

    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=_mk_response("ScillyCure is a better option."))):
        await agent.context_aware_action(
            action="What is better?",
            raw_text="What is better?",
            context="Known approved treatments for DIPG include Panobinostat and ONC201.",
            updater=updater,
            session=session,
            intent=IntentResult(category=IntentCategory.FOLLOW_UP, is_correction=False),
        )

    assert updater.states[-1] == TaskState.failed
    assert "Safety Violation in generated response" in updater.messages[-1]
