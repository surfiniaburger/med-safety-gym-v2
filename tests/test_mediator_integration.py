import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.intent_classifier import IntentCategory
from a2a.types import Message, Part, TextPart, TaskState

class MockUpdater:
    def __init__(self):
        self.responses = []
        self.state = None
        self.is_failed = False

    async def update_status(self, state, message, metadata=None):
        self.state = state
        if state == TaskState.failed:
            self.is_failed = True
        if hasattr(message.parts[0].root, 'text'):
            self.responses.append(message.parts[0].root.text)

@pytest.mark.asyncio
async def test_mediator_refinement_bypass():
    # Setup agent
    agent = SafeClawAgent()
    
    # Turn 1: New Topic (Should trigger gate)
    updater1 = MockUpdater()
    msg1 = Message(
        role="user", 
        messageId="test-1",
        parts=[Part(root=TextPart(kind="text", text="What is Panobinostat?"))]
    )
    
    # We need to mock _apply_safety_gate to track calls
    original_gate = agent._apply_safety_gate
    agent._apply_safety_gate = AsyncMock(side_effect=original_gate)
    
    await agent.run(msg1, updater1)
    
    # Verification: Should have passed gate
    assert not updater1.is_failed
    assert agent._apply_safety_gate.called
    agent._apply_safety_gate.reset_mock()
    
    # Turn 2: Refinement (Should bypass gate)
    updater2 = MockUpdater()
    msg2 = Message(
        role="user", 
        messageId="test-2",
        parts=[Part(root=TextPart(kind="text", text="Actually I meant ONC201."))]
    )
    
    await agent.run(msg2, updater2)
    
    # Verify _apply_safety_gate was NOT called for the second turn (refinement)
    assert not agent._apply_safety_gate.called
    
    # Verify it classification
    intent = agent.intent_classifier.classify("Actually I meant ONC201.")
    assert intent.category == IntentCategory.REFINEMENT
    
@pytest.mark.asyncio
async def test_mediator_loading_guidelines():
    agent = SafeClawAgent()
    # Mock guidelines distillation using AsyncMock
    agent.experience_refiner.distill_guidelines = AsyncMock(return_value="When the user says switching, treat as refinement.")
    
    await agent._load_pragmatic_guidelines()
    assert agent.intent_classifier.guidelines == "When the user says switching, treat as refinement."
