import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
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
    
    # We must mock acompletion globally for this test because _apply_safety_gate 
    # now runs AFTER generation and will scan the response.
    # We use entities from BASE_MEDICAL_KNOWLEDGE to pass parity check.
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Panobinostat and ONC201 are treatments for DIPG."
    
    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=mock_response)):
        # Turn 1: New Topic (Should trigger gate on INPUT and OUTPUT)
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
        assert not updater1.is_failed, f"Turn 1 failed: {updater1.responses}"
        # Turn 1: NEW_TOPIC triggers gate on Input AND Output
        assert agent._apply_safety_gate.call_count == 2
        agent._apply_safety_gate.reset_mock()
        
        # Turn 2: Refinement (Should bypass gate on INPUT, but still check OUTPUT)
        updater2 = MockUpdater()
        msg2 = Message(
            role="user", 
            messageId="test-2",
            parts=[Part(root=TextPart(kind="text", text="Actually I meant ONC201."))]
        )
        
        await agent.run(msg2, updater2)
        
        # Verify _apply_safety_gate was called exactly ONCE for the second turn (post-generation check only)
        assert not updater2.is_failed, f"Turn 2 failed: {updater2.responses}"
        assert agent._apply_safety_gate.call_count == 1
        
        # Verify intent classification
        from med_safety_gym.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        intent = classifier.classify("Actually I meant ONC201.")
        assert intent.category == IntentCategory.REFINEMENT

@pytest.mark.asyncio
@pytest.mark.allow_guidelines_loading
async def test_mediator_loading_guidelines():
    agent = SafeClawAgent()
    
    # Mock the experience client
    mock_exp_client = MagicMock()
    mock_exp_client.__aenter__ = AsyncMock(return_value=mock_exp_client)
    mock_exp_client.__aexit__ = AsyncMock()
    mock_exp_client.call_tool = AsyncMock(return_value="When the user says switching, treat as refinement.")
    
    agent.experience_client_factory = MagicMock(return_value=mock_exp_client)
    
    # Mock intent client to verify update
    mock_intent_client = MagicMock()
    mock_intent_client.__aenter__ = AsyncMock(return_value=mock_intent_client)
    mock_intent_client.__aexit__ = AsyncMock()
    mock_intent_client.call_tool = AsyncMock()
    
    agent.intent_client_factory = MagicMock(return_value=mock_intent_client)
    
    await agent._load_pragmatic_guidelines()
    
    # Verify the update tool was called
    # Use assert_any_call to be less sensitive to other possible calls
    mock_intent_client.call_tool.assert_any_call("update_intent_rules", {"guidelines": "When the user says switching, treat as refinement."})
