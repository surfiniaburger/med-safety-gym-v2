import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import Message, Part, TextPart

@pytest.mark.asyncio
async def test_agent_generates_sovereignty_proof():
    """
    Canon TDD: Step 1 - Test for machine-verifiable Sovereignty Proof.
    Verify that SafeClawAgent attaches a structured proof to its responses.
    """
    agent = SafeClawAgent()
    
    # Mock updater to capture results
    class MockUpdater:
        def __init__(self):
            self.final_state = None
            self.final_message = None
            self.proof = None

        async def update_status(self, state, message, metadata=None):
            self.final_state = state
            self.final_message = message
            if metadata and "proof" in metadata:
                self.proof = metadata["proof"]

    updater = MockUpdater()
    
    # Mock LLM and Safety Gate
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Panobinostat is a treatment for DIPG."
    
    msg = Message(
        role="user",
        messageId="test-sovereignty",
        parts=[Part(root=TextPart(kind="text", text="Tell me about Panobinostat."))]
    )
    
    with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=mock_response)):
        # Ensure _apply_safety_gate returns success
        agent._apply_safety_gate = AsyncMock(return_value=(True, ""))
        
        await agent.run(msg, updater)
        
        # Verify that a proof was generated and attached to metadata
        assert updater.proof is not None
        assert "entities" in updater.proof
        assert "panobinostat" in updater.proof["entities"]
        assert updater.proof["intent"] == "NEW_TOPIC"
        assert updater.proof["is_sovereign"] is True
