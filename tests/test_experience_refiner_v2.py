import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from med_safety_gym.experience_refiner import ExperienceRefiner

@pytest.mark.asyncio
async def test_distill_guidelines_from_semantic_traces():
    """
    Canon TDD: Step 1 - Test for structured semantic trace distillation.
    Verify that the refiner can distill guidelines using only structured data (zero raw text).
    """
    refiner = ExperienceRefiner()
    
    # Mock semantic traces (The 'Anatomy of the Failure' or 'Anatomy of Success')
    # This structure is what we want to transition to, removing raw messages.
    mock_traces = [
        {
            "turn_id": 1,
            "intent": "REFINEMENT",
            "is_success": False,
            "failure_reason": "EntityParityViolation: Found ONC201 not in context",
            "detected_entities": ["ONC201"],
            "context_entities": ["Panobinostat"]
        },
        {
            "turn_id": 2,
            "intent": "FOLLOW_UP",
            "is_success": True,
            "detected_entities": ["Panobinostat"]
        }
    ]
    
    # Mock database call to return these traces
    # We will assume a new method 'get_semantic_traces' or similar
    with patch.object(ExperienceRefiner, "_get_user_traces", return_value=mock_traces):
        # Mock the LLM call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "When the user tries to refine with ONC201, ensure it is in the Dipg context."
        
        with patch("med_safety_gym.experience_refiner.acompletion", AsyncMock(return_value=mock_response)) as mock_llm:
            guidelines = await refiner.distill_guidelines("test_user")
            
            assert "ONC201" in guidelines
            assert mock_llm.called
            
            # Verify that the prompt sent to the LLM contains the semantic trace but NOT raw user text
            # (We'll verify this more strictly once we define the prompting logic)
            call_args = mock_llm.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "REFINEMENT" in prompt
            assert "EntityParityViolation" in prompt
            # Strict Zero-Injection Check: Ensure no raw text placeholders are in the prompt
            assert "user_message" not in prompt.lower()
