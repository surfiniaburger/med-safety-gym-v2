import pytest
from med_safety_eval.logic import generate_safety_challenge
from med_safety_eval.schemas import NeuralSnapshot

def test_generate_hallucination_challenge():
    # If a model hallucinated (grounding < 0), the challenge should ask to verify the context
    snapshot = NeuralSnapshot(
        session_id="test",
        step=5,
        scores={"root": -20, "grounding": -10},
        metadata={"action": "recommending dose X", "observation": "context only mentions dose Y"}
    )
    
    challenge = generate_safety_challenge(snapshot)
    
    assert challenge["type"] == "HALLUCINATION_GUARD"
    assert "verify" in challenge["question"].lower()
    assert challenge["expected_answer"] is not None

def test_generate_inconsistency_challenge():
    # If a model was inconsistent, the challenge should ask to verify reasoning
    snapshot = NeuralSnapshot(
        session_id="test",
        step=5,
        scores={"root": -10, "inconsistency": -5},
        metadata={"action": "Answer: Yes", "observation": "Thought: No"}
    )
    
    challenge = generate_safety_challenge(snapshot)
    
    assert challenge["type"] == "INCONSISTENCY_GUARD"
    assert "reasoning" in challenge["question"].lower()
