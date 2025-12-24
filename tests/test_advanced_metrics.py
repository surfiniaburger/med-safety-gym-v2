
import pytest
from unittest.mock import MagicMock
from med_safety_gym.evaluation_service import EvaluationManager
from med_safety_gym.dipg_environment import DIPGEnvironment
from med_safety_gym.format_parser import ResponseFormat
from med_safety_gym.models import DIPGAction, DIPGObservation, DIPGState
from openenv_core.http_env_client import StepResult

class MockStepResult:
    def __init__(self, reward):
        self.reward = reward

def test_advanced_metrics_aggregation():
    """Verify that EvaluationManager correctly aggregates metrics from the environment."""
    
    # Mock Environment
    env = MagicMock(spec=DIPGEnvironment)
    env.response_format = ResponseFormat.CUSTOM_TAGS
    env.missing_answer_penalty = -1.0
    
    # Setup mock behavior for reset and step
    env.reset.return_value = DIPGObservation(context="ctx", question="q")
    
    # We will simulate 4 responses:
    # 1. Refusal
    # 2. Safe & Consistent
    # 3. Hallucination
    # 4. Inconsistent (Safe=False)
    
    # Define the sequence of metrics returned by env.last_metrics
    metrics_sequence = [
        {"refusal": True, "safe": True, "hallucination": False, "inconsistency": False, "format_error": False}, # 1. Refusal
        {"refusal": False, "safe": True, "hallucination": False, "inconsistency": False, "format_error": False}, # 2. Safe
        {"refusal": False, "safe": False, "hallucination": True, "inconsistency": False, "format_error": False}, # 3. Hallucination
        {"refusal": False, "safe": False, "hallucination": False, "inconsistency": True, "format_error": False}, # 4. Inconsistent
    ]
    
    # Mock step to return a dummy result and update last_metrics side-effect
    # We need a side effect that updates env.last_metrics based on the call count
    
    # Create a side effect function
    call_count = 0
    def step_side_effect(action):
        nonlocal call_count
        metrics = metrics_sequence[call_count]
        env.last_metrics = metrics # Update the property mock
        call_count += 1
        return MockStepResult(reward=1.0)

    env.step.side_effect = step_side_effect
    
    # We also need to mock the property access. 
    # Since we are setting it on the mock object in the side effect, 
    # we can just use a PropertyMock or just let the mock handle attributes.
    # But MagicMock attributes are mocks by default.
    # Let's make last_metrics a property that returns the value set in side_effect.
    # Actually, simpler: just set the attribute in side_effect.
    
    manager = EvaluationManager(env)
    
    responses = ["r1", "r2", "r3", "r4"]
    result = manager.evaluate_batch(responses)
    
    # Verify Rates
    # Total = 4
    # Refusal = 1 -> 0.25
    # Safe = 2 (Refusal is safe, Safe is safe) -> 0.5
    # Hallucination = 1 -> 0.25
    # Consistency = 1 (Only the Safe one. Refusal is safe but refusal=True. Hallucination/Inconsistent are safe=False) -> 0.25
    
    assert result.total_responses == 4
    assert result.refusal_rate == 0.25
    assert result.safe_response_rate == 0.5
    assert result.medical_hallucination_rate == 0.25
    assert result.reasoning_consistency_rate == 0.25
    
    # Verify Per-Sample Outcomes
    assert result.refusal_outcomes == [1, 0, 0, 0]
    assert result.safe_outcomes == [1, 1, 0, 0]
    assert result.hallucination_outcomes == [0, 0, 1, 0]
    assert result.consistency_outcomes == [0, 1, 0, 0]

def test_dipg_environment_metrics_logic():
    """Verify that DIPGEnvironment calculates metrics correctly."""
    # We need a real environment instance but with mocked helper methods to control logic
    env = DIPGEnvironment(
        dataset_path="tests/mock_dataset.jsonl",
        conflict_reward=0, abstain_reward=0, hallucination_penalty=0, missing_answer_penalty=0,
        hallucinated_trace_penalty=0, proof_inconsistency_penalty=0, incorrect_answer_penalty=0,
        conflict_penalty=0, abstain_penalty=0, missing_trace_penalty=0, correct_abstention_reward=0,
        verifiable_trace_reward=0, correct_synthesis_reward=0, exact_format_reward=0,
        format_mismatch_penalty=0, no_hallucination_reward=0,
        analysis_channel_start="<A>", proof_channel_start="<P>", final_channel_start="<F>", channel_end="</>"
    )
    
    # Mock helpers
    env.is_perfectly_formatted = MagicMock(return_value=True)
    env._parse_response = MagicMock(return_value={"proof": "p", "final": "f"})
    env.is_grounded = MagicMock()
    env.supports = MagicMock()
    env.is_refusal = MagicMock()
    env.is_correct_synthesis = MagicMock(return_value=True)
    
    # Case 1: Refusal
    env.is_refusal.return_value = True
    env.is_grounded.return_value = True
    env.supports.return_value = True
    
    _, metrics = env.calculate_total_reward("response", "ctx", {})
    assert metrics["refusal"] is True
    assert metrics["safe"] is True
    
    # Case 2: Hallucination
    env.is_refusal.return_value = False
    env.is_grounded.return_value = False # Not grounded -> Hallucination
    
    _, metrics = env.calculate_total_reward("response", "ctx", {})
    assert metrics["hallucination"] is True
    assert metrics["safe"] is False
    
    # Case 3: Inconsistency
    env.is_refusal.return_value = False
    env.is_grounded.return_value = True
    env.supports.return_value = False # Not supports -> Inconsistency
    
    _, metrics = env.calculate_total_reward("response", "ctx", {})
    assert metrics["inconsistency"] is True
    assert metrics["safe"] is False
    
    # Case 4: Safe & Consistent
    env.is_refusal.return_value = False
    env.is_grounded.return_value = True
    env.supports.return_value = True
    
    _, metrics = env.calculate_total_reward("response", "ctx", {})
    assert metrics["safe"] is True
    assert metrics["refusal"] is False
    assert metrics["hallucination"] is False
    assert metrics["inconsistency"] is False
