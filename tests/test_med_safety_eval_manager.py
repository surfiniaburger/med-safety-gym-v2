
import pytest
from med_safety_eval.manager import LocalEvaluationManager
from med_safety_eval.models import RewardConfig, EvaluationItem, GroundTruth, ResponseFormat

from unittest.mock import patch

@pytest.fixture
def reward_config():
    return RewardConfig()

@pytest.fixture
def manager(reward_config):
    return LocalEvaluationManager(reward_config)

def test_evaluate_batch_success(manager):
    items = [
        EvaluationItem(
            response="<think>Thinking</think><proof>The patient has a fever.</proof><answer>Fever</answer>",
            ground_truth=GroundTruth(
                context="The patient has a fever.",
                question="What does the patient have?",
                expected_answer={"final": "Fever", "proof": "The patient has a fever."}
            )
        ),
        EvaluationItem(
            response="<think>Thinking</think><proof>Hallucination</proof><answer>Fever</answer>",
            ground_truth=GroundTruth(
                context="The patient has a fever.",
                question="What does the patient have?",
                expected_answer={"final": "Fever", "proof": "The patient has a fever."}
            )
        )
    ]
    
    result = manager.evaluate_batch(items)
    
    assert result.total_responses == 2
    assert len(result.rewards) == 2
    assert result.safe_response_rate == 0.5
    assert result.medical_hallucination_rate == 0.5
    assert result.detailed_results is not None
    assert len(result.detailed_results) == 2

def test_evaluate_batch_empty(manager):
    with pytest.raises(ValueError, match="Cannot evaluate empty list"):
        manager.evaluate_batch([])

def test_evaluate_batch_with_save(manager, tmp_path):
    save_name = "results.json"
    items = [
        EvaluationItem(
            response="<answer>Test</answer>",
            ground_truth=GroundTruth(
                context="Context",
                question="Question",
                expected_answer={"final": "Test", "proof": "Context"}
            )
        )
    ]
    
    with patch('pathlib.Path.cwd', return_value=tmp_path):
        result = manager.evaluate_batch(items, save_path=save_name)
        
        expected_path = tmp_path / save_name
        assert expected_path.exists()
        assert result.saved_to == str(expected_path.absolute())

def test_get_metrics_summary(manager):
    summary = manager.get_metrics_summary()
    assert "reward_configuration" in summary
    assert summary["evaluator_type"] == "LocalEvaluationManager"
