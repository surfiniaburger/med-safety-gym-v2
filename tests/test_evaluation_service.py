"""
Tests for evaluation_service.py
"""

import pytest
import json
from pathlib import Path
from med_safety_gym.evaluation_service import EvaluationManager, EvaluationRequest, EvaluationResult
from med_safety_gym.dipg_environment import DIPGEnvironment
from med_safety_gym.format_parser import ResponseFormat


@pytest.fixture
def test_env():
    """Create a test environment"""
    return DIPGEnvironment(
        dataset_path="tests/mock_dataset.jsonl",
        conflict_reward=10.0,
        abstain_reward=10.0,
        hallucination_penalty=-20.0,
        missing_answer_penalty=-15.0,
        hallucinated_trace_penalty=-25.0,
        proof_inconsistency_penalty=-20.0,
        incorrect_answer_penalty=-20.0,
        conflict_penalty=-15.0,
        abstain_penalty=-15.0,
        missing_trace_penalty=-15.0,
        correct_abstention_reward=15.0,
        verifiable_trace_reward=10.0,
        correct_synthesis_reward=10.0,
        exact_format_reward=10.0,
        format_mismatch_penalty=-10.0,
        no_hallucination_reward=1.0,
        analysis_channel_start="<|channel|>analysis<|message|>",
        proof_channel_start="<|channel|>proof<|message|>",
        final_channel_start="<|channel|>final<|message|>",
        channel_end="<|end|>",
        response_format=ResponseFormat.CUSTOM_TAGS
    )


@pytest.fixture
def eval_manager(test_env):
    """Create an evaluation manager"""
    return EvaluationManager(test_env)


def test_evaluate_batch_custom_tags(eval_manager):
    """Test batch evaluation with custom tags format"""
    responses = [
        "<|channel|>analysis<|message|>Test analysis<|end|><|channel|>proof<|message|>Test proof<|end|><|channel|>final<|message|>Test final<|end|>",
        "<|channel|>analysis<|message|>Another analysis<|end|><|channel|>proof<|message|>Another proof<|end|><|channel|>final<|message|>Another final<|end|>"
    ]
    
    result = eval_manager.evaluate_batch(responses, ResponseFormat.CUSTOM_TAGS)
    
    assert isinstance(result, EvaluationResult)
    assert result.total_responses == 2
    assert len(result.rewards) == 2
    assert isinstance(result.mean_reward, float)
    assert isinstance(result.median_reward, float)


def test_evaluate_batch_json(eval_manager):
    """Test batch evaluation with JSON format"""
    responses = [
        '{"analysis": "Test analysis", "proof": "Test proof", "final": "Test final"}',
        '{"analysis": "Another analysis", "proof": "Another proof", "final": "Another final"}'
    ]
    
    result = eval_manager.evaluate_batch(responses, ResponseFormat.JSON)
    
    assert isinstance(result, EvaluationResult)
    assert result.total_responses == 2
    assert len(result.rewards) == 2


def test_evaluate_batch_with_save(eval_manager, tmp_path):
    """Test batch evaluation with save to file"""
    from unittest.mock import patch
    
    responses = [
        "<|channel|>analysis<|message|>Test<|end|><|channel|>proof<|message|>Test<|end|><|channel|>final<|message|>Test<|end|>"
    ]
    
    # Use relative path and mock cwd to point to tmp_path
    relative_path = "eval_results.json"
    
    with patch('pathlib.Path.cwd', return_value=tmp_path):
        result = eval_manager.evaluate_batch(
            responses,
            ResponseFormat.CUSTOM_TAGS,
            save_path=relative_path
        )
        
        expected_path = tmp_path / relative_path
        assert result.saved_to == str(expected_path)
        assert expected_path.exists()
        
        # Verify saved content
        with open(expected_path) as f:
            saved_data = json.load(f)
        
        assert "summary" in saved_data
        assert "detailed_results" in saved_data
        assert "timestamp" in saved_data


def test_evaluate_batch_handles_errors(eval_manager):
    """Test that batch evaluation handles malformed responses gracefully"""
    responses = [
        '{"malformed": "json"',  # Will fail parsing (ValueError) -> fallback -> format penalty
        '{"analysis": "Valid", "proof": "Valid", "final": "Valid"}'
    ]
    
    result = eval_manager.evaluate_batch(responses, ResponseFormat.JSON)
    
    # Should still return results for all responses
    assert result.total_responses == 2
    assert len(result.rewards) == 2
    # First response should get penalty for format error (falls back to legacy parsing)
    assert result.rewards[0] == eval_manager.environment.format_mismatch_penalty


def test_get_metrics_summary(eval_manager):
    """Test metrics summary endpoint"""
    summary = eval_manager.get_metrics_summary()
    
    assert "environment" in summary
    assert "response_format" in summary
    assert "reward_configuration" in summary
    assert summary["environment"] == "DIPG Safety Gym"
    assert summary["response_format"] == "custom_tags"


def test_evaluation_request_model():
    """Test EvaluationRequest pydantic model"""
    request = EvaluationRequest(
        responses=["test1", "test2"],
        format=ResponseFormat.JSON,
        save_path="/tmp/results.json"
    )
    
    assert len(request.responses) == 2
    assert request.format == ResponseFormat.JSON
    assert request.save_path == "/tmp/results.json"


def test_evaluation_request_defaults():
    """Test EvaluationRequest default values"""
    request = EvaluationRequest(responses=["test"])
    
    assert request.format == ResponseFormat.AUTO
    assert request.save_path is None


def test_evaluation_result_model():
    """Test EvaluationResult pydantic model"""
    result = EvaluationResult(
        total_responses=3,
        mean_reward=5.0,
        median_reward=5.0,
        std_reward=1.0,
        min_reward=4.0,
        max_reward=6.0,
        rewards=[4.0, 5.0, 6.0]
    )
    
    assert result.total_responses == 3
    assert result.mean_reward == 5.0
    assert len(result.rewards) == 3
