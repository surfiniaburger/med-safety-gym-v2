
import pytest
from fastapi.testclient import TestClient
from med_safety_gym.app import app, get_environment, MAX_EVALUATION_ITEMS
from med_safety_gym.dipg_environment import DIPGEnvironment
from med_safety_gym.format_parser import ResponseFormat

client = TestClient(app)

def test_resource_exhaustion_protection():
    """Verify that requests exceeding the item limit are rejected with 413."""
    # Create a payload with too many items
    too_many_items = ["response"] * (MAX_EVALUATION_ITEMS + 1)
    
    response = client.post("/evaluate", json={"responses": too_many_items})
    
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]

def test_environment_isolation():
    """Verify that get_environment creates distinct instances."""
    env1 = get_environment()
    env2 = get_environment()
    
    assert env1 is not env2
    assert isinstance(env1, DIPGEnvironment)
    assert isinstance(env2, DIPGEnvironment)
    
    # Verify they share the same dataset (optimization check)
    assert env1.dataset is env2.dataset

def test_valid_request_still_works(monkeypatch):
    """Verify that a valid request still works and doesn't raise a 500 error."""
    # Mock `get_environment` to return a properly configured test environment.
    # This avoids issues with dataset loading in the test suite and makes the
    # test more robust by asserting a 200 OK status.
    def get_mock_environment():
        dummy_rewards = {
            "conflict_reward": 0, "abstain_reward": 0, "hallucination_penalty": 0, "missing_answer_penalty": 0,
            "hallucinated_trace_penalty": 0, "proof_inconsistency_penalty": 0, "incorrect_answer_penalty": 0,
            "conflict_penalty": 0, "abstain_penalty": 0, "missing_trace_penalty": 0, "correct_abstention_reward": 0,
            "verifiable_trace_reward": 0, "correct_synthesis_reward": 0, "exact_format_reward": 0,
            "format_mismatch_penalty": 0, "no_hallucination_reward": 0,
        }
        dummy_channels = {
            "analysis_channel_start": "", "proof_channel_start": "", "final_channel_start": "", "channel_end": "",
        }
        return DIPGEnvironment(
            dataset_path="tests/mock_dataset.jsonl",
            **dummy_rewards,
            **dummy_channels,
            response_format=ResponseFormat.CUSTOM_TAGS
        )

    monkeypatch.setattr("med_safety_gym.app.get_environment", get_mock_environment)
    
    payload = {"responses": ["test response"]}
    response = client.post("/evaluate", json=payload)
    
    assert response.status_code == 200
    assert "mean_reward" in response.json()
