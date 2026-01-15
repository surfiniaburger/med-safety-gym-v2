"""
Tests for the DIPGSafetyEnv client.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from med_safety_gym.client import DIPGSafetyEnv, DIPGStepResult
from med_safety_gym.models import DIPGAction, DIPGObservation, DIPGState


def test_client_connection_error():
    """Test that the client handles connection errors gracefully."""
    client = DIPGSafetyEnv(base_url="http://localhost:9999", timeout=1.0)
    
    with pytest.raises((requests.exceptions.ConnectionError, ConnectionError)):
        client.reset()


def test_client_timeout():
    """Test that the client respects the timeout parameter."""
    client = DIPGSafetyEnv(base_url="http://localhost:9999", timeout=0.001)
    
    with pytest.raises((requests.exceptions.ConnectionError, requests.exceptions.Timeout, ConnectionError, TimeoutError)):
        client.reset()


# ==================================================================================
# TESTS FOR INTEGRATED EVALUATION METHODS
# ==================================================================================

def test_evaluate_model_integrated():
    """Test evaluate_model method using integrated steps (mocked)."""
    client = DIPGSafetyEnv(base_url="http://localhost:8000")
    
    # Mocking reset and step
    mock_obs = DIPGObservation(context="ctx", question="q")
    mock_step_result = DIPGStepResult(
        observation=mock_obs,
        reward=5.0,
        done=True,
        info={"safe": True, "refusal": False}
    )
    
    with patch.object(client, 'reset', return_value=DIPGStepResult(observation=mock_obs, done=False)):
        with patch.object(client, 'step', return_value=mock_step_result):
            responses = ["response1"]
            result = client.evaluate_model(responses)
            
            assert result['total_responses'] == 1
            assert result['mean_reward'] == 5.0
            assert result['safe_response_rate'] == 1.0


def test_get_metrics_summary_integrated():
    """Test get_metrics_summary method using integrated state (mocked)."""
    client = DIPGSafetyEnv(base_url="http://localhost:8000")
    
    mock_state = DIPGState(
        current_context="ctx",
        current_question="q",
        config={"environment": "Test Gym", "response_format": "json"}
    )
    
    with patch.object(client, 'state', return_value=mock_state):
        result = client.get_metrics_summary()
        
        assert result['environment'] == "Test Gym"
        assert result['response_format'] == "json"


def test_invalid_action():
    """Test that the client raises an error for an invalid action."""
    # This test requires a running server, so we'll skip it for now.
    pass

def test_server_timeout():
    """Test that the client raises an error for a server timeout."""
    # This test requires a running server that can be made to hang, so we'll skip it for now.
    pass