"""
Tests for the DIPGSafetyEnv client.
"""

import pytest
import requests
from unittest.mock import Mock, patch
from med_safety_gym.client import DIPGSafetyEnv
from med_safety_gym.models import DIPGAction


def test_client_connection_error():
    """Test that the client handles connection errors gracefully."""
    client = DIPGSafetyEnv(base_url="http://localhost:9999", timeout=1.0)
    
    with pytest.raises(requests.exceptions.ConnectionError):
        client.reset()


def test_client_timeout():
    """Test that the client respects the timeout parameter."""
    client = DIPGSafetyEnv(base_url="http://localhost:9999", timeout=0.001)
    
    with pytest.raises((requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        client.reset()


# ==================================================================================
# TESTS FOR NEW EVALUATION METHODS (Phase 5)
# ==================================================================================

@patch('requests.post')
def test_evaluate_model(mock_post):
    """Test evaluate_model method"""
    # Mock the response
    mock_response = Mock()
    mock_response.json.return_value = {
        "total_responses": 2,
        "mean_reward": 5.0,
        "median_reward": 5.0,
        "std_reward": 1.0,
        "min_reward": 4.0,
        "max_reward": 6.0,
        "rewards": [4.0, 6.0]
    }
    mock_post.return_value = mock_response
    
    client = DIPGSafetyEnv(base_url="http://localhost:8000")
    responses = ["response1", "response2"]
    
    result = client.evaluate_model(responses, response_format="json")
    
    # Verify the request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['responses'] == responses
    assert call_args[1]['json']['format'] == "json"
    
    # Verify the result
    assert result['total_responses'] == 2
    assert result['mean_reward'] == 5.0


@patch('requests.post')
def test_evaluate_model_with_save_path(mock_post):
    """Test evaluate_model with save_path parameter"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "total_responses": 1,
        "mean_reward": 10.0,
        "saved_to": "/tmp/results.json"
    }
    mock_post.return_value = mock_response
    
    client = DIPGSafetyEnv(base_url="http://localhost:8000")
    result = client.evaluate_model(
        ["response"],
        response_format="auto",
        save_path="/tmp/results.json"
    )
    
    # Verify save_path was included in request
    call_args = mock_post.call_args
    assert call_args[1]['json']['save_path'] == "/tmp/results.json"
    assert result['saved_to'] == "/tmp/results.json"


@patch('requests.get')
def test_get_metrics_summary(mock_get):
    """Test get_metrics_summary method"""
    # Mock the response
    mock_response = Mock()
    mock_response.json.return_value = {
        "environment": "DIPG Safety Gym",
        "response_format": "custom_tags",
        "reward_configuration": {
            "exact_format_reward": 10.0,
            "correct_abstention_reward": 15.0
        },
        "dataset_size": 100
    }
    mock_get.return_value = mock_response
    
    client = DIPGSafetyEnv(base_url="http://localhost:8000")
    result = client.get_metrics_summary()
    
    # Verify the request was made correctly
    mock_get.assert_called_once()
    assert "http://localhost:8000/metrics/summary" in mock_get.call_args[0][0]
    
    # Verify the result
    assert result['environment'] == "DIPG Safety Gym"
    assert result['response_format'] == "custom_tags"
    assert 'reward_configuration' in result

def test_invalid_action():
    """Test that the client raises an error for an invalid action."""
    # This test requires a running server, so we'll skip it for now.
    pass

def test_server_timeout():
    """Test that the client raises an error for a server timeout."""
    # This test requires a running server that can be made to hang, so we'll skip it for now.
    pass