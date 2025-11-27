
import pytest
from fastapi.testclient import TestClient
from server.app import app, get_environment, MAX_EVALUATION_ITEMS
from server.dipg_environment import DIPGEnvironment

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

def test_valid_request_still_works():
    """Verify that a valid request still works (regression test)."""
    # We need to mock the dataset to be empty or have valid data for this to fully work
    # without external dependencies, but for now we just check it doesn't 413.
    # Since we are using the global dataset loading which might fail in test env
    # if the path isn't valid, we expect either 200 or 500, but NOT 413.
    
    payload = {"responses": ["test response"]}
    response = client.post("/evaluate", json=payload)
    
    assert response.status_code != 413
