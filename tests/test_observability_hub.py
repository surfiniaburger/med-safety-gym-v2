import pytest
from fastapi.testclient import TestClient
# This import should fail initially, which is part of the TDD "Red" phase
try:
    from med_safety_eval.observability_hub import app
except ImportError:
    app = None

def test_hub_exists():
    """TDD: Verify that the observability_hub module exists and exposes an app."""
    assert app is not None, "med_safety_eval.observability_hub module or app not found"

@pytest.mark.skipif(app is None, reason="Hub not implemented yet")
def test_hub_websocket_connection():
    """
    Verify that the Hub accepts websocket connections on the expected route.
    """
    client = TestClient(app)
    with client.websocket_connect("/ws/gauntlet/test_session") as websocket:
        # Just connecting and disconnecting is enough to prove the route exists
        assert websocket is not None

@pytest.mark.skipif(app is None, reason="Hub not implemented yet")
def test_hub_broadcast_endpoint():
    """
    Verify that the Hub accepts POST requests to broadcast snapshots.
    """
    client = TestClient(app)
    payload = {"rubric_scores": {"safety": 1.0}}
    response = client.post("/gauntlet/stream/test_session", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
