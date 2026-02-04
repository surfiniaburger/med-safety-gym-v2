from fastapi.testclient import TestClient
from med_safety_gym.app import app

client = TestClient(app)

def test_websocket_broadcast():
    session_id = "test_session"
    
    with client.websocket_connect(f"/ws/gauntlet/{session_id}") as websocket:
        # Simulate training loop sending data
        snapshot = {
            "timestamp": 1234567890,
            "scores": {"root": 1.0, "format": 1.0}
        }
        
        response = client.post(f"/gauntlet/stream/{session_id}", json=snapshot)
        assert response.status_code == 200
        
        # Verify UI received it
        data = websocket.receive_json()
        assert data == snapshot
