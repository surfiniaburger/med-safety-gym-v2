import pytest
from fastapi.testclient import TestClient
from med_safety_eval.observability_hub import app

client = TestClient(app)

def test_root_governor_status():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "governor_active" in data
    assert data["governor_active"] is True

def test_get_manifest():
    response = client.get("/manifest")
    assert response.status_code == 200
    data = response.json()
    assert "manifest" in data
    assert "signature" in data
    
    manifest = data["manifest"]
    assert manifest["name"] == "safeclaw-core"
    assert "permissions" in manifest
    assert "tools" in manifest["permissions"]

def test_get_tool_tier():
    # Test a user tool
    response = client.get("/manifest/tier/list_issues")
    assert response.status_code == 200
    assert response.json()["tier"] == "user"

    # Test an admin tool
    response = client.get("/manifest/tier/delete_issue_comment")
    assert response.status_code == 200
    assert response.json()["tier"] == "admin"

    # Test a critical tool
    response = client.get("/manifest/tier/delete_repo")
    assert response.status_code == 200
    assert response.json()["tier"] == "critical"

    # Test unknown tool
    response = client.get("/manifest/tier/unknown_tool")
    assert response.status_code == 200
    assert response.json()["tier"] == "denied"

def test_auth_delegate():
    response = client.post("/auth/delegate", json={"session_id": "test_sess_1", "profile": "read_only"})
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "scope" in data
    assert "expires_at" in data

    # Verify we can decode it with the pubkey
    token = data["token"]
    from med_safety_gym.identity.scoped_identity import verify_delegation_token
    # For now, just checking it exists, testing verification logic in identity tests
    assert token.count(".") == 2

def test_manifest_scoped():
    # First get a token
    auth_resp = client.post("/auth/delegate", json={"session_id": "test_sess_2", "profile": "read_only"})
    token = auth_resp.json()["token"]

    # Now get the scoped manifest
    response = client.get("/manifest/scoped", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert "manifest" in data
    assert "signature" in data
    
    # Read only profile shouldn't have dangerous tools
    manifest = data["manifest"]
    # We expect tools property to be a list in the filtered dict
    tool_names = [t["name"] for t in manifest.get("tools", [])]
    assert "list_issues" in tool_names or len(tool_names) > 0 # At least some read tool
    assert "delete_repo" not in tool_names

