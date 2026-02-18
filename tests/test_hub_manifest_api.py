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
