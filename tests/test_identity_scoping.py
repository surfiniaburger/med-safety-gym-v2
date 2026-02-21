import pytest
from med_safety_gym.identity.scoped_identity import create_scoped_manifest

def test_hub_manifest_scoping():
    # Arrange
    global_manifest = {
        "user": [
            {"name": "read_file", "description": "Reads a file"},
            {"name": "write_file", "description": "Writes to a file"}
        ],
        "admin": [
            {"name": "delete_repo", "description": "Deletes the repository"}
        ]
    }
    allowed_tools = ["read_file", "delete_repo"]

    # Act
    scoped_manifest = create_scoped_manifest(global_manifest, allowed_tools)

    # Assert
    assert len(scoped_manifest["user"]) == 1
    assert scoped_manifest["user"][0]["name"] == "read_file"
    assert len(scoped_manifest["admin"]) == 1
    assert scoped_manifest["admin"][0]["name"] == "delete_repo"
    assert "write_file" not in [t["name"] for t in scoped_manifest["user"]]

def test_token_generation_and_verification():
    from med_safety_gym.identity.scoped_identity import issue_delegation_token, verify_delegation_token
    
    # Arrange
    session_id = "sess_123"
    scope = ["read_file", "list_files"]
    ttl = 3600
    claims = {
        "sub": session_id,
        "scope": scope,
    }
    secret_key = "super_secret_test_key"
    
    # Act
    token = issue_delegation_token(claims, ttl, secret_key)
    
    # Assert
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Verify Act
    decoded = verify_delegation_token(token, secret_key)
    
    # Verify Assert
    assert decoded["sub"] == session_id
    assert decoded["scope"] == scope
    assert "exp" in decoded
