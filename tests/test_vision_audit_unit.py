import pytest
import os
from med_safety_gym.vision_audit import VisionAuditService

def test_vision_audit_key_retrieval():
    """Verify that GOOGLE_API_KEY is prioritized."""
    os.environ["GOOGLE_API_KEY"] = "google_key"
    os.environ["GEMINI_API_KEY"] = "gemini_key"
    service = VisionAuditService()
    assert service.api_key == "google_key"
    
    del os.environ["GOOGLE_API_KEY"]
    service = VisionAuditService()
    assert service.api_key == "gemini_key"

def test_vision_audit_url_construction():
    """Verify the Gemini 2.0 URL construction."""
    service = VisionAuditService(api_key="test_key")
    url = service._get_url()
    assert "gemini-2.0-flash" in url
    assert "key=test_key" in url

def test_vision_audit_payload():
    """Verify the generation payload structure."""
    service = VisionAuditService()
    payload = service._build_payload("delete_repo", {"repo": "test"})
    assert "contents" in payload
    assert "delete_repo" in payload["contents"][0]["parts"][0]["text"]
    assert payload["generationConfig"]["temperature"] == 0.0
