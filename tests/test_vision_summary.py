import pytest
import os
from med_safety_gym.vision_audit import get_audit_summary

@pytest.mark.asyncio
async def test_vision_audit_summary_smoke():
    """Verify that we can get a summary from the vision audit service (smoke test)."""
    # If API key is missing, it returns a default string, so we can still test.
    summary = await get_audit_summary("delete_repo", {"repo_name": "surfiniaburger/test"})
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Even without API key, it should return a fallback message containing the tool name
    assert "delete_repo" in summary or "agent" in summary.lower()
