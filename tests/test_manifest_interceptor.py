"""
Tests for ManifestInterceptor orchestrator.
TDD Test List (Beck Canon):
1. Allowed tool with allowed args → passes
2. Disallowed tool name → blocked with reason
3. Tool with network arg to disallowed domain → blocked
4. Tool with filesystem arg containing path traversal → blocked
5. Audit log records blocked actions
"""
import pytest
from med_safety_gym.skill_manifest import SkillManifest
from med_safety_gym.manifest_interceptor import ManifestInterceptor


@pytest.fixture
def interceptor():
    """Create an interceptor with a test manifest."""
    manifest = SkillManifest.from_dict({
        "name": "test-skill",
        "version": "1.0.0",
        "permissions": {
            "net": ["api.github.com"],
            "fs": ["./workspace"],
            "tools": ["list_issues", "create_issue"],
        },
    })
    return ManifestInterceptor(manifest)


def test_allowed_tool_passes(interceptor):
    result = interceptor.intercept("list_issues", {"state": "open"})
    assert result.allowed is True


def test_disallowed_tool_blocked(interceptor):
    result = interceptor.intercept("delete_repo", {})
    assert result.allowed is False
    assert "delete_repo" in result.reason


def test_network_arg_disallowed_blocked(interceptor):
    result = interceptor.intercept("list_issues", {"url": "https://evil.com/data"})
    assert result.allowed is False
    assert "evil.com" in result.reason


def test_filesystem_traversal_blocked(interceptor):
    result = interceptor.intercept("create_issue", {"path": "../../etc/shadow"})
    assert result.allowed is False


def test_audit_log_records_blocked(interceptor):
    interceptor.intercept("delete_repo", {})
    assert len(interceptor.audit_entries) == 1
    entry = interceptor.audit_entries[0]
    assert entry["tool"] == "delete_repo"
    assert entry["allowed"] is False
