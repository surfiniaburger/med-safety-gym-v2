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
            "tools": {
                "user": ["list_issues", "create_issue"],
                "admin": ["delete_repo"]
            },
        },
    })
    return ManifestInterceptor(manifest)


def test_allowed_tool_passes(interceptor):
    result = interceptor.intercept("list_issues", {"state": "open"})
    assert result.allowed is True


def test_disallowed_tool_blocked(interceptor):
    result = interceptor.intercept("unknown_tool", {})
    assert result.allowed is False
    assert "not declared" in result.reason


def test_network_arg_disallowed_blocked(interceptor):
    result = interceptor.intercept("list_issues", {"url": "https://evil.com/data"})
    assert result.allowed is False
    assert "evil.com" in result.reason


def test_filesystem_traversal_blocked(interceptor):
    result = interceptor.intercept("create_issue", {"path": "../../etc/shadow"})
    assert result.allowed is False


def test_audit_log_records_blocked(interceptor):
    # Pass mutable audit log
    audit_log = []
    interceptor.intercept("unknown_tool", {}, audit_log=audit_log)
    assert len(audit_log) == 1
    entry = audit_log[0]
    assert entry["tool"] == "unknown_tool"
    assert entry["allowed"] is False


def test_admin_tool_blocked_by_default(interceptor):
    """Admin tools require explicit escalation."""
    result = interceptor.intercept("delete_repo", {}, escalated_tools=set())
    assert result.allowed is False
    assert "escalation" in result.reason
    assert result.tier == "admin"


def test_escalation_unlocks_admin_tool(interceptor):
    """Passing escalated_tools allows the admin tool."""
    # Session state
    escalated = set()

    # Initially blocked
    assert interceptor.intercept("delete_repo", {}, escalated_tools=escalated).allowed is False
    
    # Escalate (simulate session update)
    escalated.add("delete_repo")
    
    # Now allowed
    result = interceptor.intercept("delete_repo", {}, escalated_tools=escalated)
    assert result.allowed is True
    assert result.tier == "admin"


def test_escalate_all_admin_unlocks_everything(interceptor):
    escalated = set()
    # Simulate "escalate all" logic (now outside interceptor)
    for tool in interceptor.manifest.permissions.tools.admin:
        escalated.add(tool)

    assert interceptor.intercept("delete_repo", {}, escalated_tools=escalated).allowed is True
