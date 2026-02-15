"""
Tests for policy check functions.
TDD Test List (Beck Canon):
1. check_network — allowed domain passes
2. check_network — disallowed domain blocked
3. check_network — subdomain of allowed domain blocked (exact match)
4. check_filesystem — path inside allowed dir passes
5. check_filesystem — absolute system path blocked
6. check_filesystem — path traversal attack blocked
7. check_tool — allowed tool passes
8. check_tool — disallowed tool blocked
"""
import pytest
from med_safety_gym.policy import (
    check_network,
    check_filesystem,
    check_tool,
    PolicyResult,
)


# --- Network Policy ---

def test_network_allowed_domain_passes():
    result = check_network("api.github.com", ["api.github.com"])
    assert result.allowed is True


def test_network_disallowed_domain_blocked():
    result = check_network("evil.com", ["api.github.com"])
    assert result.allowed is False
    assert "evil.com" in result.reason


def test_network_subdomain_blocked():
    """Subdomains must be explicitly listed — no wildcard matching."""
    result = check_network("sub.api.github.com", ["api.github.com"])
    assert result.allowed is False


# --- Filesystem Policy ---

def test_filesystem_allowed_path_passes():
    result = check_filesystem("./workspace/notes.txt", ["./workspace"])
    assert result.allowed is True


def test_filesystem_absolute_system_path_blocked():
    result = check_filesystem("/etc/passwd", ["./workspace"])
    assert result.allowed is False
    assert "/etc/passwd" in result.reason


def test_filesystem_path_traversal_blocked():
    result = check_filesystem("./workspace/../../etc/passwd", ["./workspace"])
    assert result.allowed is False
    assert "traversal" in result.reason.lower() or "etc/passwd" in result.reason


# --- Tool Policy ---
# NOTE: We use os.path.abspath/realpath in implementation to secure this.
# Tests should verify that semantic containment works.

def test_filesystem_cwd_resolution():
    """Verifies that allowed='./' covers files in CWD."""
    # This might fail with normpath if it just compares '.' vs 'file.txt'
    result = check_filesystem("notes.txt", ["."])
    assert result.allowed is True

def test_filesystem_traversal_realpath():
    """Verifies that '..' resolution is checked against absolute path."""
    # If allowed is /tmp/safe, and we try /tmp/safe/../secret
    # Realpath should become /tmp/secret, catching the escape.
    
    # We simulate this by using CWD.
    # Allowed: CWD/safe
    # Access: CWD/safe/../secret -> CWD/secret (should be blocked)
    
    import os
    cwd = os.getcwd()
    safe_dir = os.path.join(cwd, "safe_dir")
    secret_file = os.path.join(cwd, "secret.txt")
    
    # Accessing secret via "safe_dir/../secret.txt"
    traversal_path = os.path.join(safe_dir, "..", "secret.txt")
    
    result = check_filesystem(traversal_path, [safe_dir])
    assert result.allowed is False
    assert "blocked" in result.reason.lower()


def test_tool_allowed_passes():
    result = check_tool("list_issues", ["list_issues", "create_issue"])
    assert result.allowed is True


def test_tool_disallowed_blocked():
    result = check_tool("delete_repo", ["list_issues", "create_issue"])
    assert result.allowed is False
    assert "delete_repo" in result.reason
