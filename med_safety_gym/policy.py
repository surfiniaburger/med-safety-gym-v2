"""
SafeClaw Policy Checks — Pure Functions

Each function checks a single invariant and returns a PolicyResult.
No side effects, no I/O. Easy to test. (Farley: <10 lines per function)
"""
import os
from dataclasses import dataclass
from typing import List


@dataclass
class PolicyResult:
    """Outcome of a single policy check."""
    allowed: bool
    reason: str


def check_network(domain: str, allowed_domains: List[str]) -> PolicyResult:
    """Check if a domain is in the allowed list (exact match)."""
    if domain in allowed_domains:
        return PolicyResult(allowed=True, reason="")
    return PolicyResult(
        allowed=False,
        reason=f"Network access to '{domain}' blocked. Allowed: {allowed_domains}",
    )


def check_filesystem(path: str, allowed_paths: List[str]) -> PolicyResult:
    """Check if a file path is within any allowed directory."""
    try:
        # Resolve to absolute path, following symlinks
        # We use os.path.realpath for maximum compatibility and security
        real_path = os.path.realpath(path)
        
        for allowed in allowed_paths:
            # Resolve allowed path too (e.g. if allowed is ".")
            real_allowed = os.path.realpath(allowed)
            
            # Check if real_path starts with real_allowed
            # Use os.path.commonpath to robustly check if the resolved
            # path is contained within an allowed directory.
            if os.path.commonpath([real_allowed, real_path]) == real_allowed:
                return PolicyResult(allowed=True, reason="")
                
    except Exception as e:
        return PolicyResult(allowed=False, reason=f"Path resolution failed: {e}")

    return PolicyResult(
        allowed=False,
        reason=f"Filesystem access to '{path}' blocked. Allowed: {allowed_paths}",
    )


def check_tool(tool_name: str, allowed_tools: List[str]) -> PolicyResult:
    """Check if a tool name is in the allowed list."""
    if tool_name in allowed_tools:
        return PolicyResult(allowed=True, reason="")
    return PolicyResult(
        allowed=False,
        reason=f"Tool '{tool_name}' blocked. Allowed: {allowed_tools}",
    )
