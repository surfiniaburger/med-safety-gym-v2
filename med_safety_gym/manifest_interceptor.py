"""
SafeClaw Manifest Interceptor — Orchestrator

Wires together the SkillManifest and policy checks.
Sits between the agent and MCP tool execution layer.

Design: Single responsibility — intercept and audit. (Farley)
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib.parse import urlparse

from .skill_manifest import SkillManifest
from .policy import check_tool, check_network, check_filesystem, PolicyResult

logger = logging.getLogger(__name__)


@dataclass
class InterceptResult:
    """Outcome of a full interception check."""
    allowed: bool
    reason: str


class ManifestInterceptor:
    """Checks every tool call against the active manifest."""

    def __init__(self, manifest: SkillManifest):
        self.manifest = manifest
        self.audit_entries: List[Dict[str, Any]] = []

    def intercept(self, tool_name: str, tool_args: Dict[str, Any]) -> InterceptResult:
        """Run all policy checks for a tool call."""
        # 1. Check tool name
        tool_result = check_tool(tool_name, self.manifest.permissions.tools)
        if not tool_result.allowed:
            self._audit(tool_name, tool_args, False, tool_result.reason)
            return InterceptResult(allowed=False, reason=tool_result.reason)

        # 2. Scan args for network URLs
        net_result = self._check_args_for_urls(tool_args)
        if not net_result.allowed:
            self._audit(tool_name, tool_args, False, net_result.reason)
            return InterceptResult(allowed=False, reason=net_result.reason)

        # 3. Scan args for filesystem paths
        fs_result = self._check_args_for_paths(tool_args)
        if not fs_result.allowed:
            self._audit(tool_name, tool_args, False, fs_result.reason)
            return InterceptResult(allowed=False, reason=fs_result.reason)

        self._audit(tool_name, tool_args, True, "")
        return InterceptResult(allowed=True, reason="")

    def _check_args_for_urls(self, args: Dict[str, Any]) -> PolicyResult:
        """Scan tool arguments for URLs and validate domains."""
        for value in args.values():
            if not isinstance(value, str):
                continue
            parsed = urlparse(value)
            if parsed.scheme in ("http", "https") and parsed.hostname:
                result = check_network(parsed.hostname, self.manifest.permissions.net)
                if not result.allowed:
                    return result
        return PolicyResult(allowed=True, reason="")

    def _check_args_for_paths(self, args: Dict[str, Any]) -> PolicyResult:
        """Scan tool arguments for filesystem paths and validate."""
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            # Heuristic: if arg name suggests a path or value looks like one
            if key in ("path", "file", "filepath", "filename", "directory"):
                result = check_filesystem(value, self.manifest.permissions.fs)
                if not result.allowed:
                    return result
            # Also catch values that look like paths
            elif value.startswith("./") or value.startswith("/") or ".." in value:
                result = check_filesystem(value, self.manifest.permissions.fs)
                if not result.allowed:
                    return result
        return PolicyResult(allowed=True, reason="")

    def _audit(self, tool: str, args: Dict, allowed: bool, reason: str):
        """Record the interception for review."""
        entry = {
            "tool": tool,
            "args": args,
            "allowed": allowed,
            "reason": reason,
        }
        self.audit_entries.append(entry)
        level = logging.INFO if allowed else logging.WARNING
        logger.log(level, f"Manifest {'ALLOW' if allowed else 'BLOCK'}: {tool} — {reason}")
