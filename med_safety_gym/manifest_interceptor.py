"""
SafeClaw Manifest Interceptor — Orchestrator

Wires together the SkillManifest and policy checks.
Sits between the agent and MCP tool execution layer.
Tier-aware: user tools pass, write tools pass, admin tools require escalation.

Design: Single responsibility — intercept and audit. (Farley)
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from .skill_manifest import SkillManifest
from .policy import check_tool, check_network, check_filesystem, PolicyResult

logger = logging.getLogger(__name__)


@dataclass
class InterceptResult:
    """Outcome of a full interception check."""
    allowed: bool
    reason: str
    tier: str = ""  # "user", "write", "admin", or "denied"


class ManifestInterceptor:
    """Checks every tool call against the active manifest."""

    def __init__(self, manifest: SkillManifest):
        self.manifest = manifest

    def intercept(self, tool_name: str, tool_args: Dict[str, Any], escalated_tools: Set[str] = None, audit_log: List[Dict] = None) -> InterceptResult:
        """Run all policy checks for a tool call."""
        escalated_tools = escalated_tools or set()
        # 1. Check tool tier
        tier = self.manifest.permissions.tools.tier_for(tool_name)

        if tier == "denied":
            reason = f"Tool '{tool_name}' not declared in manifest"
            self._audit(tool_name, tool_args, False, reason, tier, audit_log)
            return InterceptResult(allowed=False, reason=reason, tier=tier)

        if tier == "admin" and tool_name not in escalated_tools:
            reason = f"Tool '{tool_name}' requires admin escalation. Use 'gh: unlock admin tools' first."
            self._audit(tool_name, tool_args, False, reason, tier, audit_log)
            return InterceptResult(allowed=False, reason=reason, tier=tier)

        # 2. Scan args for network URLs
        net_result = self._check_args_for_urls(tool_args)
        if not net_result.allowed:
            self._audit(tool_name, tool_args, False, net_result.reason, tier, audit_log)
            return InterceptResult(allowed=False, reason=net_result.reason, tier=tier)

        # 3. Scan args for filesystem paths
        fs_result = self._check_args_for_paths(tool_args)
        if not fs_result.allowed:
            self._audit(tool_name, tool_args, False, fs_result.reason, tier, audit_log)
            return InterceptResult(allowed=False, reason=fs_result.reason, tier=tier)

        self._audit(tool_name, tool_args, True, "", tier, audit_log)
        return InterceptResult(allowed=True, reason="", tier=tier)

    # escalate/escalate_all_admin methods removed - responsibility moved to session/agent

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
            if key in ("path", "file", "filepath", "filename", "directory"):
                result = check_filesystem(value, self.manifest.permissions.fs)
                if not result.allowed:
                    return result
            elif value.startswith("./") or value.startswith("/") or ".." in value:
                result = check_filesystem(value, self.manifest.permissions.fs)
                if not result.allowed:
                    return result
        return PolicyResult(allowed=True, reason="")

    def _audit(self, tool: str, args: Dict, allowed: bool, reason: str, tier: str, audit_log: List[Dict] = None):
        """Record the interception for review."""
        entry = {
            "tool": tool,
            "args": args,
            "allowed": allowed,
            "reason": reason,
            "tier": tier,
        }
        if audit_log is not None:
            audit_log.append(entry)
        level = logging.INFO if allowed else logging.WARNING
        logger.log(level, f"Manifest {'ALLOW' if allowed else 'BLOCK'} [{tier}]: {tool} — {reason}")
