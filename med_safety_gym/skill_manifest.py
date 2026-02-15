"""
SafeClaw Skill Manifest — Data Model

Defines the schema for claw_manifest.json files that declare
a skill's required permissions (network, filesystem, tools).

Supports tiered tools: user (always) / write (auth) / admin (hidden).

Design: Pure data, no I/O side effects. Small, focused classes. (Farley)
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolTiers:
    """Tools organized by permission tier."""
    user: List[str] = field(default_factory=list)
    write: List[str] = field(default_factory=list)
    admin: List[str] = field(default_factory=list)

    @property
    def all_tools(self) -> List[str]:
        """Flat list of every tool across all tiers."""
        return self.user + self.write + self.admin

    def tier_for(self, tool_name: str) -> str:
        """Return the tier a tool belongs to, or 'denied'."""
        if tool_name in self.user:
            return "user"
        if tool_name in self.write:
            return "write"
        if tool_name in self.admin:
            return "admin"
        return "denied"


@dataclass
class PermissionSet:
    """Declares what a skill is allowed to access."""
    net: List[str] = field(default_factory=list)
    fs: List[str] = field(default_factory=list)
    tools: ToolTiers = field(default_factory=ToolTiers)


@dataclass
class ScopeConfig:
    """Auto-generated FastMCP scope/visibility config for a tool."""
    auth: str  # "", "write", or "admin"
    tags: List[str] = field(default_factory=list)


@dataclass
class SkillManifest:
    """Top-level manifest for a SafeClaw skill package."""
    name: str
    version: str = "0.0.0"
    permissions: PermissionSet = field(default_factory=PermissionSet)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillManifest":
        """Create a manifest from a dictionary, using safe defaults.
        
        Handles both flat tools list (backward compat) and tiered dict.
        """
        perms_data = data.get("permissions", {})
        raw_tools = perms_data.get("tools", [])

        # Backward compat: flat list → all tools go to "user" tier
        if isinstance(raw_tools, list):
            tool_tiers = ToolTiers(user=raw_tools, write=[], admin=[])
        else:
            tool_tiers = ToolTiers(
                user=raw_tools.get("user", []),
                write=raw_tools.get("write", []),
                admin=raw_tools.get("admin", []),
            )

        permissions = PermissionSet(
            net=perms_data.get("net", []),
            fs=perms_data.get("fs", []),
            tools=tool_tiers,
        )
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            permissions=permissions,
        )

    def generate_scope_config(self) -> Dict[str, ScopeConfig]:
        """Derive FastMCP auth/visibility settings from the manifest tiers."""
        config = {}
        for tool_name in self.permissions.tools.user:
            config[tool_name] = ScopeConfig(auth="", tags=[])
        for tool_name in self.permissions.tools.write:
            config[tool_name] = ScopeConfig(auth="write", tags=[])
        for tool_name in self.permissions.tools.admin:
            config[tool_name] = ScopeConfig(auth="admin", tags=["admin"])
        return config


def load_manifest(path: str) -> SkillManifest:
    """Load and validate a claw_manifest.json file."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded manifest: {data.get('name', 'unknown')}")
    return SkillManifest.from_dict(data)


# Restrictive fallback: no network, workspace-only filesystem, no tools
DEFAULT_MANIFEST = SkillManifest(
    name="default-restricted",
    version="0.0.0",
    permissions=PermissionSet(net=[], fs=["./workspace"], tools=ToolTiers()),
)
