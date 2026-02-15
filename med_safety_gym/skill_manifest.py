"""
SafeClaw Skill Manifest — Data Model

Defines the schema for claw_manifest.json files that declare
a skill's required permissions (network, filesystem, tools).

Design: Pure data, no I/O side effects. Small, focused classes. (Farley)
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class PermissionSet:
    """Declares what a skill is allowed to access."""
    net: List[str] = field(default_factory=list)
    fs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)


@dataclass
class SkillManifest:
    """Top-level manifest for a SafeClaw skill package."""
    name: str
    version: str = "0.0.0"
    permissions: PermissionSet = field(default_factory=PermissionSet)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillManifest":
        """Create a manifest from a dictionary, using safe defaults."""
        perms_data = data.get("permissions", {})
        permissions = PermissionSet(
            net=perms_data.get("net", []),
            fs=perms_data.get("fs", []),
            tools=perms_data.get("tools", []),
        )
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            permissions=permissions,
        )


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
    permissions=PermissionSet(net=[], fs=["./workspace"], tools=[]),
)
