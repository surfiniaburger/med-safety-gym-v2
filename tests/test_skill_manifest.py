"""
Tests for SkillManifest data model.
TDD Test List (Beck Canon):
1. Load valid manifest from dict
2. Load manifest with missing fields → uses defaults
3. Load manifest from JSON file
4. Default manifest blocks all network access
5. Default manifest restricts filesystem to ./workspace
"""
import json
import os
import tempfile
import pytest
from med_safety_gym.skill_manifest import (
    SkillManifest,
    PermissionSet,
    load_manifest,
    DEFAULT_MANIFEST,
)


def test_load_valid_manifest_from_dict():
    """A well-formed dict produces a valid SkillManifest."""
    data = {
        "name": "weather-skill",
        "version": "1.0.0",
        "permissions": {
            "net": ["api.weather.gov"],
            "fs": ["./workspace/weather-cache.json"],
            "tools": {
                "user": ["get_weather"],
                "admin": ["reset_weather_satellite"]
            },
        },
    }
    manifest = SkillManifest.from_dict(data)
    assert manifest.name == "weather-skill"
    assert manifest.version == "1.0.0"
    assert "api.weather.gov" in manifest.permissions.net
    assert "get_weather" in manifest.permissions.tools.user
    assert "reset_weather_satellite" in manifest.permissions.tools.admin


def test_load_manifest_with_missing_fields_uses_defaults():
    """Missing fields fall back to safe defaults."""
    data = {"name": "bare-skill"}
    manifest = SkillManifest.from_dict(data)
    assert manifest.version == "0.0.0"
    assert manifest.permissions.net == []
    assert manifest.permissions.fs == []
    assert manifest.permissions.tools.all_tools == []


def test_load_manifest_from_json_file():
    """load_manifest reads and parses a JSON file."""
    data = {
        "name": "file-skill",
        "version": "2.0.0",
        "permissions": {"net": ["example.com"], "fs": [], "tools": ["read_file"]},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name

    try:
        manifest = load_manifest(tmp_path)
        assert manifest.name == "file-skill"
        assert manifest.version == "2.0.0"
        # Flat list should be mapped to 'user' tier
        assert "read_file" in manifest.permissions.tools.user
    finally:
        os.remove(tmp_path)


def test_default_manifest_blocks_all_network():
    """The built-in DEFAULT_MANIFEST allows no network access."""
    assert DEFAULT_MANIFEST.permissions.net == []


def test_default_manifest_restricts_fs_to_workspace():
    """The built-in DEFAULT_MANIFEST only allows ./workspace."""
    assert DEFAULT_MANIFEST.permissions.fs == ["./workspace"]


def test_generate_scope_config():
    """Verify scope generation from tiers."""
    data = {
        "name": "tiered-skill",
        "permissions": {
            "tools": {
                "user": ["public_tool"],
                "write": ["writer_tool"],
                "admin": ["admin_tool"]
            }
        }
    }
    manifest = SkillManifest.from_dict(data)
    config = manifest.generate_scope_config()
    
    assert config["public_tool"].auth == ""
    assert config["public_tool"].tags == []
    
    assert config["writer_tool"].auth == "write"
    
    assert config["admin_tool"].auth == "admin"
    assert "admin" in config["admin_tool"].tags
