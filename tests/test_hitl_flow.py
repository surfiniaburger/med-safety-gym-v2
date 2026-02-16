import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.manifest_interceptor import ManifestInterceptor
from med_safety_gym.skill_manifest import SkillManifest, PermissionSet, ToolTiers
from med_safety_gym.session_memory import SessionMemory
from a2a.types import TaskState

@pytest.mark.asyncio
async def test_critical_tool_requires_local_auth():
    """Verify that a critical tool triggers the local auth guard."""
    # Setup mocks
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="Action successful")
    mock_client_factory = MagicMock(return_value=mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    # Initialize agent with custom manifest
    agent = SafeClawAgent(github_client_factory=mock_client_factory)
    agent.interceptor = ManifestInterceptor(SkillManifest(
        name="test-manifest",
        permissions=PermissionSet(
            tools=ToolTiers(
                critical=["dangerous_action"]
            )
        )
    ))

    session = SessionMemory("test_user")
    # Do NOT pre-escalate; we want to see it trigger the guard

    # Mock the auth guard and vision audit
    with patch("med_safety_gym.claw_agent.require_local_auth", return_value=True) as mock_auth, \
         patch("med_safety_gym.claw_agent.get_audit_summary", return_value="Summary") as mock_audit:
        
        await agent._call_tool_with_interception(
            "dangerous_action", 
            {}, 
            mock_client, 
            mock_updater, 
            session=session
        )
        
        # Verify local auth was called
        mock_auth.assert_called_once()
        # Verify it entered input_required state (for Telegram confirmation)
        mock_updater.update_status.assert_any_call(
            TaskState.input_required,
            ANY,
            metadata=ANY
        )

@pytest.mark.asyncio
async def test_critical_tool_disallowed_if_auth_fails():
    """Verify that if local auth fails, the tool is not called."""
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client_factory = MagicMock(return_value=mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    agent = SafeClawAgent(github_client_factory=mock_client_factory)
    agent.interceptor = ManifestInterceptor(SkillManifest(
        name="test-manifest",
        permissions=PermissionSet(tools=ToolTiers(critical=["dangerous_action"]))
    ))

    session = SessionMemory("test_user")
    # Do NOT pre-escalate

    # Mock the auth guard to return False (Denied)
    with patch("med_safety_gym.claw_agent.require_local_auth", return_value=False) as mock_auth:
        result = await agent._call_tool_with_interception(
            "dangerous_action", 
            {}, 
            mock_client, 
            mock_updater, 
            session=session
        )
        
        # Verify tool was NOT called
        mock_client.call_tool.assert_not_called()
        # Verify it returned None (Blocked)
        assert result is None
        # Verify status update for denial
        mock_updater.update_status.assert_called()
        args, _ = mock_updater.update_status.call_args
        assert "System authentication failed" in str(args[1])
