"""
Integration test for Admin Escalation flow.
Verifies that:
1. 'gh: unlock admin tools' correctly escalates permissions.
2. 'gh: delete comment' is blocked before escalation.
3. 'gh: delete comment' is allowed after escalation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.manifest_interceptor import ManifestInterceptor
from med_safety_gym.skill_manifest import SkillManifest, PermissionSet, ToolTiers
from med_safety_gym.session_memory import SessionMemory

@pytest.mark.asyncio
async def test_admin_unlock_is_rejected_zero_trust():
    """Verify 'gh: unlock admin tools' is rejected with Zero-Trust message."""
    # Setup mocks
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client_factory = MagicMock(return_value=mock_client)
    
    # Initialize agent
    agent = SafeClawAgent(github_client_factory=mock_client_factory)
    agent.auth_token = "valid-token"
    
    # Create session
    session = SessionMemory("test_user")

    from unittest.mock import patch
    with patch("med_safety_gym.claw_agent.verify_delegation_token"):
        await agent.github_action("gh: unlock admin tools", mock_updater, session=session)
    
    # Verify Session NOT escalated
    assert len(session.escalated_tools) == 0
    
    # Verify status update with Zero-Trust message
    mock_updater.update_status.assert_called()
    msg = mock_updater.update_status.call_args_list[-1][0][1]
    assert "Zero-Trust Policy" in str(msg)
    assert "session unlock is disabled" in str(msg).lower()


@pytest.mark.asyncio
async def test_delete_comment_routing_and_blocking():
    """Verify 'delete comment' is routed, blocked, then allowed."""
    # Setup mocks
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="Deleted comment")
    mock_client_factory = MagicMock(return_value=mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    agent = SafeClawAgent(github_client_factory=mock_client_factory)
    agent.auth_token = "valid-token"
    
    # Force a known manifest state with admin tools
    agent.interceptor = ManifestInterceptor(SkillManifest(
        name="test-manifest",
        permissions=PermissionSet(
            tools=ToolTiers(
                user=["list_issues"], 
                admin=["delete_issue_comment", "unlock_admin_tools"]
            )
        )
    ))

    # Create session
    session = SessionMemory("test_user")

    # 1. Attempt delete without escalation -> INTERVENTION REQUIRED (JIT)
    from unittest.mock import patch
    with patch("med_safety_gym.claw_agent.verify_delegation_token"):
        await agent.github_action("gh: delete comment 123 on issue 1", mock_updater, session=session)
    
    # Check it triggered intervention
    args, _ = mock_updater.update_status.call_args
    assert "INTERVENTION REQUIRED" in str(args[1])
    # The message includes the tool name
    assert "delete_issue_comment" in str(args[1])
    
    # Check server was NOT called
    mock_client.call_tool.assert_not_called()
    
    # 2. Escalate (manually for test speed)
    session.escalate_tool("delete_issue_comment")
    assert session.is_tool_escalated("delete_issue_comment")
    
    # 3. Attempt delete again -> ALLOWED
    with patch("med_safety_gym.claw_agent.verify_delegation_token"):
        await agent.github_action("gh: delete comment 123 on issue 1", mock_updater, session=session)
    
    # Check server called
    mock_client.call_tool.assert_called_with(
        "delete_issue_comment", 
        {"issue_number": 1, "comment_id": 123}
    )
    # Check success message
    msg = mock_updater.update_status.call_args_list[-1][0][1]
    assert "Deleted comment" in str(msg)
