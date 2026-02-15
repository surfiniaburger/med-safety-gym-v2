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

@pytest.mark.asyncio
async def test_admin_escalation_command():
    """Verify 'gh: unlock admin tools' triggers escalation."""
    # Setup mocks
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="Admin tools unlocked")
    mock_client_factory = MagicMock(return_value=mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    # Initialize agent
    agent = SafeClawAgent(github_client_factory=mock_client_factory)
    
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

    # Execute command
    await agent.github_action("gh: unlock admin tools", mock_updater)
    
    # Verify Interceptor was escalated
    # escalate_all_admin() adds all admin tools to _escalated_tools
    # We check if *any* admin tool is now in escalated set
    assert len(agent.interceptor._escalated_tools) > 0
    
    # Verify Server tool was called
    mock_client.call_tool.assert_called_with("unlock_admin_tools", {})
    
    # Verify status update
    mock_updater.update_status.assert_called()
    # Check string representation as message structure can vary (parts vs content)
    msg = mock_updater.update_status.call_args_list[-1][0][1]
    assert "Admin Escalation" in str(msg)


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

    # 1. Attempt delete without escalation -> BLOCKED
    await agent.github_action("gh: delete comment 123 on issue 1", mock_updater)
    
    # Check it was blocked
    args, _ = mock_updater.update_status.call_args
    # Check string representation for robustness
    assert "BLOCKED" in str(args[1])
    # The reason should mention admin escalation
    assert "admin escalation" in str(args[1])
    
    # Check server was NOT called
    mock_client.call_tool.assert_not_called()
    
    # 2. Escalate (manually for test speed)
    agent.interceptor.escalate("delete_issue_comment")
    
    # 3. Attempt delete again -> ALLOWED
    await agent.github_action("gh: delete comment 123 on issue 1", mock_updater)
    
    # Check server called
    mock_client.call_tool.assert_called_with(
        "delete_issue_comment", 
        {"issue_number": 1, "comment_id": 123}
    )
    # Check success message
    msg = mock_updater.update_status.call_args_list[-1][0][1]
    assert "Deleted comment" in str(msg)
