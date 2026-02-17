
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from med_safety_gym.claw_agent import SafeClawAgent
from a2a.types import TaskState

@pytest.mark.asyncio
async def test_github_routing_edge_cases():
    """Test various natural language commands and how they route."""
    mock_updater = MagicMock()
    mock_updater.update_status = AsyncMock()
    
    # Mock the github client
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.call_tool = AsyncMock(return_value="Success")
    
    mock_factory = MagicMock(return_value=mock_client)
    
    agent = SafeClawAgent(github_client_factory=mock_factory)
    
    # Mock the interceptor to be permissive but tier-aware
    agent.interceptor = MagicMock()
    def mock_intercept(tool_name, tool_args, audit_log=None):
        check = MagicMock()
        check.allowed = True
        if "delete" in tool_name: check.tier = "critical"
        elif "unlock" in tool_name: check.tier = "admin"
        else: check.tier = "user"
        return check
    
    agent.interceptor.intercept.side_effect = mock_intercept
    agent._ensure_governor_interceptor = AsyncMock()
    
    test_cases = [
        ("gh: set repo surfiniaburger/softclaw", "configure_repo"),
        ("gh: list issues", "list_issues"),
        ("gh: create issue title=\"Test\" body=\"Msg\"", "create_issue"),
        ("gh: what are the prs", "list_pull_requests"),
        ("gh: unlock admin tools", "unlock_admin_tools"),
        ("unlock admin tools", "unlock_admin_tools"),
        ("gh: delete repo", "delete_repo"),
        ("delete_repo", "delete_repo"),
    ]
    
    for cmd, expected_tool in test_cases:
        mock_client.call_tool.reset_mock()
        mock_updater.update_status.reset_mock()
        
        await agent.github_action(cmd, mock_updater)
        
        # Check calls
        calls = [call.args[0] for call in mock_client.call_tool.call_args_list]
        status_states = [call.args[0] for call in mock_updater.update_status.call_args_list]
        messages = [str(call.args[1]) for call in mock_updater.update_status.call_args_list]
        
        if "unlock" in cmd:
            # Should be rejected with advice, in working state
            assert TaskState.working in status_states
            # Check messages (the second argument to update_status)
            assert any("Zero-Trust" in msg for msg in messages)
            assert expected_tool not in calls
        elif "delete" in cmd:
            # Should trigger JIT intervention
            assert TaskState.input_required in status_states
            assert any("INTERVENTION REQUIRED" in msg for msg in messages)
            assert expected_tool not in calls
        else:
            # Standard tools should proceed
            assert expected_tool in calls
