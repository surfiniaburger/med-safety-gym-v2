import pytest
import time
from med_safety_gym.session_memory import SessionMemory

@pytest.mark.asyncio
async def test_jit_escalation_lifecycle():
    """Verify tool escalation logic: JIT and TTL expiration."""
    session = SessionMemory("test_user")
    
    # 1. Initially not escalated
    assert not session.is_tool_escalated("delete_repo")
    
    # 2. Escalate with short TTL
    session.escalate_tool("delete_repo", ttl=1) # 1 second
    assert session.is_tool_escalated("delete_repo")
    assert not session.is_tool_escalated("other_tool") # Least Privilege: isolation
    
    # 3. Wait for expiration
    time.sleep(1.1)
    assert not session.is_tool_escalated("delete_repo")

@pytest.mark.asyncio
async def test_escalation_persistence_with_expiration():
    """Verify that expired tools are cleared or ignored."""
    session = SessionMemory("test_user")
    
    # Escalate two tools with different TTLs
    session.escalate_tool("tool_a", ttl=10)
    session.escalate_tool("tool_b", ttl=-1) # Already expired
    
    assert session.is_tool_escalated("tool_a")
    assert not session.is_tool_escalated("tool_b")
