
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from mcp.server.fastmcp import Context

@pytest.mark.asyncio
async def test_log_contrastive_pair():
    # We need to mock SessionStore to avoid actual DB writes in this test
    from med_safety_gym.experience_server import log_contrastive_pair
    
    ctx = Context()
    # Mocking the internal store
    import med_safety_gym.experience_server as es
    es._store = MagicMock()
    
    semantic_trace = {"turn_id": 1, "intent": "NEW_TOPIC", "is_success": True}
    result = await log_contrastive_pair("test_session", True, semantic_trace, ctx)
    
    assert "logged" in result
    es._store.log_contrastive_pair.assert_called_once()

@pytest.mark.asyncio
async def test_distill_guidelines():
    from med_safety_gym.experience_server import distill_guidelines
    import med_safety_gym.experience_server as es
    es._refiner = AsyncMock()
    es._refiner.distill_guidelines.return_value = "When the user asks X, do Y."
    
    ctx = Context()
    result = await distill_guidelines(ctx=ctx, limit=5)
    assert "When the user" in result
    es._refiner.distill_guidelines.assert_called_once_with(limit=5)
