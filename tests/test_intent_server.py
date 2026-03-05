
import pytest
import asyncio
from med_safety_gym.intent_server import mcp
from mcp.server.fastmcp import Context

@pytest.mark.asyncio
async def test_classify_intent_new_topic():
    from med_safety_gym.intent_server import classify_intent
    # Mocking Context since we're testing the function directly
    ctx = Context()
    result = await classify_intent("What is DIPG?", ctx)
    assert result["category"] == "NEW_TOPIC"
    assert result["is_correction"] is False

@pytest.mark.asyncio
async def test_classify_intent_correction():
    from med_safety_gym.intent_server import classify_intent
    ctx = Context()
    result = await classify_intent("No, I meant the other trial.", ctx)
    assert result["is_correction"] is True
    assert result["category"] == "REFINEMENT"

@pytest.mark.asyncio
async def test_classify_intent_follow_up():
    from med_safety_gym.intent_server import classify_intent
    ctx = Context()
    result = await classify_intent("How much does ONC201 cost?", ctx)
    assert result["category"] == "FOLLOW_UP"
