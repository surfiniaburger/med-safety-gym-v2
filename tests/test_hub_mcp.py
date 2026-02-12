import pytest
from httpx import AsyncClient, ASGITransport
from med_safety_eval.observability_hub import app, mcp

@pytest.mark.anyio
async def test_hub_rest_health():
    """Verify the legacy REST health check still works."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "observability_hub"

@pytest.mark.anyio
async def test_hub_mcp_tools():
    """Verify that the Hub now exposes evaluation tools via FastMCP."""
    # List tools via the mcp instance directly for unit testing
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "evaluate_medical_safety" in tool_names
    assert "sync_github_history" in tool_names
    assert "query_history" in tool_names

@pytest.mark.anyio
async def test_hub_rag_endpoint():
    """Verify the RAG query logic (backed by DataAgent)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/gauntlet/rag?query=hallucination")
    assert response.status_code == 200
    assert "context" in response.json()
