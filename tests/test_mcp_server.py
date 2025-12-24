"""
Tests for MCP server functionality.
"""
import pytest
import json
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def server_params():
    """Fixture for MCP server parameters."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "med_safety_gym.mcp_server"],
        env=os.environ.copy()
    )


@pytest.mark.anyio
async def test_mcp_server_list_tools(server_params):
    """Test that MCP server lists available tools."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            
            # Verify we have the expected tools
            tool_names = [tool.name for tool in tools.tools]
            assert "get_eval_tasks" in tool_names
            assert "evaluate_batch" in tool_names
            assert len(tool_names) == 2


@pytest.mark.anyio
async def test_mcp_get_eval_tasks(server_params):
    """Test get_eval_tasks tool."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Call get_eval_tasks with max_samples=5
            result = await session.call_tool("get_eval_tasks", arguments={
                "max_samples": 5,
                "shuffle": True
            })
            
            # Parse the result
            result_data = json.loads(result.content[0].text)
            
            # Verify structure
            assert "tasks" in result_data
            assert "total_tasks" in result_data
            assert "dataset_size" in result_data
            
            # Verify we got the requested number of tasks
            assert len(result_data["tasks"]) == 5
            assert result_data["total_tasks"] == 5
            
            # Verify each task has the expected fields
            for task in result_data["tasks"]:
                assert "task_id" in task
                assert "context" in task
                assert "question" in task
                assert "expected_answer" in task


@pytest.mark.anyio
async def test_mcp_evaluate_batch(server_params):
    """Test evaluate_batch tool."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Create a mock evaluation
            evaluations = [{
                "response": '{"analysis": "test analysis", "proof": "test proof", "final": "test answer"}',
                "ground_truth": {
                    "context": "Test medical context about DIPG.",
                    "question": "What is the test question?",
                    "expected_answer": {"final": "test answer", "proof": "expected proof"}
                }
            }]
            
            # Call evaluate_batch
            result = await session.call_tool("evaluate_batch", arguments={
                "evaluations": evaluations,
                "format": "json"
            })
            
            # Parse the result
            result_data = json.loads(result.content[0].text)
            
            # Verify evaluation metrics structure
            assert "total_responses" in result_data
            assert "mean_reward" in result_data
            assert "median_reward" in result_data
            assert "std_reward" in result_data
            assert "rewards" in result_data
            
            # Verify advanced safety metrics
            assert "refusal_rate" in result_data
            assert "safe_response_rate" in result_data
            assert "medical_hallucination_rate" in result_data
            assert "reasoning_consistency_rate" in result_data
            
            # Verify we evaluated the correct number of items
            assert result_data["total_responses"] == 1
            assert len(result_data["rewards"]) == 1


@pytest.mark.anyio
async def test_mcp_evaluate_batch_multiple(server_params):
    """Test evaluate_batch with multiple evaluations."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Create multiple mock evaluations
            evaluations = [
                {
                    "response": '{"analysis": "test1", "proof": "proof1", "final": "answer1"}',
                    "ground_truth": {
                        "context": "Context 1",
                        "question": "Question 1?",
                        "expected_answer": {"final": "answer1"}
                    }
                },
                {
                    "response": '{"analysis": "test2", "proof": "proof2", "final": "answer2"}',
                    "ground_truth": {
                        "context": "Context 2",
                        "question": "Question 2?",
                        "expected_answer": {"final": "answer2"}
                    }
                }
            ]
            
            result = await session.call_tool("evaluate_batch", arguments={
                "evaluations": evaluations,
                "format": "json"
            })
            
            result_data = json.loads(result.content[0].text)
            
            # Verify we evaluated all items
            assert result_data["total_responses"] == 2
            assert len(result_data["rewards"]) == 2
