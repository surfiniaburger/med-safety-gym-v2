"""
FastMCP-based MCP Server for DIPG Safety Gym

This server exposes evaluation tools via the Model Context Protocol (MCP)
using FastMCP for simplified HTTP-based communication.

Based on Google's currency agent pattern from adk-samples.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any

from fastmcp import FastMCP

# Import our evaluation components
try:
    from .app import get_environment, EvaluationManager
except ImportError:
    from med_safety_gym.app import get_environment, EvaluationManager

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# Initialize FastMCP server
mcp = FastMCP("DIPG Evaluation MCP Server 🏥")


@mcp.tool()
def get_eval_tasks(
    max_samples: int = 100,
    shuffle: bool = True
) -> Dict[str, Any]:
    """Get evaluation tasks from the DIPG Safety Gym dataset.
    
    This tool retrieves medical safety evaluation tasks that can be used
    to test AI models for hallucination, reasoning quality, and safety.
    
    Args:
        max_samples: Maximum number of tasks to return (default: 100)
        shuffle: Whether to shuffle tasks before sampling (default: True)
        
    Returns:
        Dictionary containing:
        - tasks: List of task dictionaries with task_id, question, context
        - total_tasks: Number of tasks returned
        - dataset_size: Total size of the dataset
        
    Example:
        >>> result = get_eval_tasks(max_samples=10, shuffle=True)
        >>> tasks = result["tasks"]
        >>> for task in tasks:
        ...     print(f"Task {task['task_id']}: {task['question']}")
    """
    logger.info(f"🛠️ Tool: get_eval_tasks called (max_samples={max_samples}, shuffle={shuffle})")
    
    try:
        env = get_environment()
        tasks = env.get_eval_tasks(max_samples=max_samples, shuffle=shuffle)
        
        result = {
            "tasks": tasks,
            "total_tasks": len(tasks),
            "dataset_size": len(env.dataset)
        }
        
        logger.info(f"✅ Returning {len(tasks)} tasks")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_eval_tasks: {e}", exc_info=True)
        return {"error": f"Failed to get tasks: {str(e)}"}


@mcp.tool()
def evaluate_responses(
    evaluations: List[Dict[str, Any]],
    format: str = "json"
) -> Dict[str, Any]:
    """Evaluate model responses against the DIPG Safety Gym.
    
    This tool scores AI model responses for medical safety, checking for
    hallucinations, reasoning quality, and appropriate abstention.
    
    Args:
        evaluations: List of evaluation items, each containing:
            - response: The model's response text
            - ground_truth: Dict with context, question, expected_answer
        format: Response format - "json", "custom_tags", "xml", "yaml", or "auto"
        
    Returns:
        Dictionary containing:
        - mean_reward: Average reward score
        - safe_response_rate: Percentage of safe responses
        - hallucination_rate: Percentage of hallucinated responses
        - refusal_rate: Percentage of appropriate refusals
        - reasoning_consistency_rate: Reasoning quality score
        - And other detailed metrics
        
    Example:
        >>> evaluations = [{
        ...     "response": '{"analysis": "...", "proof": "...", "final": "..."}',
        ...     "ground_truth": {
        ...         "context": "Patient history...",
        ...         "question": "What is the diagnosis?",
        ...         "expected_answer": {"final": "...", "proof": "..."}
        ...     }
        ... }]
        >>> result = evaluate_responses(evaluations, format="json")
        >>> print(f"Mean reward: {result['mean_reward']}")
    """
    logger.info(f"🛠️ Tool: evaluate_responses called ({len(evaluations)} evaluations, format={format})")
    
    try:
        env = get_environment()
        eval_manager = EvaluationManager(env)
        
        # Evaluate using the stateless mode
        result = eval_manager.evaluate_with_ground_truth(
            evaluations=evaluations,
            response_format=format
        )
        
        # Convert Pydantic model to dict if needed
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        
        logger.info(f"✅ Evaluation complete. Mean reward: {result.get('mean_reward', 'N/A')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in evaluate_responses: {e}", exc_info=True)
        return {"error": f"Evaluation failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8081))
    logger.info(f"🚀 Starting DIPG Evaluation MCP Server on port {port}")
    logger.info("📚 Available tools: get_eval_tasks, evaluate_responses")
    
    # Run FastMCP server with HTTP transport
    # host="0.0.0.0" required for Cloud Run
    asyncio.run(
        mcp.run_async(
            transport="http",
            host="0.0.0.0",
            port=port,
        )
    )
