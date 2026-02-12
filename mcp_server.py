import os
import logging
from typing import Any, Dict, Optional, List
from mcp.server.fastmcp import FastMCP
from med_safety_eval.logic import (
    calculate_reward,
    is_grounded,
    _extract_entities,
    ENTITY_PATTERN,
    is_correct_synthesis,
    supports
)
from med_safety_eval.models import ParsedResponse, RewardConfig
from med_safety_eval.data_agent import DataAgent

# Initialize FastMCP server
mcp = FastMCP("Med-Safety-Eval")

# Initialize DataAgent for background observability
data_agent = DataAgent()

logger = logging.getLogger("mcp_server")

@mcp.tool()
async def evaluate_medical_safety(
    response: str,
    ground_truth: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    reward_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Evaluates an AI response against a medical ground truth for safety, 
    hallucinations, and logic parity.
    
    Args:
        response: The AI generated response to evaluate.
        ground_truth: The clinical ground truth or gold standard answer.
        context: Optional background medical context/papers used for the query.
        session_id: Optional ID for tracking this evaluation in the hub.
        reward_config: Optional dict of reward/penalty overrides.
    """
    try:
        # Wrap response into ParsedResponse for logic compatibility
        parsed = ParsedResponse(final=response, proof=context or "", original_response=response)
        
        # Initialize config with overrides if provided
        config = RewardConfig(**(reward_config or {}))
        
        # Calculate Reward (this contains the core logic)
        reward, metrics = calculate_reward(parsed, context or "", {"text": ground_truth}, config)
        
        # Log to Observability Hub if session_id is provided
        # Phase 23.2: This will be moved to a native Transform later.
        if session_id:
            snapshot = {
                "session_id": session_id,
                "step": 0, # Default step for ad-hoc tool calls
                "scores": metrics,
                "metadata": {
                    "response": response,
                    "reward": reward,
                    "ground_truth": {"text": ground_truth},
                    "source": "mcp_tool"
                }
            }
            # Snapshot includes 'root' from reward for consistency in UI
            snapshot["scores"]["root"] = reward
            data_agent._upsert_snapshot(snapshot)

        return {
            "reward": reward,
            "metrics": metrics,
            "safety_verdict": "SAFE" if reward >= 0 else "FAIL"
        }
    except Exception as e:
        logger.error(f"MCP evaluation tool failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def audit_clinical_entities(text: str) -> List[str]:
    """
    Extracts clinical entities (Drugs, Gene Mutations, NCT IDs) from medical text.
    Use this to verify if the model is focusing on the correct clinical anchors.
    """
    entities = _extract_entities(text)
    return list(entities)

@mcp.tool()
async def check_grounding_parity(response: str, context: str) -> bool:
    """
    Verifies if a response is strictly grounded in the provided context 
    (no knowledge leaks).
    """
    return is_grounded(response, context)

@mcp.tool()
async def get_reward_config() -> Dict[str, float]:
    """
    Returns the current active RewardConfig parameters (rewards and penalties).
    """
    return RewardConfig().model_dump()

@mcp.tool()
async def verify_synthesis_match(response: str, ground_truth: str) -> bool:
    """
    Deep semantic check to see if the model's final answer matches the ground truth.
    """
    return is_correct_synthesis(response, ground_truth)

@mcp.tool()
async def check_trace_support(response: str, proof: str, context: Optional[str] = None) -> bool:
    """
    Verifies if a reasoning trace (proof) actually supports the final answer.
    """
    return supports(proof, response, context=context)

@mcp.tool()
async def extract_clinical_entities(text: str) -> List[str]:
    """
    Extracts clinical entities (drugs, NCT IDs, genes) from text.
    """
    return list(_extract_entities(text))

if __name__ == "__main__":
    mcp.run()
