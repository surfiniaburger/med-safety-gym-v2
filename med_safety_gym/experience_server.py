
"""
SafeClaw Experience MCP Server (Refiner)
Handles distillation of pragmatic guidelines and contrastive pair logging.
Following: SafeClaw Practices (Farley & Beck)
"""

import logging
from typing import Optional
from fastmcp import FastMCP, Context
from med_safety_gym.experience_refiner import ExperienceRefiner
from med_safety_gym.session_memory import SessionStore

# Initialize FastMCP Server
mcp = FastMCP("SafeClaw Refiner")
logger = logging.getLogger(__name__)

# Isolated components
_refiner = ExperienceRefiner()
_store = SessionStore()

_TRACE_ALLOWED_KEYS = frozenset({
    "turn_id",
    "intent",
    "is_success",
    "failure_reason",
    "detected_entities",
    "context_entities",
    "error",
})

_TRACE_FORBIDDEN_KEYS = frozenset({
    "raw_text",
    "raw_user_text",
    "user_prompt",
    "prompt",
    "input_text",
    "message",
    "content",
})


def _validate_semantic_trace(semantic_trace: dict) -> None:
    """Enforce zero-injection boundary: trace must be structured metadata only."""
    if not isinstance(semantic_trace, dict):
        raise ValueError("semantic_trace must be a dict")

    unknown_keys = set(semantic_trace) - _TRACE_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError(f"semantic_trace contains unsupported keys: {sorted(unknown_keys)}")

    forbidden = set(semantic_trace).intersection(_TRACE_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"semantic_trace contains forbidden raw-text keys: {sorted(forbidden)}")

@mcp.tool()
async def log_contrastive_pair(session_id: str, is_success: bool, semantic_trace: dict, trajectory: list[dict[str, str]], ctx: Context) -> str:
    """
    Logs a contrastive pair (success or failure) to the persistent store.
    """
    logger.info(f"Logging contrastive pair for session {session_id} (Success: {is_success})")
    _validate_semantic_trace(semantic_trace)
    
    # Reconstruct a dummy session object for the store
    class DummySession:
        def __init__(self, sid, messages):
            self.user_id = sid
            self._messages = messages
            self.turn_count = semantic_trace.get("turn_id", 0)
        
        def get_messages(self, limit=None):
            return self._messages

    session = DummySession(session_id, trajectory)
    _store.log_contrastive_pair(session, is_success, semantic_trace)
    
    return f"Contrastive pair logged for session: {session_id}"

@mcp.tool()
async def distill_guidelines(ctx: Context, limit: int = 10) -> str:
    """
    Distills pragmatic guidelines from recent interaction traces.
    """
    logger.info(f"Distilling guidelines from last {limit} traces...")
    guidelines = await _refiner.distill_guidelines(limit=limit)
    return guidelines

if __name__ == "__main__":
    mcp.run()
