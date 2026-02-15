
import logging
from typing import Any, Callable, Optional, Union
from a2a.types import Message, TaskState
from a2a.utils import new_agent_text_message
from .mcp_client_adapter import MCPClientAdapter

logger = logging.getLogger(__name__)

BASE_MEDICAL_KNOWLEDGE = """
Known approved treatments for DIPG (Diffuse Intrinsic Pontine Glioma):
- Panobinostat (histone deacetylase inhibitor)
- ONC201 (dopamine receptor antagonist)
- Radiation therapy

Active clinical trials:
- NCT03416530 (ONC201 trial)
- NCT02717455 (Panobinostat trial)
"""

class SafeClawAgent:
    """
    A2A Agent that enforces strict safety invariants via MCP tools.
    """
    def __init__(self, client_factory: Optional[Callable[[], Any]] = None):
        # Default factory spawns the local server
        self.client_factory = client_factory or (
            lambda: MCPClientAdapter(
                command="uv", 
                # Need absolute path or ensure cwd is correct. 
                # We assume running from root or 'uv run' handles modules.
                args=["run", "python", "-m", "med_safety_gym.mcp_server"]
            )
        )

    async def run(self, message: Message, updater: Any, session: Any = None) -> None:
        """
        Main entry point for A2A loop.
        Extracts the user's message and performs safety-checked action.
        
        Args:
            message: A2A message from user
            updater: Task status updater
            session: Optional SessionMemory for conversation context
        """
        # Extract text from message
        if not message.parts:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("No message content received.")
            )
            return
        
        # Get the text content
        text_content = message.parts[0].root.text if message.parts else ""
        
        if not text_content:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("Empty message received.")
            )
            return
        
        # For now, use the message text as the "action"
        # In production, you'd use an LLM to extract intent/action
        action = text_content
        
        # Get conversation context (excluding current action to avoid self-validation)
        session_context = ""
        if session:
            session_context = session.get_medical_context(exclude_latest=True)
        
        # Merge with base knowledge for a robust safety context
        context = f"{BASE_MEDICAL_KNOWLEDGE}\n\nCONVERSATION CONTEXT:\n{session_context}"
        
        # Call the safety-checked action
        await self.context_aware_action(action, context, updater)

    async def context_aware_action(self, action: str, context: str, updater: Any) -> None:
        """
        Executes an action if it passes the Entity Parity check.
        """
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Verifying safety of action: {action}...")
        )
        
        # 1. Check Safety via MCP integration
        client = self.client_factory()
        
        is_safe = False
        reason = "Unknown error"
        
        try:
            # Handle both Context Manager (Adapter) and Mock (Function/Object)
            if hasattr(client, '__aenter__'):
                async with client as session:
                    result = await session.call_tool("check_entity_parity", {
                        "action": action, 
                        "context": context
                    })
            else:
                # Assuming valid client interface (mock)
                result = await client.call_tool("check_entity_parity", {
                    "action": action, 
                    "context": context
                })

            # Parse result
            if isinstance(result, dict):
                is_safe = result.get("is_safe", False)
                reason = result.get("reason", "No reason provided")
            elif isinstance(result, str):
                # Fallback if string returned
                import json
                try:
                    data = json.loads(result)
                    is_safe = data.get("is_safe", False)
                    reason = data.get("reason", "No reason provided")
                except:
                    reason = f"Invalid response format: {result}"
            
        except Exception as e:
            logger.error(f"MCP Check Failed: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message(f"❌ Safety Verification Error: {e}")
            )
            return

        # 2. Result Handling
        if not is_safe:
            # BLOCKED
            msg = f"🚫 Action BLOCKED by Guardian.\nReason: {reason}"
            # Use 'failed' to signal clearly.
            await updater.update_status(TaskState.failed, new_agent_text_message(msg))
            logger.warning(f"Safety Violation: {reason}")
            return

        # 3. Execution (Simulated for now)
        msg = f"✅ Safety Check Passed. Executing: {action}"
        await updater.update_status(TaskState.completed, new_agent_text_message(msg))
        logger.info(f"Action Executed: {action}")
