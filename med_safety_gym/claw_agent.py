
import logging
import re
import json
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
    def __init__(
        self, 
        client_factory: Optional[Callable[[], Any]] = None,
        github_client_factory: Optional[Callable[[], Any]] = None
    ):
        # Default safety factory
        self.client_factory = client_factory or (
            lambda: MCPClientAdapter(
                command="uv", 
                args=["run", "python", "-m", "med_safety_gym.mcp_server"]
            )
        )
        # GitHub factory
        self.github_client_factory = github_client_factory or (
            lambda: MCPClientAdapter(
                command="uv", 
                args=["run", "python", "-m", "med_safety_gym.github_tools"]
            )
        )
        self._github_session = None

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
        action = text_content
        
        # ROUTING LOGIC:
        # We classify if this is a GitHub operation or a Clinical action.
        # GitHub keywords: 'repo', 'issues', 'github', 'gh:', 'pr ', 'branch', 'pull', 'merge'
        github_keywords = {
            'gh:', 'github', 'repo', 'say ', 'voice ', '!v '
        }
        # Harden GitHub detection with word boundaries for sensitive keywords
        # Only auto-route very specific GitHub-related tokens or phrases.
        # Generic words like 'pull', 'merge', 'issue', 'branch' now require 'gh:' prefix.
        is_github = (any(k in action.lower() for k in github_keywords) or 
                     re.search(r'\b(pr|pull request|pull requests|list issues|create issue)\b', action.lower()))

        if is_github:
            logger.info(f"Routing to GitHub Tools: {action}")
            await self.github_action(action, updater, session=session)
            return

        # Default: Clinical Action (Strictest Guardian Path)
        logger.info(f"Routing to Medical Guardian: {action}")
        
        # Get conversation context (excluding current action to avoid self-validation)
        session_context = ""
        if session:
            session_context = session.get_medical_context(exclude_latest=True)
        
        # Merge with base knowledge for a robust safety context
        context = f"{BASE_MEDICAL_KNOWLEDGE}\n\nCONVERSATION CONTEXT:\n{session_context}"
        
        await self.context_aware_action(action, context, updater)

    async def github_action(self, action: str, updater: Any, session: Any = None) -> None:
        """
        Interacts with the GitHub MCP server to perform repo operations.
        Example intents: 
        - 'gh: list issues'
        - 'gh: set repo surfiniaburger/med-safety-gym-v2'
        - 'gh: create issue title="Bug" body="Found error"'
        """
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"🔄 Processing GitHub operation: {action}...")
        )
        
        # Per PR feedback: Manage client with async with to prevent resource leaks
        async with self.github_client_factory() as session_client:
            # RESTORE STATE: If the session memory has a repo, but the tool client doesn't know it, configure it.
            # This solves the "bot restart" state loss problem.
            if session and session.github_repo:
                 await session_client.call_tool("configure_repo", {"repo_name": session.github_repo})
            
            try:
                # Clean command
                cmd = action.lower().replace("gh:", "").strip()
                
                # 1. Set Repo (Improved with Regex as per code review)
                set_repo_match = re.match(r'(set|configure|use)\s+repo\s+(?:to\s+)?(.+)', cmd)
                if set_repo_match:
                    repo_name = set_repo_match.group(2).strip()
                    result = await session_client.call_tool("configure_repo", {"repo_name": repo_name})
                    # Sync back to session memory for persistence
                    if session:
                        session.github_repo = repo_name
                
                # 2. Issues (Improved with Regex for better extraction)
                elif re.search(r'\bissues?\b', cmd):
                    if re.search(r'\b(create|new|add)\b', cmd):
                        title_match = re.search(r'title="([^"]+)"', action)
                        body_match = re.search(r'body="([^"]+)"', action)
                        title = title_match.group(1) if title_match else "Untitled"
                        body = body_match.group(1) if body_match else ""
                        result = await session_client.call_tool("create_issue", {"title": title, "body": body})
                    else:
                        result = await session_client.call_tool("list_issues", {})
                
                # 3. Pull Requests (Improved with Regex)
                elif re.search(r'\b(pulls?|prs?|requests)\b', cmd):
                    result = await session_client.call_tool("list_pull_requests", {})
                
                # 4. Check/Info (Default list issues or meta info)
                elif re.search(r'\b(check|info)\b', cmd):
                    result = await session_client.call_tool("list_issues", {})
                
                # 5. Voice/Speech (Handled primarily by the Telegram Mirror)
                elif re.search(r'\b(say|voice|!v)\b', cmd):
                    # Strip the keyword and return text for the bridge to speak
                    for kw in ["say", "voice", "!v"]:
                        cmd = re.sub(rf'\b{re.escape(kw)}\b', "", cmd, flags=re.IGNORECASE).strip()
                    result = cmd if cmd else "Voice mode activated. What would you like me to say?"
                    
                else:
                    result = f"Command recognized as GitHub, but specific action unknown: '{cmd}'. Try 'list issues' or 'set repo'."

                await updater.update_status(TaskState.completed, new_agent_text_message(str(result)))

            except Exception as e:
                logger.error(f"GitHub Action Failed: {e}")
                await updater.update_status(
                    TaskState.failed, 
                    new_agent_text_message(f"❌ GitHub Error: {e}")
                )

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
                try:
                    data = json.loads(result)
                    is_safe = data.get("is_safe", False)
                    reason = data.get("reason", "No reason provided")
                except json.JSONDecodeError:
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
