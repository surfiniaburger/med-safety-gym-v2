
import logging
import os
import re
import json
import asyncio
from typing import Any, Callable, Optional, Union
from a2a.types import Message, TaskState
from a2a.utils import new_agent_text_message
from .mcp_client_adapter import MCPClientAdapter
from .skill_manifest import load_manifest, DEFAULT_MANIFEST
from .manifest_interceptor import ManifestInterceptor
from .auth_guard import require_local_auth
from .vision_audit import get_audit_summary

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

        # Load manifest for permission enforcement
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "claw_manifest.json"
        )
        if os.path.exists(manifest_path):
            self.interceptor = ManifestInterceptor(load_manifest(manifest_path))
        else:
            self.interceptor = ManifestInterceptor(DEFAULT_MANIFEST)

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
            'gh:', 'github', 'repo', 'say ', 'voice ', '!v ', 'unlock', 'admin'
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
            session_context = await session.get_medical_context(exclude_latest=True)
        
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
                
                # 0. JIT Escalation (Least Privilege)
                # Instead of a global "unlock", we now recommend using the tool directly.
                # If they ask to "unlock", we'll tell them to just ask for what they need.
                if "unlock" in cmd and "admin" in cmd:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message("🔒 Zero-Trust Policy: Global session unlock is disabled. Please request the specific high-stakes action you need (e.g., 'gh: delete repo'), and I will request a Just-In-Time (JIT) confirmation.")
                    )
                    return

                # 1. Set Repo (Improved with Regex as per code review)
                set_repo_match = re.match(r'(set|configure|use)\s+repo\s+(?:to\s+)?(.+)', cmd)
                if set_repo_match:
                    repo_name = set_repo_match.group(2).strip()
                    # Use helper for interception + execution
                    result = await self._call_tool_with_interception(
                        "configure_repo", 
                        {"repo_name": repo_name}, 
                        session_client, 
                        updater, 
                        session
                    )
                    if result is None: return # Blocked

                    # Sync back to session memory for persistence
                    if session:
                        session.github_repo = repo_name
                
                # 2. Delete Repository (Critical Tool)
                elif re.search(r'\b(delete|remove)\s+(repo|repository)\b', cmd) or cmd == "delete_repo":
                    result = await self._call_tool_with_interception(
                        "delete_repo", 
                        {}, 
                        session_client, 
                        updater, 
                        session
                    )
                    if result is None: return # Blocked
                
                # 3. Delete Comment Action (Admin Tool) - Must be before generic 'issues' check
                elif "delete" in cmd and "comment" in cmd:
                    # Regex: delete comment 123 on issue 456
                    match = re.search(r'delete\s+comment\s+(\d+)\s+(?:on|from)\s+issue\s+(\d+)', cmd)
                    if match:
                        comment_id = int(match.group(1))
                        issue_num = int(match.group(2))
                        
                        result = await self._call_tool_with_interception(
                            "delete_issue_comment", 
                            {"issue_number": issue_num, "comment_id": comment_id},
                            session_client, 
                            updater, 
                            session
                        )
                        if result is None: return # Blocked
                    else:
                        await updater.update_status(TaskState.failed, new_agent_text_message("⚠️ Usage: gh: delete comment <id> on issue <num>"))
                        return

                # 3. Issues (Improved with Regex for better extraction)
                elif re.search(r'\bissues?\b', cmd):
                    tool_name = "create_issue" if re.search(r'\b(create|new|add)\b', cmd) else "list_issues"
                    
                    if tool_name == "create_issue":
                        title_match = re.search(r'title="([^"]+)"', action)
                        body_match = re.search(r'body="([^"]+)"', action)
                        title = title_match.group(1) if title_match else "Untitled"
                        body = body_match.group(1) if body_match else ""
                        tool_args = {"title": title, "body": body}
                    else:
                        tool_args = {}

                    result = await self._call_tool_with_interception(
                        tool_name, 
                        tool_args, 
                        session_client, 
                        updater, 
                        session
                    )
                    if result is None: return # Blocked
                
                # 3. Pull Requests (Improved with Regex)
                elif re.search(r'\b(pulls?|prs?|requests)\b', cmd):
                    result = await self._call_tool_with_interception(
                        "list_pull_requests", 
                        {}, 
                        session_client, 
                        updater, 
                        session
                    )
                    if result is None: return # Blocked
                
                # 5. Check/Info (Default list issues or meta info)
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

    async def _call_tool_with_interception(
        self,
        tool_name: str,
        tool_args: dict,
        session_client: Any,
        updater: Any,
        session: Any = None
    ) -> Optional[Any]:
        """Orchestrate tool execution with manifest checks and security guards."""
        check = self._check_manifest_interception(tool_name, tool_args, updater, session)
        if not check.allowed:
            return None

        if not await self._apply_security_guards(check, tool_name, tool_args, updater, session=session):
            return None

        return await session_client.call_tool(tool_name, tool_args)

    async def execute_confirmed_tool(self, tool_name: str, tool_args: dict, updater: Any, session: Any = None) -> None:
        """Bypass guards and execute a tool that has been manually approved by the user."""
        
        # JIT Escalation: Escalate only the approved tool with a short-lived TTL (5 mins)
        if session:
            session.escalate_tool(tool_name)
            
            # Persist to DB
            from .session_memory import SessionStore
            store = SessionStore()
            store.save(session)
        
        async with self.github_client_factory() as session_client:
            # Restore repo from session
            if session and session.github_repo:
                 await session_client.call_tool("configure_repo", {"repo_name": session.github_repo})
            
            try:
                result = await session_client.call_tool(tool_name, tool_args)
                await updater.update_status(TaskState.completed, new_agent_text_message(str(result)))
            except Exception as e:
                await updater.update_status(TaskState.failed, new_agent_text_message(f"❌ Execution error: {e}"))

    def _check_manifest_interception(self, tool_name: str, tool_args: dict, updater: Any, session: Any) -> Any:
        """Verify the tool call against the manifest policy."""
        audit_log = getattr(session, "audit_log", []) if session else []
        check = self.interceptor.intercept(tool_name, tool_args, audit_log)
        
        # We don't block here if the updater handles the message, but for consistency:
        if not check.allowed:
            asyncio.create_task(updater.update_status(
                TaskState.failed, 
                new_agent_text_message(f"🚨 BLOCKED: {check.reason}")
            ))
        return check

    async def _apply_security_guards(self, check: Any, tool_name: str, tool_args: dict, updater: Any, session: Any = None) -> bool:
        """Enforce additional guards for critical or sensitive actions."""
        # JIT Bypass: If the tool is already escalated in the session (time-bound), skip HITL.
        if session and session.is_tool_escalated(tool_name):
            return True

        # 1. Vision Audit (Generate summary for the user)
        audit_summary = await get_audit_summary(tool_name, tool_args)
        
        # 2. Critical Tier: Local Biometrics/Auth FIRST
        if check.tier == "critical":
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"🔐 CRITICAL ACTION: {tool_name}\n{audit_summary}\nPlease verify on your Mac...")
            )
            if not require_local_auth(f"Authorize tool: {tool_name}"):
                await updater.update_status(TaskState.failed, new_agent_text_message("🚫 System authentication failed or canceled."))
                return False

        # 3. Sensitive Tiers (Admin, Critical): Require Telegram Approval
        if check.tier in ("admin", "critical"):
            # Signal that we need input/confirmation
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(f"🚨 INTERVENTION REQUIRED\n\n{audit_summary}\n\nDo you want to proceed?"),
                metadata={"tool_name": tool_name, "tool_args": tool_args}
            )
            # We return False here to STOP the current execution flow.
            # The bridge will handle the resume by calling a different method or re-running with a flag.
            return False

        return True

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
