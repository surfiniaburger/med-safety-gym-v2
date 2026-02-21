
import logging
import atexit
import os
import re
from mcp.server.fastmcp import FastMCP
import json
import asyncio
import httpx
from typing import Any, Callable, Optional, Union
from a2a.types import Message, TaskState
from a2a.utils import new_agent_text_message
from .mcp_client_adapter import MCPClientAdapter
from .skill_manifest import load_manifest, DEFAULT_MANIFEST, SkillManifest
from .manifest_interceptor import ManifestInterceptor
from .auth_guard import require_local_auth
from .vision_audit import get_audit_summary
from .identity.scoped_identity import verify_delegation_token
import jwt

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
        self._github_client = None
        self._github_session = None
        
        # SafeClaw Governor Configuration
        self.hub_url = os.environ.get("SAFECLAW_HUB_URL", "http://localhost:8000")
        self.interceptor = None # Will be initialized on first run or boot
        self.auth_token = None
        self.session_id = "agent_session_" + os.urandom(4).hex()

    async def _ensure_governor_interceptor(self):
        """Fetch the central manifest from the Governor (Hub) and initialize interceptor."""
        if self.interceptor:
            return

        manifest = await self._init_scoped_session()
        self.interceptor = ManifestInterceptor(manifest)
        logger.info(f"SafeClaw Governor initialized with policy: {manifest.name}")

    async def _init_scoped_session(self) -> SkillManifest:
        """Fetch delegation token and scoped manifest from Hub."""
        try:
            async with httpx.AsyncClient() as client:
                await self._ensure_hub_running(client)
                
                # 1. Fetch Delegation Token
                profile = os.environ.get("SAFECLAW_AGENT_PROFILE", "read_only")
                auth_resp = await client.post(
                    f"{self.hub_url}/auth/delegate", 
                    json={"session_id": self.session_id, "profile": profile},
                    timeout=10.0
                )
                auth_resp.raise_for_status()
                auth_data = auth_resp.json()
                self.auth_token = auth_data.get("token")
                
                # 2. Fetch Public Key for verification
                pub_resp = await client.get(f"{self.hub_url}/manifest/pubkey", timeout=10.0)
                pub_resp.raise_for_status()
                pub_pem = pub_resp.json().get("pubkey")

                # 3. Fetch Scoped Manifest
                man_resp = await client.get(
                    f"{self.hub_url}/manifest/scoped", 
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    timeout=10.0
                )
                man_resp.raise_for_status()
                data = man_resp.json()

                return self._verify_and_parse_manifest(data, pub_pem)
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to initialize scoped session: {e}. Falling back to restricted policy.")
            return DEFAULT_MANIFEST
        except Exception as e:
            logger.error(f"Unexpected error initializing session: {e}")
            return DEFAULT_MANIFEST

    def _verify_and_parse_manifest(self, data: dict, pub_pem: Optional[str]) -> SkillManifest:
        """Check the Hub's digital signature against its public key."""
        from .crypto import verify_signature
        from cryptography.hazmat.primitives import serialization
        
        manifest_dict = data.get("manifest")
        signature_hex = data.get("signature")
        
        if not all([manifest_dict, signature_hex, pub_pem]):
            logger.warning("Security material missing from Hub response.")
            return DEFAULT_MANIFEST

        # Verification Invariant: Signature must match canonicalized JSON
        manifest_json = json.dumps(manifest_dict, sort_keys=True)
        pub_key = serialization.load_pem_public_key(pub_pem.encode())
        
        if verify_signature(manifest_json.encode(), bytes.fromhex(signature_hex), pub_key):
            return SkillManifest.from_dict(manifest_dict)
        
        logger.error("SECURITY ALERT: Manifest signature verification failed!")
        return DEFAULT_MANIFEST

    async def _ensure_hub_running(self, client: httpx.AsyncClient):
        """Heartbeat check; triggers local boot if Governor is missing."""
        try:
            # Short timeout for the presence check
            await client.get(f"{self.hub_url}/health", timeout=2.0)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ProtocolError):
            logger.info(f"Governor not found at {self.hub_url}. Attempting local boot...")
            await self._start_local_hub()
            
            # Poll for max 10 seconds (PR Feedback resolution)
            for i in range(10):
                try:
                    await client.get(f"{self.hub_url}/health", timeout=1.0)
                    logger.info("Local hub is responsive.")
                    break
                except (httpx.ConnectError, httpx.TimeoutException):
                    await asyncio.sleep(1.0)
            else:
                logger.error("Local hub did not become responsive in time.")

    async def _start_local_hub(self):
        """Spawn a local instance of the Observability Hub."""
        import subprocess
        import sys
        from urllib.parse import urlparse
        
        parsed = urlparse(self.hub_url)
        port = parsed.port or 8000
        
        # Launch Hub as background process
        cmd = [sys.executable, "-m", "uvicorn", "med_safety_eval.observability_hub:app", "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(proc.terminate)
        logger.info(f"Local SafeClaw Hub spawned on port {port} (Cleanup registered).")

    async def run(self, message: Message, updater: Any, session: Any = None) -> None:
        """
        Main entry point for A2A loop.
        Extracts the user's message and performs safety-checked action.
        
        Args:
            message: A2A message from user
            updater: Task status updater
            session: Optional SessionMemory for conversation context
        """
        # 0. Ensure Governor Interceptor
        await self._ensure_governor_interceptor()

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
        # Use contextual regex patterns to avoid false positives (e.g., 'admin' in 'administer')
        github_patterns = [
            r'\b(configure_repo|delete_repo|create_issue|list_issues|list_pull_requests)\b',
            r'\b(unlock|admin)\s+(tools|permissions|access)\b',
            r'\b(delete|remove)\s+(repo|repository|comment)\b',
            r'^(list|show|get|create|new|add)\s+(issues?|prs?|pulls?|repos?|repositories|comment)\b'
        ]
        
        is_github = (
            action.lower().startswith("gh:") or 
            'gh:' in action.lower() or
            re.search(r'^(!v|voice|say)\b', action.lower()) or
            any(re.search(p, action.lower()) for p in github_patterns)
        )

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

    async def _get_github_session(self):
        """Persistent GitHub session helper."""
        if self._github_session is None:
            self._github_client = self.github_client_factory()
            self._github_session = await self._github_client.__aenter__()
        return self._github_session

    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        await self._close_github_session()

    async def _close_github_session(self):
        """Cleanup GitHub session."""
        if self._github_client:
            await self._github_client.__aexit__(None, None, None)
            self._github_client = None
            self._github_session = None

    async def github_action(self, action: str, updater: Any, session: Any = None) -> None:
        """Interacts with the GitHub MCP server to perform repo operations."""
        await updater.update_status(TaskState.working, new_agent_text_message(f"🔄 Processing GitHub operation: {action}..."))
        
        session_client = await self._get_github_session()
        if session and session.github_repo:
             await session_client.call_tool("configure_repo", {"repo_name": session.github_repo})
        
        try:
            cmd = action.lower().replace("gh:", "").strip()
            result = await self._dispatch_github_cmd(cmd, action, session_client, updater, session)
            if result is not None:
                await updater.update_status(TaskState.completed, new_agent_text_message(str(result)))
        except Exception as e:
            logger.error(f"GitHub Action Failed: {e}")
            await updater.update_status(TaskState.failed, new_agent_text_message(f"❌ GitHub Error: {e}"))

    async def _dispatch_github_cmd(self, cmd: str, action: str, client: Any, updater: Any, session: Any) -> Any:
        """Dispatch GitHub command to specific handler."""
        if "unlock" in cmd and "admin" in cmd: return await self._handle_unlock_policy(updater)
        
        handlers = [
            (r'(set|configure|use)\s+repo\s+(?:to\s+)?(.+)', self._handle_repo_config),
            (r'\b(delete|remove)\s+(repo|repository)\b|delete_repo', self._handle_repo_deletion),
            (r'delete\s+comment\s+(\d+)\s+(?:on|from)\s+issue\s+(\d+)', self._handle_comment_deletion),
            (r'\bissues?\b', self._handle_issue_ops),
            (r'\b(pulls?|prs?|requests)\b', self._handle_pr_ops),
            (r'\b(check|info)\b', self._handle_info_ops),
            (r'\b(say|voice|!v)\b', self._handle_voice_ops)
        ]
        
        for pattern, handler in handlers:
            match = re.search(pattern, cmd)
            if match: return await handler(match, cmd, action, client, updater, session)
            
        return f"Command recognized as GitHub, but specific action unknown: '{cmd}'."

    async def _handle_unlock_policy(self, updater: Any):
        msg = "🔒 Zero-Trust Policy: Global session unlock is disabled. Request specific actions (e.g. 'gh: delete repo') for JIT confirmation."
        await updater.update_status(TaskState.working, new_agent_text_message(msg))
        return None

    async def _handle_repo_config(self, match, cmd, action, client, updater, session):
        repo_name = match.group(2).strip()
        res = await self._call_tool_with_interception("configure_repo", {"repo_name": repo_name}, client, updater, session)
        if res and session: session.github_repo = repo_name
        return res

    async def _handle_repo_deletion(self, match, cmd, action, client, updater, session):
        return await self._call_tool_with_interception("delete_repo", {}, client, updater, session)

    async def _handle_comment_deletion(self, match, cmd, action, client, updater, session):
        args = {"issue_number": int(match.group(2)), "comment_id": int(match.group(1))}
        return await self._call_tool_with_interception("delete_issue_comment", args, client, updater, session)

    async def _handle_issue_ops(self, match, cmd, action, client, updater, session):
        is_create = re.search(r'\b(create|new|add)\b', cmd)
        tool = "create_issue" if is_create else "list_issues"
        args = self._extract_issue_args(action) if is_create else {}
        return await self._call_tool_with_interception(tool, args, client, updater, session)

    def _extract_issue_args(self, action: str) -> dict:
        t = re.search(r'title="([^"]+)"', action)
        b = re.search(r'body="([^"]+)"', action)
        return {"title": t.group(1) if t else "Untitled", "body": b.group(1) if b else ""}

    async def _handle_pr_ops(self, match, cmd, action, client, updater, session):
        return await self._call_tool_with_interception("list_pull_requests", {}, client, updater, session)

    async def _handle_info_ops(self, match, cmd, action, client, updater, session):
        return await client.call_tool("list_issues", {})

    async def _handle_voice_ops(self, match, cmd, action, client, updater, session):
        for kw in ["say", "voice", "!v"]: cmd = re.sub(rf'\b{re.escape(kw)}\b', "", cmd, flags=re.IGNORECASE).strip()
        return cmd or "Voice mode activated. What would you like me to say?"

    async def _call_tool_with_interception(
        self,
        tool_name: str,
        tool_args: dict,
        session_client: Any,
        updater: Any,
        session: Any = None
    ) -> Optional[Any]:
        """Orchestrate tool execution with manifest checks and security guards."""
        check = await self._verify_and_gate_tool_call(tool_name, tool_args, updater, session)
        if not check or not check.allowed:
            return None

        if not await self._apply_security_guards(check, tool_name, tool_args, updater, session=session):
            return None

        return await session_client.call_tool(tool_name, tool_args)

    async def execute_confirmed_tool(self, tool_name: str, tool_args: dict, updater: Any, session: Any = None) -> None:
        """Bypass guards and execute a tool that has been manually approved by the user."""
        await self._ensure_governor_interceptor()
        
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

    async def _verify_and_gate_tool_call(self, tool_name: str, tool_args: dict, updater: Any, session: Any) -> Any:
        """Verify the tool call against the scoped token and manifest policy."""
        if not self.auth_token:
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message("🚨 BLOCKED: No delegation token present in session.")
            )
            return None
            
        secret = os.environ.get("JWT_SECRET")
        if not secret:
            logger.error("JWT_SECRET environment variable is missing!")
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message("🚨 BLOCKED: Server component configuration error (Identity).")
            )
            return None

        try:
            verify_delegation_token(self.auth_token, secret)
        except jwt.ExpiredSignatureError:
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message("🚨 BLOCKED: Delegation token has expired.")
            )
            return None
        except jwt.InvalidTokenError as e:
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message(f"🚨 BLOCKED: Invalid delegation token. {e}")
            )
            return None

        # Check against manifest interceptor rules
        audit_log = getattr(session, "audit_log", []) if session else []
        check = self.interceptor.intercept(tool_name, tool_args, audit_log)
        
        if not check.allowed:
            await updater.update_status(
                TaskState.failed, 
                new_agent_text_message(f"🚨 BLOCKED: {check.reason}")
            )
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
