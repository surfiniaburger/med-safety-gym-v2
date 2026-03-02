
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
from .identity.secret_store import SecretStore, KeyringSecretStore
import jwt
from .intent_classifier import IntentClassifier, IntentCategory, IntentResult

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
        github_client_factory: Optional[Callable[[], Any]] = None,
        secret_store: Optional[SecretStore] = None
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
        self.hub_pub_key = None # Governor's public key for identity verification
        self.secret_store = secret_store or KeyringSecretStore()
        self.session_id = "agent_session_" + os.urandom(4).hex()

    async def _ensure_governor_interceptor(self):
        """Fetch the central manifest from the Governor (Hub) and initialize interceptor."""
        if self.interceptor:
            return

        # Attempt to load existing secrets from store before fetching
        await self._load_secrets_from_store()

        manifest = await self._init_scoped_session()
        self.interceptor = ManifestInterceptor(manifest)
        logger.info(f"SafeClaw Governor initialized with policy: {manifest.name}")

    async def _load_secrets_from_store(self):
        """Pre-load session secrets from persistent storage."""
        if not self.auth_token:
            self.auth_token = self.secret_store.get_secret("auth_token")
            if self.auth_token:
                logger.info("Loaded delegation token from secret store.")
        
        if not self.hub_pub_key:
            self.hub_pub_key = self.secret_store.get_secret("hub_pub_key")
            if self.hub_pub_key:
                logger.info("Loaded Governor public key from secret store.")

    async def _init_scoped_session(self) -> SkillManifest:
        """Fetch delegation token and scoped manifest from Hub."""
        try:
            async with httpx.AsyncClient() as client:
                await self._ensure_hub_running(client)
                
                # 1. Attempt session resumption if token and pubkey are already loaded
                if self.auth_token and self.hub_pub_key:
                    try:
                        logger.info("Attempting to resume session with stored credentials...")
                        man_resp = await client.get(
                            f"{self.hub_url}/manifest/scoped", 
                            headers={"Authorization": f"Bearer {self.auth_token}"},
                            timeout=5.0
                        )
                        if man_resp.status_code == 200:
                            data = man_resp.json()
                            manifest = self._verify_and_parse_manifest(data, self.hub_pub_key)
                            if manifest != DEFAULT_MANIFEST:
                                logger.info("Session resumed successfully.")
                                return manifest
                        logger.warning("Stored token invalid or expired. Proceeding with full handshake.")
                    except jwt.ExpiredSignatureError:
                        logger.warning("Delegation token has expired (TTL exceeded). Performing full handshake.")
                    except (httpx.HTTPError, json.JSONDecodeError, jwt.PyJWTError) as e:
                        logger.warning(f"Session resumption failed: {e}. Handshaking...")
                    except Exception as e:
                        logger.error(f"Unexpected error during session resumption: {e}")

                # 2. Full Handshake: Fetch Delegation Token
                profile = os.environ.get("SAFECLAW_AGENT_PROFILE", "read_only")
                auth_resp = await client.post(
                    f"{self.hub_url}/auth/delegate", 
                    json={"session_id": self.session_id, "profile": profile},
                    timeout=10.0
                )
                auth_resp.raise_for_status()
                auth_data = auth_resp.json()
                self.auth_token = auth_data.get("token")
                if self.auth_token:
                    self.secret_store.set_secret("auth_token", self.auth_token)
                
                # 3. Fetch Public Key for verification
                pub_resp = await client.get(f"{self.hub_url}/manifest/pubkey", timeout=10.0)
                pub_resp.raise_for_status()
                pub_pem = pub_resp.json().get("pubkey")
                self.hub_pub_key = pub_pem
                if self.hub_pub_key:
                    self.secret_store.set_secret("hub_pub_key", pub_pem)

                # 4. Fetch Scoped Manifest
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
        
        from .intent_classifier import IntentClassifier, IntentCategory
        
        # 1. Classify Intent
        classifier = IntentClassifier()
        intent = classifier.classify(text_content)
        
        # 2. Apply Mediator Pattern (Structural enrichment for context injection)
        # We also pass the intent to context_aware_action to help with safety gating
        if intent.category != IntentCategory.NEW_TOPIC or intent.is_correction:
            action = f"[{intent.category.name} | Correction: {intent.is_correction}] {text_content}"
        else:
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
        
        await self.context_aware_action(action, text_content, context, updater, intent=intent)

    async def _get_github_session(self):
        """Persistent GitHub session helper."""
        if self._github_session is None:
            self._github_client = self.github_client_factory()
            self._github_session = await self._github_client.__aenter__()
        return self._github_session

    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info("SafeClaw Agent shutting down...")
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
            (r'\b(say|voice|!v)\b', self._handle_voice_ops),
            (r'\b(repos?|repositories)\b', self._handle_repo_list_fallback)
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

    async def _handle_repo_list_fallback(self, match, cmd, action, client, updater, session):
        return "I can manage your current repository, issues, and PRs, but I don't have a direct tool to list all repositories. You can configure a specific repository via `gh: set repo <repo_name>`."

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
            
        if not self.hub_pub_key:
            # If we lost the key somehow, re-fetch or fail
            await self._ensure_governor_interceptor()
            if not self.hub_pub_key:
                await updater.update_status(
                    TaskState.failed, 
                    new_agent_text_message("🚨 BLOCKED: Unable to retrieve Governor public key.")
                )
                return None

        try:
            # Verify using the Governor's public key (Asymmetric)
            verify_delegation_token(self.auth_token, self.hub_pub_key)
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

    async def context_aware_action(self, action: str, raw_text: str, context: str, updater: Any, intent: Optional[IntentResult] = None) -> None:
        """
        Executes an action using the LLM. 
        Note: The Entity Parity check has been removed here because it was blocking 
        valid conversational user inputs (like asking "What are the side effects?").
        Safety is instead enforced by a combination of the 'check_entity_parity' safety gate 
        (for new medical actions) and the strict system prompt bound to the verified context.
        """
        # Safety Gate: Only run entity parity for new topics or explicit actions
        # This prevents blocking conversational follow-ups while maintaining safety for new intents.
        if intent is None or intent.category == IntentCategory.NEW_TOPIC:
            is_safe, failure_reason = await self._apply_safety_gate(action, context, updater)
            if not is_safe:
                await updater.update_status(TaskState.failed, new_agent_text_message(f"❌ Safety Violation: {failure_reason}"))
                return

        await updater.update_status(TaskState.working, new_agent_text_message(f"✅ Input received. Generating response..."))
        logger.info(f"Action Executed (Generating LLM response): {action}")

        try:
            from litellm import acompletion
            import os
            
            # Use environment variable or fallback to a default Gemini model
            model = os.environ.get("LITELLM_MODEL") or os.environ.get("USER_LLM_MODEL") or "gemini/gemini-2.5-flash"
            
            prompt = (
                f"You are SafeClaw, a strict but helpful medical AI assistant. "
                f"Use the following verified context to answer the user.\n"
                f"If the user asks a question not covered by the context, you MUST state that you do not know.\n\n"
                f"Context:\n{context}\n\n"
                f"User: {action}\n\n"
                f"SafeClaw:"
            )
            
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            
            # Handle both object and dict responses for robustness
            message_obj = response.choices[0].message
            if hasattr(message_obj, "content"):
                response_text = message_obj.content
            else:
                response_text = message_obj.get("content") if isinstance(message_obj, dict) else str(message_obj)
                
            await updater.update_status(TaskState.completed, new_agent_text_message(str(response_text)))
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, new_agent_text_message(f"❌ Failed to generate response: {e}"))

    async def _apply_safety_gate(self, action: str, context: str, updater: Any) -> tuple[bool, str]:
        """
        Dedicated safety gate logic (Dave Farley Habit: Small, focused function).
        Verifies entity parity before allowing an action to proceed.
        """
        await updater.update_status(TaskState.working, new_agent_text_message(f"🛡️ Running safety verification: {action}..."))
        
        async with self.client_factory() as safety_client:
            safety_check = await safety_client.call_tool("check_entity_parity", {
                "action": action,
                "context": context
            })
            
            if not safety_check.get("is_safe", False):
                reason = safety_check.get("reason", "Unknown safety violation.")
                logger.warning(f"Safety Gate Blocked Action: {reason}")
                return False, reason
                
        return True, ""
