import os
import json
import logging
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VisionAuditService:
    """Generates human-readable intent summaries for tool calls using Gemini."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or self._get_env_api_key()
        self.model = "gemini-2.0-flash"

    def _get_env_api_key(self) -> str:
        """Retrieve API key from environment (prioritize GOOGLE_API_KEY)."""
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def _get_url(self) -> str:
        """Construct the Gemini API URL."""
        base = "https://generativelanguage.googleapis.com/v1beta/models"
        return f"{base}/{self.model}:generateContent?key={self.api_key}"

    def _build_payload(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Build the Gemini generation payload."""
        prompt = (
            f"You are a Security Audit AI. Summarize the user's intent based on this tool call in ONE concise sentence. "
            f"Be neutral but clear about destructive actions.\n\n"
            f"Tool: {tool_name}\nArguments: {json.dumps(args)}"
        )
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }

    async def audit_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Produce a concise (1-sentence) summary of what the tool will do."""
        if not self.api_key:
            return f"Agent intends to call {tool_name} with {json.dumps(args)}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self._get_url(), json=self._build_payload(tool_name, args), timeout=10.0)
                resp.raise_for_status()
                return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Vision Audit failed: {e}")
            return f"The agent is requesting to execute '{tool_name}' on the repository."

async def get_audit_summary(tool_name: str, args: Dict[str, Any]) -> str:
    """Convenience helper for auditing."""
    return await VisionAuditService().audit_tool_call(tool_name, args)
