import os
import json
import logging
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VisionAuditService:
    """
    Generates human-readable intent summaries for tool calls using Gemini.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    async def audit_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Produce a concise (1-sentence) summary of what the tool will do.
        """
        if not self.api_key:
            return f"Agent intends to call {tool_name} with {json.dumps(args)}"

        prompt = (
            f"You are a Security Audit AI. Summarize the user's intent based on this tool call in ONE concise sentence. "
            f"Be neutral but clear about destructive actions.\n\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {json.dumps(args)}"
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_url}?key={self.api_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.0}
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Vision Audit failed: {e}")
            return f"The agent is requesting to execute '{tool_name}' on the repository."

async def get_audit_summary(tool_name: str, args: Dict[str, Any]) -> str:
    """Convenience helper for auditing."""
    service = VisionAuditService()
    return await service.audit_tool_call(tool_name, args)
