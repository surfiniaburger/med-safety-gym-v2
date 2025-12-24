"""
ADK Agent for DIPG Evaluation with A2A Integration (LiteLLM)

This agent uses Google's Agent Development Kit (ADK) to orchestrate
DIPG safety evaluations via MCP tools, and exposes itself via the
Agent-to-Agent (A2A) protocol.

Uses LiteLLM for model inference to support various LLM providers.
"""

import logging
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams
from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

load_dotenv()

SYSTEM_INSTRUCTION = (
    "You are a specialized assistant for evaluating medical AI models using the DIPG Safety Gym. "
    "Your purpose is to help researchers and developers assess AI models for medical safety, "
    "including detection of hallucinations, reasoning quality, and appropriate abstention. "
    "\n\n"
    "You have access to two tools:\n"
    "1. 'get_eval_tasks' - Retrieve evaluation tasks from the DIPG dataset\n"
    "2. 'evaluate_responses' - Score model responses for safety metrics\n"
    "\n"
    "When users ask for evaluation tasks, use get_eval_tasks. "
    "When they provide model responses to evaluate, use evaluate_responses. "
    "Always explain the safety metrics in the results clearly."
)

logger.info("--- 🔧 Loading MCP tools from FastMCP Server... ---")
logger.info("--- 🤖 Creating ADK DIPG Evaluation Agent (LiteLLM)... ---")

# Create the ADK agent using LiteLLM
# Default to ollama/llama3 if not specified, or use user's suggestion
model_name = os.getenv("LITELLM_MODEL", "ollama/gpt-oss:20b-cloud")

root_agent = Agent(
    name="dipg_eval_agent",
    model=LiteLlm(model=model_name),
    description="An agent that helps evaluate medical AI models for safety using the DIPG Safety Gym",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=os.getenv("MCP_SERVER_URL", "http://localhost:8081/mcp")
            )
        )
    ],
)

logger.info(f"✅ ADK Agent created successfully (Model: {model_name})")
logger.info("--- 🔗 Creating A2A wrapper... ---")

# Make the agent A2A-compatible
a2a_app = to_a2a(root_agent, port=int(os.getenv("A2A_PORT", 10000)))

logger.info("✅ A2A wrapper created")
logger.info(f"🚀 Agent ready to serve on port {os.getenv('A2A_PORT', 10000)}")
