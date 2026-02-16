
import argparse
import uvicorn
import logging

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .claw_agent import SafeClawAgent
from .executor import Executor

def create_app(host="0.0.0.0", port=8003, card_url=None):
    """Factory to create the A2A application instance."""
    # Define the skill
    skill = AgentSkill(
        id="safe-claw-guardian",
        name="Safe Medical Operations",
        description="Executes medical operations with strict Entity Parity verification via MCP tools.",
        tags=["medical", "safety", "guardian", "mcp"],
        examples=[]
    )

    agent_card = AgentCard(
        name="SafeClaw",
        description="A safe, entity-parity-enforced agent for medical operations.",
        url=card_url or f"http://{host}:{port}/",
        version='2.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    # Use the Executor pattern from green_server
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(SafeClawAgent),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    starlette_app = server.build()

    @starlette_app.on_event("shutdown")
    async def on_shutdown():
        logger.info("🛑 Shutting down SafeClaw A2A Server...")
        await request_handler.agent_executor.shutdown()

    return starlette_app

app = create_app()

def main():
    parser = argparse.ArgumentParser(description="Run the SafeClaw Agent with Guardian safety checks.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Prioritize environment variable for CARD_URL
    import os
    card_url = args.card_url or os.environ.get("CARD_URL")

    global app
    app = create_app(host=args.host, port=args.port, card_url=card_url)
    
    print(f"🤖 SafeClaw Agent serving on {args.host}:{args.port}")
    print(f"🛡️  Guardian safety checks: ENABLED")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
