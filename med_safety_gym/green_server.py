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

from .green_agent import GreenAgent
from .executor import Executor

def main():
    parser = argparse.ArgumentParser(description="Run the DIPG Safety Gym Green Agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=10001, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Define the skill
    skill = AgentSkill(
        id="medical-safety-eval",
        name="Medical Safety Evaluation",
        description="Evaluates AI models for medical safety using the DIPG benchmark.",
        tags=["medical", "safety", "evaluation"],
        examples=[]
    )

    agent_card = AgentCard(
        name="DIPG Green Agent",
        description="A specialized evaluator for medical AI safety.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    # Use our custom Executor which instantiates GreenAgent
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(GreenAgent),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 DIPG Green Agent serving on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)

if __name__ == '__main__':
    main()
