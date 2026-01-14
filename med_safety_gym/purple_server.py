import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .purple_executor import PurpleExecutor


def main():
    parser = argparse.ArgumentParser(description="Run the Purple Safety Agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="URL for the agent card")
    args = parser.parse_args()

    # Prioritize environment variable for CARD_URL
    import os
    card_url = args.card_url or os.environ.get("CARD_URL")

    skill = AgentSkill(
        id="med-safety-qa",
        name="Medical Safety Question Answering",
        description="Answers medical questions based on provided context, using a strict XML format to ensure safety and traceability. Abstains if the answer is not in the context.",
        tags=["medical", "safety", "RAG", "DIPG"],
        examples=[
            "Given a context about a clinical trial, what is the primary outcome?",
            "Based on this document, what are the side effects of the treatment?"
        ]
    )

    agent_card = AgentCard(
        name="DIPG Safety Purple Agent",
        description="A safety-tuned agent for the Med-Safety-Gym DIPG benchmark. It is designed to avoid hallucinations and abstain from answering when information is not present.",
        url=card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
