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

from .benchmark_agent import BenchmarkAgent
from .executor import Executor

logger = logging.getLogger(__name__)

def create_app(host="0.0.0.0", port=8004, card_url=None):
    """Factory to create the Benchmark A2A application instance."""
    skill = AgentSkill(
        id="med-safety-benchmark",
        name="Medical Safety Benchmarking",
        description="Orchestrates prober-based evaluations against other agents.",
        tags=["benchmarking", "safety", "evaluation"],
        examples=[
            "target_agent_url: http://localhost:8003/, scenario: recollection, num_turns: 4"
        ]
    )

    agent_card = AgentCard(
        name="BenchmarkServer",
        description="A specialized service for running agentic benchmarks.",
        url=card_url or f"http://{host}:{port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text', 'data'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(BenchmarkAgent),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    return server.build()

def main():
    parser = argparse.ArgumentParser(description="Run the Med-Safety Benchmark Server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind")
    args = parser.parse_args()

    app = create_app(host=args.host, port=args.port)
    print(f"🚀 Benchmark Server serving on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
