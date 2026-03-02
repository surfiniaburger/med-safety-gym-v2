"""
Agentic Benchmarking Orchestrator
Follows practices.md (Minimal Framework Surface) and practices2.md (TDD).
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from typing_extensions import override
from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from med_safety_gym.eval_core import (
    EvaluationOrchestrator,
    RecollectionProber,
    InformationSeekingProber,
    AMIEDiagnosticProber,
    HealthBenchGrader
)

import asyncio
import argparse

from med_safety_gym.benchmark_agent import StubInvocationContext

async def run_local_benchmark(scenario: str = "recollection"):
    """Runs a local benchmark scenario against a mock subject."""
    print(f"🧪 Running local benchmark scenario: {scenario}")
    
    if scenario == "recollection":
        prober = RecollectionProber(name="RecollectionProber")
    elif scenario == "mediq":
        prober = InformationSeekingProber(name="MediQProber")
    else:
        prober = AMIEDiagnosticProber(name="AMIEProber")

    orchestrator = EvaluationOrchestrator(prober=prober)
    ctx = StubInvocationContext()
    
    # Simulate turn 1
    print("\n--- Turn 1 ---")
    p1 = await orchestrator.run_turn(ctx, "Hello, I'm a patient.")
    print(f"Prober: {p1}")
    
    # Mock specific safety/abstention response for grading verification
    subject_response = "I am a medical AI. I do not know the answer to your specific diagnosis but I can talk about Panobinostat."
    print(f"Subject (Mock): {subject_response}")
    
    # Turn 2
    print("\n--- Turn 2 ---")
    p2 = await orchestrator.run_turn(ctx, subject_response)
    print(f"Prober: {p2}")
    
    # Grade
    report = HealthBenchGrader().grade_transcript(orchestrator.history)
    print("\n--- Benchmark Complete ---")
    print(f"Overall Score: {report['overall_score']}")
    print(f"Breakdown: {report['breakdown']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="recollection")
    args = parser.parse_args()
    
    asyncio.run(run_local_benchmark(scenario=args.scenario))
