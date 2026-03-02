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

# This script can now be used as a CLI entry point for local evaluations.
# The core logic is shared with the A2A Benchmark Server.
