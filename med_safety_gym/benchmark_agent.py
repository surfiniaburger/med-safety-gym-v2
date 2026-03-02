"""
Benchmark Agent Implementation.
Exposes prober capabilities over the A2A protocol.
"""

import logging
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, HttpUrl

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from .messenger import Messenger
from .eval_core import (
    EvaluationOrchestrator,
    RecollectionProber,
    InformationSeekingProber,
    AMIEDiagnosticProber,
    HealthBenchGrader
)
from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger(__name__)

class BenchmarkRequest(BaseModel):
    """Request format for starting a benchmark round."""
    target_agent_url: HttpUrl
    scenario: str = "recollection" # recollection, mediq, amie
    num_turns: int = 4

class BenchmarkAgent:
    """
    A2A Agent that runs prober scenarios against a target agent.
    """
    def __init__(self):
        self.messenger = Messenger()
        self.grader = HealthBenchGrader()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        
        try:
            request = BenchmarkRequest.model_validate_json(input_text)
        except Exception as e:
            await updater.reject(new_agent_text_message(f"Invalid BenchmarkRequest: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting benchmark scenario: {request.scenario}")
        )

        # 1. Select Prober
        if request.scenario == "recollection":
            prober = RecollectionProber(name="RecollectionProber")
        elif request.scenario == "mediq":
            prober = InformationSeekingProber(name="MediQProber")
        elif request.scenario == "amie":
            prober = AMIEDiagnosticProber(name="AMIEProber")
        else:
            await updater.failed(new_agent_text_message(f"Unknown scenario: {request.scenario}"))
            return

        # 2. Setup Orchestrator (Mocking InvocationContext for now as we are bridging ADK -> A2A)
        # In a full ADK-native server this would be simpler.
        orchestrator = EvaluationOrchestrator(prober=prober)
        
        from unittest.mock import MagicMock, AsyncMock
        ctx = MagicMock(spec=InvocationContext)
        ctx.session = MagicMock()
        ctx.session.state = {}
        
        # Robustly mock ADK internal plugin manager for A2A bridging
        ctx.plugin_manager = AsyncMock()
        ctx.plugin_manager.run_before_agent_callback.return_value = None
        ctx.plugin_manager.run_after_agent_callback.return_value = None
        ctx.plugin_manager.run_on_agent_event_callback.return_value = None

        # 3. Execution Loop
        target_url = str(request.target_agent_url)
        subject_input = "System Start" # Initial trigger for some probers
        
        for turn_idx in range(request.num_turns):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Turn {turn_idx + 1}/{request.num_turns}")
            )
            
            # Get Prober's next move
            prober_msg = await orchestrator.run_turn(ctx, subject_input)
            
            # Send to Target Agent via A2A
            try:
                subject_input = await self.messenger.talk_to_agent(
                    message=prober_msg,
                    url=target_url,
                    new_conversation=(turn_idx == 0)
                )
            except Exception as e:
                logger.error(f"Target agent failed: {e}")
                subject_input = f"ERROR: {e}"
                break

        # 4. Grading
        report = self.grader.grade_transcript(orchestrator.history)
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(kind="text", text=f"Benchmark Complete. Score: {report['overall_score']:.2f}")),
                Part(root=DataPart(kind="data", data={
                    "scenario": request.scenario,
                    "target": target_url,
                    "report": report,
                    "history": orchestrator.history
                }))
            ],
            name="Benchmark Report"
        )
        
        await updater.complete(new_agent_text_message("Benchmark finished successfully."))
