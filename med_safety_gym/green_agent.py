"""
DIPG Safety Gym - Green Agent (A2A Evaluator)

This agent implements the A2A protocol for AgentBeats Green Agents.
It receives an EvalRequest, executes the safety evaluation against the
target Purple Agent, and reports the results via artifacts.
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from .app import get_environment
from .evaluation_service import EvaluationManager, EvaluationItem, GroundTruth
from .messenger import Messenger

logger = logging.getLogger(__name__)

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: Dict[str, HttpUrl] # role -> agent URL
    config: Dict[str, Any]

class GreenAgent:
    required_roles: List[str] = ["purple_agent"]
    
    def __init__(self):
        self.env = get_environment()
        self.eval_manager = EvaluationManager(self.env)
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Implementation of the Green Agent logic for AgentBeats.
        """
        input_text = get_message_text(message)
        
        try:
            # Parse the incoming EvalRequest
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working, 
            new_agent_text_message("Starting medical safety evaluation...")
        )

        # 1. Configuration
        purple_agent_url = str(request.participants["purple_agent"])
        num_samples = request.config.get("num_samples", 10)
        
        # 2. Fetch Tasks
        tasks = self.env.get_eval_tasks(max_samples=num_samples, shuffle=True)
        evaluations = []

        # 3. Assessment Loop
        for i, task in enumerate(tasks):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating task {i+1}/{num_samples}: {task['task_id']}")
            )
            
            prompt = f"Context: {task['context']}\n\nQuestion: {task['question']}"
            
            try:
                # Talk to Purple Agent via A2A
                response_text = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_agent_url
                )
                
                evaluations.append(EvaluationItem(
                    response=response_text,
                    ground_truth=GroundTruth(
                        context=task['context'],
                        question=task['question'],
                        expected_answer=task['expected_answer']
                    )
                ))
            except Exception as e:
                logger.error(f"Failed to query Purple Agent: {e}")
                evaluations.append(EvaluationItem(
                    response=f"ERROR: {e}",
                    ground_truth=GroundTruth(
                        context=task['context'],
                        question=task['question'],
                        expected_answer=task['expected_answer']
                    )
                ))

        # 4. Batch Scoring
        result = self.eval_manager.evaluate_with_ground_truth(evaluations)
        
        # 5. Report Results as A2A Artifact
        summary_dump = result.model_dump()
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(kind="text", text=f"Assessment complete. Mean Reward: {result.mean_reward:.2f}")),
                Part(root=DataPart(kind="data", data={
                    "summary": summary_dump,
                    "detailed_results": result.detailed_results
                }))
            ],
            name="Assessment Result",
        )
        
        logger.info(f"✅ Assessment complete for {purple_agent_url}")
