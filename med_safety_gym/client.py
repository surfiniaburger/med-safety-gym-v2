# src/envs/dipg_safety_env/client.py
"""
Client implementation for the custom DIPGSafetyEnv.

This file defines the `DIPGSafetyEnv` class, which acts as the "remote control"
for the environment server. Its primary job is to handle the communication:
  1.  It takes Python objects (like an Action) from the agent's code.
  2.  It converts them into JSON to send to the server via WebSockets.
  3.  It receives JSON responses from the server.
  4.  It parses that JSON back into structured objects (DIPGStepResult).
"""

import requests
import statistics
import json
import os
from typing import Any, Dict, Generic, Optional, Type, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# Legacy TypeVars for backward compatibility
ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")

from .models import DIPGAction, DIPGObservation, DIPGState

@dataclass
class DIPGStepResult(StepResult[DIPGObservation]):
    """
    DIPG-specific StepResult that includes evaluation metrics in the 'info' field
    for backward compatibility with the legacy evaluation suite.
    """
    info: Dict[str, Any] = field(default_factory=dict)

class DIPGSafetyEnv(EnvClient[DIPGAction, DIPGObservation, DIPGState]):
    """
    Client for interacting with the `DIPGSafetyEnv` server.
    """
    
    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initializes the client.
        """
        # Store original base for custom HTTP endpoints (like fetching tasks)
        self._base = base_url.rstrip("/")
        self._http = requests.Session()
        self._timeout = timeout
        
        super().__init__(
            base_url=base_url, 
            message_timeout_s=timeout, 
            connect_timeout_s=timeout
        )

    def _step_payload(self, action: DIPGAction) -> dict:
        """Formats the action for transmission."""
        return {"llm_response": action.llm_response}

    def _parse_result(self, payload: dict) -> DIPGStepResult:
        """Parses server response into structured DIPGStepResult."""
        obs_data = payload.get("observation")

        if isinstance(obs_data, dict) and "observation" in obs_data:
            actual_obs_data = obs_data.get("observation")
        else:
            actual_obs_data = obs_data if isinstance(obs_data, dict) else {}

        if not isinstance(actual_obs_data, dict):
            actual_obs_data = {}
        
        # Integration: Extract metrics from metadata for easy access via .info
        # Check both the possible nested location and the top-level obs_data
        metrics = actual_obs_data.get("metadata", {}) or (obs_data.get("metadata", {}) if isinstance(obs_data, dict) else {})

        obs = DIPGObservation(
            context=actual_obs_data.get("context", ""),
            question=actual_obs_data.get("question", ""),
            metadata=metrics
        )
        
        return DIPGStepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            info=metrics
        )

    def _parse_state(self, payload: dict) -> DIPGState:
        """Parses state payload."""
        return DIPGState(**payload)
    
    # ==================================================================================
    # INTEGRATED EVALUATION METHODS
    # ==================================================================================
    
    def evaluate_model(
        self,
        responses: list[str],
        response_format: str = "auto",
        save_path: str | None = None
    ) -> dict:
        """
        Evaluate a batch of model responses using integrated environment steps.
        
        This method performs a standard gym loop (reset + step) for each response,
        aggregating metrics directly from the StepResult.info field.
        """
        rewards = []
        detailed_results = []
        
        # Metric Counters
        metrics_counts = {
            "refusal": 0,
            "safe": 0,
            "hallucination": 0,
            "inconsistency": 0,
            "format_error": 0
        }
        
        for idx, response in enumerate(responses):
            try:
                obs_result = self.reset()
                step_result = self.step(DIPGAction(llm_response=response))
                
                reward = step_result.reward or 0.0
                metrics = step_result.info
                
                rewards.append(reward)
                
                for key in metrics_counts.keys():
                    if metrics.get(key):
                        metrics_counts[key] += 1
                
                detailed_results.append({
                    "index": idx,
                    "response": response,
                    "reward": reward,
                    "metrics": metrics,
                    "context": obs_result.observation.context,
                    "question": obs_result.observation.question
                })
            except Exception as e:
                rewards.append(-1.0) 
                detailed_results.append({"index": idx, "error": str(e), "reward": -1.0})

        total = len(responses)
        result = {
            "total_responses": total,
            "mean_reward": statistics.mean(rewards) if rewards else 0.0,
            "median_reward": statistics.median(rewards) if rewards else 0.0,
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "rewards": rewards,
            "refusal_rate": metrics_counts["refusal"] / total if total > 0 else 0.0,
            "safe_response_rate": metrics_counts["safe"] / total if total > 0 else 0.0,
            "medical_hallucination_rate": metrics_counts["hallucination"] / total if total > 0 else 0.0,
            "reasoning_consistency_rate": metrics_counts["inconsistency"] / total if total > 0 else 0.0,
            "detailed_results": detailed_results
        }
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            result["saved_to"] = save_path

        return result

    def get_eval_tasks(self, max_samples: int = None, shuffle: bool = True) -> list[dict]:
        """
        Fetch task list (using the coordinated /eval/tasks endpoint).
        """
        params = {"shuffle": str(shuffle).lower()}
        if max_samples is not None:
            params["max_samples"] = max_samples
            
        response = requests.get(
            f"{self._base}/eval/tasks",
            params=params,
            timeout=self._timeout
        )
        response.raise_for_status()
        return response.json().get("tasks", [])
    
    def get_metrics_summary(self) -> dict:
        """
        Get environment configuration from integrated state.
        """
        state = self.state()
        return getattr(state, 'config', {})