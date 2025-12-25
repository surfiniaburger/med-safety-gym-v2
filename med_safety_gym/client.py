# src/envs/dipg_safety_env/client.py
"""
Client implementation for the custom DIPGSafetyEnv.

This file defines the `DIPGSafetyEnv` class, which acts as the "remote control"
for the environment server. Its primary job is to handle the HTTP communication:
  1.  It takes Python objects (like an Action) from the agent's code.
  2.  It converts them into JSON to send to the server.
  3.  It receives JSON responses from the server.
  4.  It parses that JSON back into useful Python objects (like Observations and Rewards).
"""

import requests
import importlib

# Robust import for openenv_core components due to potential path variations
HTTPEnvClient = None
StepResult = None

_POSSIBLE_PATHS = [
    'openenv_core.http_env_client',
    'openenv.core.http_env_client',
    'openenv_core',
    'openenv.core'
]

for path in _POSSIBLE_PATHS:
    try:
        module = importlib.import_module(path)
        if HTTPEnvClient is None and hasattr(module, 'HTTPEnvClient'):
            HTTPEnvClient = getattr(module, 'HTTPEnvClient')
        if StepResult is None and hasattr(module, 'StepResult'):
            StepResult = getattr(module, 'StepResult')
    except (ImportError, ModuleNotFoundError):
        continue

# Check specifically for StepResult in client_types if not found yet
if StepResult is None:
    for path in ['openenv_core.client_types', 'openenv.core.client_types']:
        try:
            module = importlib.import_module(path)
            if hasattr(module, 'StepResult'):
                StepResult = getattr(module, 'StepResult')
                break
        except (ImportError, ModuleNotFoundError):
            continue

if HTTPEnvClient is None:
    # Fallback to prevent immediate crash, though inheritance will fail if this is None.
    # We raise specific error to help debug.
    raise ImportError("Could not find HTTPEnvClient in openenv-core. Check installation.")

if StepResult is None:
    # Fallback shim
    class StepResult:
        def __init__(self, observation, reward, done, info=None):
            self.observation = observation
            self.reward = reward
            self.done = done
            self.info = info or {}

from .models import DIPGAction, DIPGObservation, DIPGState


class DIPGSafetyEnv(HTTPEnvClient[DIPGAction, DIPGObservation]):
    """
    Client for interacting with the `DIPGSafetyEnv` server.

    This class inherits from the base `HTTPEnvClient` and is specialized to handle
    the specific data types of our environment: `DIPGAction` and `DIPGObservation`.
    """
    
    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initializes the client.
        
        Args:
            base_url: The URL of the running environment server.
            timeout: The number of seconds to wait for a server response.
        """
        # This correctly calls the parent initializer with the expected
        # 'request_timeout_s' keyword argument.
        super().__init__(base_url=base_url, request_timeout_s=timeout)
    # ----------------------------------------

    def _step_payload(self, action: DIPGAction) -> dict:
        """
        Formats the `DIPGAction` object into a JSON-serializable dictionary.
        
        This dictionary becomes the body of the HTTP POST request sent to the
        server's `/step` endpoint.

        Args:
            action: The `DIPGAction` object containing the model's response.

        Returns:
            A dictionary to be sent as the JSON request body.
        """
        return {"llm_response": action.llm_response}

    def _parse_result(self, payload: dict) -> StepResult[DIPGObservation]:
        """
        Parses the JSON payload from the server into a `StepResult`,
        robustly handling inconsistencies and potential missing data.

        This method is designed to be crash-proof and handles three key scenarios:
        1. The single-nested 'observation' dictionary from the `/reset` endpoint.
        2. The double-nested 'observation' dictionary from the `/step` endpoint.
        3. A payload where the 'observation' key might be missing entirely.

        Args:
            payload: The raw dictionary parsed from the server's JSON response.

        Returns:
            A structured `StepResult` object.
        """
        # Safely get the top-level 'observation' object. It could be a dict or None.
        obs_data = payload.get("observation")

        # Check if the object is a dictionary and contains the nested 'observation' key.
        # This identifies the double-nested structure from the /step endpoint.
        if isinstance(obs_data, dict) and "observation" in obs_data:
            # If so, go one level deeper to get the actual data payload.
            actual_obs_data = obs_data.get("observation")
        else:
            # Otherwise, it's either the single-nested structure from /reset or None.
            actual_obs_data = obs_data if isinstance(obs_data, dict) else {}

        # To prevent crashes, ensure `actual_obs_data` is a dictionary before
        # we try to access keys from it. If it was None, it becomes an empty dict.
        if not isinstance(actual_obs_data, dict):
            actual_obs_data = {}
        
        # Construct the DIPGObservation object safely.
        # Using .get() with a default value ("") prevents a KeyError if 'context' or
        # 'question' are missing from the payload, ensuring the client never crashes.
        obs = DIPGObservation(
            context=actual_obs_data.get("context", ""),
            question=actual_obs_data.get("question", ""),
        )
        
        # Assemble and return the final, structured StepResult.
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    

    def _parse_state(self, payload: dict) -> DIPGState:
        """
        Parses the JSON payload from the server's `/state` endpoint into a `DIPGState` object.
        
        Args:
            payload: The raw dictionary parsed from the server's JSON response.
            
        Returns:
            A structured `DIPGState` object.
        """
        return DIPGState(**payload)
    
    # ==================================================================================
    # EVALUATION HELPER METHODS (NEW - Phase 5)
    # ==================================================================================
    
    def evaluate_model(
        self,
        responses: list[str],
        response_format: str = "auto",
        save_path: str | None = None
    ) -> dict:
        """
        Evaluate a batch of model responses using the evaluation service.
        
        This is a convenience method that calls the `/evaluate` endpoint
        to get aggregate metrics for a list of responses.
        
        Args:
            responses: List of model-generated responses to evaluate
            response_format: Expected format ("json", "xml", "yaml", "custom_tags", or "auto")
            save_path: Optional path to save detailed evaluation results
            
        Returns:
            Dictionary with evaluation results including mean_reward, median_reward, etc.
            
        Example:
            ```python
            client = DIPGSafetyEnv("http://localhost:8000")
            responses = ['{"analysis": "...", "proof": "...", "final": "..."}']
            results = client.evaluate_model(responses, response_format="json")
            print(f"Mean reward: {results['mean_reward']}")
            ```
        """
        payload = {
            "responses": responses,
            "format": response_format
        }
        
        if save_path:
            payload["save_path"] = save_path
        
        response = requests.post(
            f"{self._base}/evaluate",
            json=payload,
            timeout=self._timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_metrics_summary(self) -> dict:
        """
        Get summary of environment metrics and configuration.
        
        Returns:
            Dictionary with environment configuration
            
        Example:
            ```python
            client = DIPGSafetyEnv("http://localhost:8000")
            summary = client.get_metrics_summary()
            print(f"Current format: {summary['response_format']}")
            ```
        """
        response = requests.get(
            f"{self._base}/metrics/summary",
            timeout=self._timeout
        )
        response.raise_for_status()
        
        return response.json()