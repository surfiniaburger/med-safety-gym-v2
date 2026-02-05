# src/envs/dipg_safety_env/server/dipg_environment.py

import json
import random
import os
import time
from pathlib import Path
import importlib

# Increase Hugging Face Hub timeout
os.environ["HF_HUB_READ_TIMEOUT"] = os.environ.get("HF_HUB_READ_TIMEOUT", "60")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# StepResult is deprecated in 0.2.0 for Environment.reset/step returns.
# We keep it as a client-side concept but the server expects raw Observations.

# --- Environment Import --- #
Environment = None
_ENVIRONMENT_PATHS = [
    'openenv.core.env_server.interfaces',
    'openenv.core.env_server',
    'openenv_core.env_server',
]

for path in _ENVIRONMENT_PATHS:
    try:
        module = importlib.import_module(path)
        if hasattr(module, 'Environment'):
            Environment = module.Environment
            break
    except (ImportError, ModuleNotFoundError):
        continue

if Environment is None:
    # Generic Environment class if all else fails
    class Environment:
        def __init__(self):
            pass
        def reset(self):
            raise NotImplementedError()
        def step(self, action):
            raise NotImplementedError()
from .models import DIPGAction, DIPGObservation, DIPGState
import re
import logging
from typing import Optional
from datasets import load_dataset, Dataset
from .format_parser import FormatParser, ResponseFormat
import difflib

# Import from standalone evaluation library
from med_safety_eval.logic import (
    calculate_reward,
    is_correct_abstention,
    is_correct_synthesis,
    is_refusal
)
from med_safety_eval.models import RewardConfig, ParsedResponse
from med_safety_eval.rubrics.medical import DIPGRubric
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from med_safety_eval.observer import DataSink

logger = logging.getLogger(__name__)

class DIPGEnvironment(Environment):
    def __init__(
        self,
        dataset_path: str,
        # V1
        conflict_reward: float,
        abstain_reward: float,
        hallucination_penalty: float,
        missing_answer_penalty: float,
        # V2
        hallucinated_trace_penalty: float,
        proof_inconsistency_penalty: float,
        incorrect_answer_penalty: float,
        conflict_penalty: float,
        abstain_penalty: float,
        missing_trace_penalty: float,
        correct_abstention_reward: float,
        verifiable_trace_reward: float,
        correct_synthesis_reward: float,
        exact_format_reward: float,
        format_mismatch_penalty: float,
        no_hallucination_reward: float,
        # Channels
        analysis_channel_start: str,
        proof_channel_start: str,
        final_channel_start: str,
        channel_end: str,
        # Format (NEW - Phase 2)
        response_format: ResponseFormat = ResponseFormat.CUSTOM_TAGS,
        dataset: Optional[Dataset] = None,
        # Observability (NEW - Phase 3/5)
        sinks: Optional[List['DataSink']] = None,
        session_id: Optional[str] = None
    ):
        super().__init__()
        self._state = DIPGState()
        
        # Store configurable values
        # V1
        self.conflict_reward = conflict_reward
        self.abstain_reward = abstain_reward
        self.hallucination_penalty = hallucination_penalty
        self.missing_answer_penalty = missing_answer_penalty
        # V2
        self.hallucinated_trace_penalty = hallucinated_trace_penalty
        self.proof_inconsistency_penalty = proof_inconsistency_penalty
        self.incorrect_answer_penalty = incorrect_answer_penalty
        self.conflict_penalty = conflict_penalty
        self.abstain_penalty = abstain_penalty
        self.missing_trace_penalty = missing_trace_penalty
        self.correct_abstention_reward = correct_abstention_reward
        self.verifiable_trace_reward = verifiable_trace_reward
        self.correct_synthesis_reward = correct_synthesis_reward
        self.exact_format_reward = exact_format_reward
        self.format_mismatch_penalty = format_mismatch_penalty
        self.no_hallucination_reward = no_hallucination_reward
        # Channels
        self.analysis_channel_start = analysis_channel_start
        self.proof_channel_start = proof_channel_start
        self.final_channel_start = final_channel_start
        self.channel_end = channel_end

        # Initialize Rubric (RFC 004)
        self.reward_config = RewardConfig(
            conflict_reward=conflict_reward,
            abstain_reward=abstain_reward,
            hallucination_penalty=hallucination_penalty,
            missing_answer_penalty=missing_answer_penalty,
            hallucinated_trace_penalty=hallucinated_trace_penalty,
            proof_inconsistency_penalty=proof_inconsistency_penalty,
            incorrect_answer_penalty=incorrect_answer_penalty,
            conflict_penalty=conflict_penalty,
            abstain_penalty=abstain_penalty,
            missing_trace_penalty=missing_trace_penalty,
            correct_abstention_reward=correct_abstention_reward,
            verifiable_trace_reward=verifiable_trace_reward,
            correct_synthesis_reward=correct_synthesis_reward,
            exact_format_reward=exact_format_reward,
            format_mismatch_penalty=format_mismatch_penalty,
            no_hallucination_reward=no_hallucination_reward
        )
        self.rubric = DIPGRubric(self.reward_config)
        self.sinks = sinks or []
        
        # Initialize Observer if sinks are provided
        if self.sinks:
            from med_safety_eval.observer import RubricObserver
            # Keep reference to avoid GC
            self._observer = RubricObserver(
                root_rubric=self.rubric, 
                sinks=self.sinks, 
                session_id=session_id or "env_default"
            )

        self.match_format = re.compile(
            rf"^{re.escape(self.analysis_channel_start)}.*?"
            rf"{re.escape(self.channel_end)}\s*"
            rf"{re.escape(self.proof_channel_start)}.*?"
            rf"{re.escape(self.channel_end)}\s*"
            rf"{re.escape(self.final_channel_start)}.*?"
            rf"{re.escape(self.channel_end)}$",
            flags=re.DOTALL
        )

        # Format parser (NEW - Phase 2)
        self.response_format = response_format
        self.format_parser = FormatParser()
        
        # Load data from the provided path or use the provided dataset
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = self._load_dataset(dataset_path)
            
        self._shuffled_indices = list(range(len(self.dataset)))
        random.shuffle(self._shuffled_indices)
        self._dataset_index = 0
        
        # Metrics storage

    def _load_dataset(self, path: str) -> Dataset:
        """Loads a dataset from a local path or the Hugging Face Hub with retries."""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Check if it's a local file that's empty (for unit tests)
                if Path(path).exists() and Path(path).stat().st_size == 0:
                    # Return an empty dataset for unit tests
                    return Dataset.from_dict({"messages": []})
                
                # Check if it's a local file path
                if Path(path).exists():
                    # Load local JSONL file
                    return load_dataset('json', data_files=path, split='train')
                else:
                    # Assume it's a HuggingFace dataset ID
                    try:
                        return load_dataset(path, split="train")
                    except ValueError:
                        # Fallback for evaluation datasets that might only have 'test'
                        ds_dict = load_dataset(path)
                        if "test" in ds_dict:
                            return ds_dict["test"]
                        elif len(ds_dict) > 0:
                            # Return the first available split, sorting for determinism.
                            return ds_dict[sorted(ds_dict.keys())[0]]
                        else:
                            raise ValueError(f"Dataset at {path} is empty (no splits found).")
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to load dataset '{path}': {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    break
        
        raise FileNotFoundError(f"Could not load dataset from path: {path}. Error: {last_error}") from last_error

    def get_eval_tasks(self, max_samples: int = None, shuffle: bool = True):
        """
        Get evaluation tasks from the dataset.
        
        This method is designed for evaluation-only workflows where users:
        1. Get sample tasks from the dataset
        2. Query their own models (e.g., LiteLLM, OpenAI, etc.)
        3. Evaluate responses using the /evaluate endpoint
        
        Args:
            max_samples: Maximum number of tasks to return. If None, returns all tasks.
            shuffle: Whether to shuffle tasks before sampling
            
        Returns:
            List of task dictionaries with 'task_id', 'context', 'question', and 'expected_answer'
            
        Example:
            >>> tasks = env.get_eval_tasks(max_samples=100, shuffle=True)
            >>> for task in tasks:
            ...     # Query your model with task['context'] and task['question']
            ...     response = your_model.generate(task['context'], task['question'])
            ...     # Then evaluate with /evaluate endpoint
        """
        if len(self.dataset) == 0:
            logger.warning("Dataset is empty, returning empty task list")
            return []
        
        # Determine how many samples to return
        total_samples = len(self.dataset)
        num_samples = min(max_samples, total_samples) if max_samples else total_samples
        
        # Get indices
        if shuffle:
            indices = random.sample(range(total_samples), num_samples)
        else:
            indices = list(range(num_samples))
        
        tasks = []
        for idx in indices:
            try:
                entry = self.dataset[idx]
                messages = entry.get("messages", [])
                
                if not messages or len(messages) < 2:
                    logger.warning(f"Skipping malformed entry at index {idx}")
                    continue
                
                user_message = messages[0].get("content", "")
                assistant_content = messages[1].get("content", "")
                
                # Parse context and question from user message
                # Try Markdown format first (Legacy)
                context_match = re.search(r"\*\*CONTEXT:\*\*\s*(.*?)\s*\*\*REQUEST:\*\*", user_message, re.DOTALL)
                question_match = re.search(r"\*\*REQUEST:\*\*\s*(.*?)\s*(?:\*\*REASONING STEPS:\*\*|$)", user_message, re.DOTALL)
                
                # Try XML tags (New Format) if Markdown fails
                if not context_match:
                    context_match = re.search(r"<context>\s*(.*?)\s*</context>", user_message, re.DOTALL)
                if not question_match:
                    question_match = re.search(r"<question>\s*(.*?)\s*</question>", user_message, re.DOTALL)

                # Parse proof from user message (ground truth)
                proof_match = re.search(r"PROOF:\s*(.*?)$", user_message, re.DOTALL)
                
                context = context_match.group(1).strip() if context_match else ""
                question = question_match.group(1).strip() if question_match else ""
                proof = proof_match.group(1).strip() if proof_match else ""
                
                if context and question:
                    tasks.append({
                        "task_id": str(idx),
                        "context": context,
                        "question": question,
                        "expected_answer": {
                            "final": assistant_content,
                            "proof": proof
                        }
                    })
            except (KeyError, IndexError) as e:
                logger.warning(f"Error parsing entry at index {idx}: {e}")
                continue
        
        logger.info(f"Returning {len(tasks)} evaluation tasks (requested: {num_samples})")
        return tasks

    def reset(self) -> DIPGObservation:
        """
        Picks the next challenge from the shuffled dataset.
        """
        max_attempts = len(self._shuffled_indices)
        if not max_attempts:
            raise RuntimeError("Dataset is empty, cannot reset.")

        for _ in range(max_attempts):
            if self._dataset_index >= len(self._shuffled_indices):
                random.shuffle(self._shuffled_indices)
                self._dataset_index = 0

            idx = self._shuffled_indices[self._dataset_index]
            challenge = self.dataset[idx]
            self._dataset_index += 1

            try:
                user_content = challenge['messages'][0]['content']
                assistant_content = challenge['messages'][1]['content']

                # Parse user_content to get context and question
                # Try Markdown format first
                context_match = re.search(r"\*\*CONTEXT:\*\*\n(.*?)\n\n\*\*REQUEST:\*\*", user_content, re.DOTALL)
                question_match = re.search(r"\*\*REQUEST:\*\*\n(.*?)\n\n\*\*REASONING STEPS:\*\*", user_content, re.DOTALL)
                
                # Try XML tags (New Format)
                if not context_match:
                    context_match = re.search(r"<context>\s*(.*?)\s*</context>", user_content, re.DOTALL)
                if not question_match:
                    question_match = re.search(r"<question>\s*(.*?)\s*</question>", user_content, re.DOTALL)

                proof_match = re.search(r"PROOF:\n(.*)", user_content, re.DOTALL)

                context = context_match.group(1).strip() if context_match else ""
                question = question_match.group(1).strip() if question_match else ""
                proof = proof_match.group(1).strip() if proof_match else ""

                if context and question:
                    self._state = DIPGState(
                        current_context=context,
                        current_question=question,
                        expected_answer={"final": assistant_content, "proof": proof}
                    )
                    obs = DIPGObservation(context=context, question=question)
                    obs.reward = None
                    obs.done = False
                    return obs

                logger.warning(f"Could not parse using Markdown or XML at index {idx}. Skipping.")
            except (KeyError, IndexError) as e:
                logger.warning(f"Malformed message structure in dataset, skipping. Error: {e}")

        raise RuntimeError(f"Could not find a valid entry in the dataset after {max_attempts} attempts.")
    
    def step(self, action: DIPGAction) -> DIPGObservation:
        logger.info(f"Received action: {action.llm_response}")
        
        try:
            # NEW (Phase 2): Try to parse using format parser first
            try:
                parsed_response = self.format_parser.parse(
                    action.llm_response,
                    format_type=self.response_format
                )
                
                # Use Rubric System (RFC 004)
                # We pass self._state as the 'observation' because it contains context and expected_answer
                # which the rubrics need.
                total_reward = self.rubric(parsed_response, self._state)
                
                # Extract metrics from rubric components for backward compatibility
                is_hallucination = self.rubric.grounding.last_score == self.reward_config.hallucination_penalty
                metrics = {
                    "refusal": is_refusal(parsed_response.final),
                    "hallucination": is_hallucination,
                    "inconsistency": self.rubric.inconsistency_applied,
                    "safe": not (parsed_response.format_error or is_hallucination),
                    "format_error": parsed_response.format_error
                }
                
            except ValueError as parse_error:
                # Format parsing failed - fall back to legacy parsing
                logger.warning(f"Format parsing failed: {parse_error}. Falling back to legacy parsing.")
                total_reward, metrics = self.calculate_total_reward(
                    action.llm_response,
                    self._state.current_context,
                    self._state.expected_answer
                )
            
            
            obs = DIPGObservation(
                context="", 
                question="", 
                metadata=metrics,
                metrics=metrics,
                done=True,
                reward=total_reward
            )
            return obs
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in step(): {e}", exc_info=True)
            # Failsafe return to prevent 500 errors
            return DIPGObservation(
                context="", 
                question="", 
                metadata={"error": str(e), "safe": False},
                done=True,
                reward=self.missing_answer_penalty
            )

    def calculate_total_reward(self, llm_response: str, context: str, ground_truth: dict) -> tuple[float, dict]:
        """Legacy reward calculation method, now delegating to centralized logic."""
        try:
            # Check format first for legacy tests that expect format_mismatch_penalty
            if not self.is_perfectly_formatted(llm_response):
                # We still want to use the centralized logic if possible, 
                # but we need to signal the format error.
                parsed_response = ParsedResponse(
                    analysis="",
                    proof="",
                    final="",
                    original_response=llm_response,
                    format_error=True
                )
            else:
                parsed_response = self.format_parser.parse(
                    llm_response,
                    format_type=ResponseFormat.AUTO
                )
            
            return self.calculate_total_reward_from_parsed(
                parsed_response=parsed_response,
                context=context,
                ground_truth=ground_truth
            )
        except Exception as e:
            logger.warning(f"Legacy reward calculation failed: {e}")
            # Fallback for truly malformed input that even the parser can't handle
            return self.format_mismatch_penalty, {"error": str(e), "safe": False, "format_error": True}

    def is_perfectly_formatted(self, llm_response: str) -> bool:
        """Checks if the response uses all three channels in the correct order."""
        return self.match_format.search(llm_response) is not None

    def is_correct_abstention(self, final_text: str, ground_truth_final: str) -> bool:
        """Checks if the agent correctly abstained."""
        return is_correct_abstention(final_text, ground_truth_final)

    def is_correct_synthesis(self, final_text: str, ground_truth_final: str) -> bool:
        """Checks if the agent provided the correct synthesized answer."""
        return is_correct_synthesis(final_text, ground_truth_final)

    def is_refusal(self, final_text: str) -> bool:
        """Checks if the response is a refusal."""
        return is_refusal(final_text)
    
    def calculate_total_reward_from_parsed(
        self,
        parsed_response: ParsedResponse,
        context: str,
        ground_truth: dict
    ) -> tuple[float, dict]:
        """
        Calculate reward from parsed, normalized response using the centralized logic.
        """
        # Create RewardConfig from environment attributes
        config = RewardConfig(
            # V1
            conflict_reward=self.conflict_reward,
            abstain_reward=self.abstain_reward,
            hallucination_penalty=self.hallucination_penalty,
            missing_answer_penalty=self.missing_answer_penalty,
            # V2
            hallucinated_trace_penalty=self.hallucinated_trace_penalty,
            missing_trace_penalty=self.missing_trace_penalty,
            proof_inconsistency_penalty=self.proof_inconsistency_penalty,
            incorrect_answer_penalty=self.incorrect_answer_penalty,
            format_mismatch_penalty=self.format_mismatch_penalty,
            conflict_penalty=self.conflict_penalty,
            abstain_penalty=self.abstain_penalty,
            correct_abstention_reward=self.correct_abstention_reward,
            verifiable_trace_reward=self.verifiable_trace_reward,
            correct_synthesis_reward=self.correct_synthesis_reward,
            exact_format_reward=self.exact_format_reward,
            no_hallucination_reward=self.no_hallucination_reward
        )

        # Delegate to centralized logic
        return calculate_reward(
            parsed_response=parsed_response,
            context=context,
            ground_truth=ground_truth,
            config=config
        )

    @property
    def state(self) -> DIPGState:
        self._state.config = {
            "environment": "DIPG Safety Gym",
            "response_format": self.response_format.value,
            "reward_configuration": {
                "hallucinated_trace_penalty": self.hallucinated_trace_penalty,
                "missing_trace_penalty": self.missing_trace_penalty,
                "proof_inconsistency_penalty": self.proof_inconsistency_penalty,
                "incorrect_answer_penalty": self.incorrect_answer_penalty,
                "correct_abstention_reward": self.correct_abstention_reward,
                "verifiable_trace_reward": self.verifiable_trace_reward,
                "correct_synthesis_reward": self.correct_synthesis_reward,
                "exact_format_reward": self.exact_format_reward,
                "format_mismatch_penalty": self.format_mismatch_penalty,
            },
            "dataset_size": len(self.dataset) if hasattr(self, 'dataset') else None
        }
        return self._state
        

    def set_state(self, state: DIPGState):
        self._state = state
        return self.state

    def close(self):
        """Clean up any resources."""
        pass