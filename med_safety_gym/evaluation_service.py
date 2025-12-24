"""
Evaluation service for DIPG Safety Gym.

Provides batch evaluation capabilities for models, supporting multiple formats
and consistent metrics across different training methods (SFT, GRPO, etc.).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
import logging
from pathlib import Path
from datetime import datetime
import statistics

from .dipg_environment import DIPGEnvironment
from .format_parser import ResponseFormat
from .models import DIPGAction, DIPGState

logger = logging.getLogger(__name__)


class GroundTruth(BaseModel):
    """Ground truth data for evaluation"""
    context: str = Field(..., description="Context/background information")
    question: str = Field(..., description="The question being asked")
    expected_answer: Dict[str, str] = Field(
        ...,
        description="Expected answer with 'final' and 'proof' fields"
    )


class EvaluationItem(BaseModel):
    """Single evaluation item with response and ground truth"""
    response: str = Field(..., description="Model response to evaluate")
    ground_truth: Optional[GroundTruth] = Field(
        default=None,
        description="Ground truth for this specific response (optional)"
    )


class EvaluationRequest(BaseModel):
    """
    Request model for batch evaluation.
    
    Supports two modes:
    1. Simple mode: Just responses (uses server's dataset for ground truth)
    2. Stateless mode: Responses + ground truth (self-contained, cloud-native)
    """
    # Simple mode (backward compatible)
    responses: Optional[List[str]] = Field(
        default=None,
        description="List of model responses (simple mode)"
    )
    
    # Stateless mode (recommended)
    evaluations: Optional[List[EvaluationItem]] = Field(
        default=None,
        description="List of evaluation items with ground truth (stateless mode)"
    )
    
    format: ResponseFormat = Field(
        default=ResponseFormat.AUTO,
        description="Expected format of responses (auto-detect by default)"
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Optional path to save evaluation results"
    )
    
    def model_post_init(self, __context):
        """Validate that at least one mode is provided"""
        if self.responses is None and self.evaluations is None:
            raise ValueError("Must provide either 'responses' or 'evaluations'")
        if self.responses is not None and self.evaluations is not None:
            raise ValueError("Cannot provide both 'responses' and 'evaluations'")


class EvaluationResult(BaseModel):
    """Result model for batch evaluation"""
    total_responses: int
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    rewards: List[float]
    
    # Advanced Safety Metrics (aggregate rates)
    refusal_rate: float = 0.0
    safe_response_rate: float = 0.0
    medical_hallucination_rate: float = 0.0
    reasoning_consistency_rate: float = 0.0
    
    # Per-sample metrics for CI calculation (list of 0/1 outcomes)
    refusal_outcomes: Optional[List[int]] = None
    safe_outcomes: Optional[List[int]] = None
    hallucination_outcomes: Optional[List[int]] = None
    consistency_outcomes: Optional[List[int]] = None
    
    # Detailed results (optional per-request)
    detailed_results: Optional[List[Dict[str, Any]]] = None
    
    saved_to: Optional[str] = None


class EvaluationManager:
    """
    Manages batch evaluation of model responses.
    
    This provides a unified evaluation interface that works across
    different training methods and response formats.
    """
    
    def __init__(self, environment: DIPGEnvironment):
        """
        Initialize the evaluation manager.
        
        Args:
            environment: The DIPG environment instance to use for evaluation
        """
        self.environment = environment
    
    def evaluate_batch(
        self,
        responses: List[str],
        response_format: ResponseFormat = ResponseFormat.AUTO,
        save_path: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a batch of model responses.
        
        Args:
            responses: List of model-generated responses
            response_format: Expected format (or AUTO to detect)
            save_path: Optional path to save detailed results
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(responses)} responses")
        
        # Store original format setting
        original_format = self.environment.response_format
        
        try:
            # Temporarily set the format for this evaluation
            self.environment.response_format = response_format
            
            # Evaluate each response
            rewards = []
            detailed_results = []
            
            # Metric Counters
            refusal_count = 0
            safe_count = 0
            hallucination_count = 0
            consistency_count = 0
            
            # Per-sample outcomes for CI calculation
            refusal_outcomes = []
            safe_outcomes = []
            hallucination_outcomes = []
            consistency_outcomes = []
            
            for idx, response in enumerate(responses):
                observation = None
                try:
                    # Reset environment to get a new challenge
                    observation = self.environment.reset()
                    
                    # Create action with the response
                    action = DIPGAction(llm_response=response)
                    
                    # Get reward
                    step_result = self.environment.step(action)
                    reward = step_result.reward
                    
                    # Collect Metrics
                    metrics = self.environment.last_metrics
                    
                    # Track aggregate counts and per-sample outcomes
                    is_refusal = 1 if metrics.get("refusal") else 0
                    is_safe = 1 if metrics.get("safe") else 0
                    is_hallucination = 1 if metrics.get("hallucination") else 0
                    # Consistent = safe AND not refusal (refusals have no reasoning to verify)
                    is_consistent = 1 if (metrics.get("safe") and not metrics.get("refusal")) else 0
                    
                    refusal_count += is_refusal
                    safe_count += is_safe
                    hallucination_count += is_hallucination
                    consistency_count += is_consistent
                    
                    refusal_outcomes.append(is_refusal)
                    safe_outcomes.append(is_safe)
                    hallucination_outcomes.append(is_hallucination)
                    consistency_outcomes.append(is_consistent)

                    rewards.append(reward)
                    detailed_results.append({
                        "index": idx,
                        "response": response,
                        "reward": reward,
                        "metrics": metrics,
                        "context": observation.context,
                        "question": observation.question
                    })
                    
                except Exception as e:
                    logger.error(f"Error evaluating response {idx}: {e}")
                    # Assign minimum penalty for failed evaluations
                    rewards.append(self.environment.missing_answer_penalty)
                    
                    error_entry = {
                        "index": idx,
                        "response": response,
                        "reward": self.environment.missing_answer_penalty,
                        "error": str(e)
                    }
                    if observation:
                        error_entry["context"] = observation.context
                        error_entry["question"] = observation.question
                    detailed_results.append(error_entry)
            
            # Calculate aggregate metrics
            total = len(responses)
            result = EvaluationResult(
                total_responses=total,
                mean_reward=statistics.mean(rewards) if rewards else 0.0,
                median_reward=statistics.median(rewards) if rewards else 0.0,
                std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                min_reward=min(rewards) if rewards else 0.0,
                max_reward=max(rewards) if rewards else 0.0,
                rewards=rewards,
                # Aggregate metrics
                refusal_rate=refusal_count / total if total > 0 else 0.0,
                safe_response_rate=safe_count / total if total > 0 else 0.0,
                medical_hallucination_rate=hallucination_count / total if total > 0 else 0.0,
                reasoning_consistency_rate=consistency_count / total if total > 0 else 0.0,
                # Per-sample outcomes for CI calculation
                refusal_outcomes=refusal_outcomes,
                safe_outcomes=safe_outcomes,
                hallucination_outcomes=hallucination_outcomes,
                consistency_outcomes=consistency_outcomes,
                detailed_results=detailed_results
            )
            
            # Save detailed results if requested
            if save_path:
                saved_path = self._save_results(detailed_results, result, save_path)
                result.saved_to = saved_path
                logger.info(f"Saved detailed results to {saved_path}")
            
            logger.info(f"Evaluation complete. Mean reward: {result.mean_reward:.2f}")
            return result
            
        finally:
            # Restore original format setting
            self.environment.response_format = original_format
    
    def evaluate_with_ground_truth(
        self,
        evaluations: List[EvaluationItem],
        response_format: ResponseFormat = ResponseFormat.AUTO,
        save_path: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate responses with provided ground truth (stateless mode).
        
        This is the recommended cloud-native approach following AWS SageMaker
        and Google Vertex AI best practices for stateless batch evaluation.
        
        Args:
            evaluations: List of evaluation items with responses and ground truth
            response_format: Expected format (or AUTO to detect)
            save_path: Optional path to save detailed results
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(evaluations)} items with ground truth (stateless mode)")
        
        # Store original format setting
        original_format = self.environment.response_format
        
        try:
            # Temporarily set the format for this evaluation
            self.environment.response_format = response_format
            
            # Evaluate each item
            rewards = []
            detailed_results = []
            
            # Metric Counters
            refusal_count = 0
            safe_count = 0
            hallucination_count = 0
            consistency_count = 0
            
            # Per-sample outcomes for CI calculation
            refusal_outcomes = []
            safe_outcomes = []
            hallucination_outcomes = []
            consistency_outcomes = []
            
            for idx, item in enumerate(evaluations):
                try:
                    # Set up the environment state with the provided ground truth
                    self.environment.set_state(DIPGState(
                        current_context=item.ground_truth.context,
                        current_question=item.ground_truth.question,
                        expected_answer=item.ground_truth.expected_answer
                    ))
                    
                    # Create action with the response
                    action = DIPGAction(llm_response=item.response)
                    
                    # Get reward using the provided ground truth
                    step_result = self.environment.step(action)
                    reward = step_result.reward
                    
                    # Collect Metrics
                    metrics = self.environment.last_metrics
                    
                    # Track aggregate counts and per-sample outcomes
                    is_refusal = 1 if metrics.get("refusal") else 0
                    is_safe = 1 if metrics.get("safe") else 0
                    is_hallucination = 1 if metrics.get("hallucination") else 0
                    is_consistent = 1 if (metrics.get("safe") and not metrics.get("refusal")) else 0
                    
                    refusal_count += is_refusal
                    safe_count += is_safe
                    hallucination_count += is_hallucination
                    consistency_count += is_consistent
                    
                    refusal_outcomes.append(is_refusal)
                    safe_outcomes.append(is_safe)
                    hallucination_outcomes.append(is_hallucination)
                    consistency_outcomes.append(is_consistent)
                    
                    rewards.append(reward)
                    detailed_results.append({
                        "index": idx,
                        "response": item.response,
                        "reward": reward,
                        "metrics": metrics,
                        "context": item.ground_truth.context,
                        "question": item.ground_truth.question,
                        "expected_answer": item.ground_truth.expected_answer
                    })
                    
                except Exception as e:
                    logger.error(f"Error evaluating item {idx}: {e}")
                    # Assign minimum penalty for failed evaluations
                    rewards.append(self.environment.missing_answer_penalty)
                    
                    # Add zero outcomes for failed evaluations
                    refusal_outcomes.append(0)
                    safe_outcomes.append(0)
                    hallucination_outcomes.append(0)
                    consistency_outcomes.append(0)
                    
                    error_entry = {
                        "index": idx,
                        "response": item.response,
                        "reward": self.environment.missing_answer_penalty,
                        "error": str(e)
                    }
                    if item.ground_truth:
                        error_entry["context"] = item.ground_truth.context
                        error_entry["question"] = item.ground_truth.question
                    detailed_results.append(error_entry)
            
            # Calculate aggregate metrics
            total = len(evaluations)
            result = EvaluationResult(
                total_responses=total,
                mean_reward=statistics.mean(rewards) if rewards else 0.0,
                median_reward=statistics.median(rewards) if rewards else 0.0,
                std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                min_reward=min(rewards) if rewards else 0.0,
                max_reward=max(rewards) if rewards else 0.0,
                rewards=rewards,
                # Aggregate metrics
                refusal_rate=refusal_count / total if total > 0 else 0.0,
                safe_response_rate=safe_count / total if total > 0 else 0.0,
                medical_hallucination_rate=hallucination_count / total if total > 0 else 0.0,
                reasoning_consistency_rate=consistency_count / total if total > 0 else 0.0,
                # Per-sample outcomes for CI calculation
                refusal_outcomes=refusal_outcomes,
                safe_outcomes=safe_outcomes,
                hallucination_outcomes=hallucination_outcomes,
                consistency_outcomes=consistency_outcomes,
                detailed_results=detailed_results
            )
            
            # Save detailed results if requested
            if save_path:
                saved_path = self._save_results(detailed_results, result, save_path)
                result.saved_to = saved_path
                logger.info(f"Saved detailed results to {saved_path}")
            
            logger.info(f"Stateless evaluation complete. Mean reward: {result.mean_reward:.2f}")
            return result
            
        finally:
            # Restore original format setting
            self.environment.response_format = original_format
    
    def _save_results(
        self,
        detailed_results: List[Dict[str, Any]],
        summary: EvaluationResult,
        save_path: str
    ) -> str:
        """
        Save detailed evaluation results to a file.
        
        Args:
            detailed_results: List of per-response results
            summary: Aggregate metrics
            save_path: Path to save results
            
        Returns:
            Actual path where results were saved
        """
        # Security check: Prevent path traversal
        # 1. Must not be absolute
        if Path(save_path).is_absolute():
            raise ValueError(f"Invalid save_path: Absolute paths are not allowed ({save_path})")
            
        # 2. Must resolve to a path inside the current working directory
        safe_base = Path.cwd().resolve()
        requested_path = (safe_base / save_path).resolve()
        
        if not str(requested_path).startswith(str(safe_base)):
            raise ValueError(f"Invalid save_path: Path traversal detected ({save_path})")
            
        output_path = requested_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary.model_dump(exclude={"saved_to"}),
            "detailed_results": detailed_results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return str(output_path)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of environment metrics and configuration.
        
        Returns:
            Dictionary with environment configuration and statistics
        """
        return {
            "environment": "DIPG Safety Gym",
            "response_format": self.environment.response_format.value,
            "reward_configuration": {
                "hallucinated_trace_penalty": self.environment.hallucinated_trace_penalty,
                "missing_trace_penalty": self.environment.missing_trace_penalty,
                "proof_inconsistency_penalty": self.environment.proof_inconsistency_penalty,
                "incorrect_answer_penalty": self.environment.incorrect_answer_penalty,
                "correct_abstention_reward": self.environment.correct_abstention_reward,
                "verifiable_trace_reward": self.environment.verifiable_trace_reward,
                "correct_synthesis_reward": self.environment.correct_synthesis_reward,
                "exact_format_reward": self.environment.exact_format_reward,
                "format_mismatch_penalty": self.environment.format_mismatch_penalty,
            },
            "dataset_size": len(self.environment.dataset) if hasattr(self.environment, 'dataset') else None
        }
