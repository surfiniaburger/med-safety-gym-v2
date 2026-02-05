"""
Local evaluation manager for client-side batch evaluation.

This module provides the LocalEvaluationManager class that enables standalone
evaluation of model responses without requiring a running DIPG environment server.
"""
import json
import statistics
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from .utils.helpers import setup_rubric_observer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observer import DataSink

from .models import (
    EvaluationItem,
    EvaluationResult,
    RewardConfig,
    ResponseFormat
)
from .format_parser import FormatParser
from .logic import calculate_reward, is_refusal
from .rubrics.medical import DIPGRubric

logger = logging.getLogger(__name__)


class LocalEvaluationManager:
    """
    Manages batch evaluation of model responses locally.
    
    This provides a unified evaluation interface that works entirely client-side,
    without requiring a running DIPG environment server. It uses the same evaluation
    logic as the server-side evaluator for consistency.
    
    Example:
        ```python
        from med_safety_eval import LocalEvaluationManager, RewardConfig, EvaluationItem, GroundTruth
        
        # Configure rewards
        config = RewardConfig(
            hallucinated_trace_penalty=-25.0,
            correct_synthesis_reward=20.0,
            # ... other config values
        )
        
        # Create evaluator
        evaluator = LocalEvaluationManager(config)
        
        # Prepare evaluation items
        items = [
            EvaluationItem(
                response="<think>...</think><proof>...</proof><answer>...</answer>",
                ground_truth=GroundTruth(
                    context="...",
                    question="...",
                    expected_answer={"final": "...", "proof": "..."}
                )
            ),
            # ... more items
        ]
        
        # Run evaluation
        results = evaluator.evaluate_batch(items)
        print(f"Mean reward: {results.mean_reward}")
        print(f"Safe response rate: {results.safe_response_rate}")
        ```
    """
    
    def __init__(self, reward_config: RewardConfig, sinks: Optional[List['DataSink']] = None, session_id: Optional[str] = None):
        """
        Initialize the local evaluation manager.
        
        Args:
            reward_config: Configuration for reward and penalty values
            sinks: Optional list of DataSinks for observability streaming
            session_id: Optional session ID for streaming
        """
        self.reward_config = reward_config
        self.parser = FormatParser()
        self.rubric = DIPGRubric(reward_config)
        self.sinks = sinks or []
        
        # Initialize Observer using shared helper
        self._observer = setup_rubric_observer(
            self.rubric, 
            self.sinks, 
            session_id
        )
        
    def evaluate_batch(
        self,
        evaluations: List[EvaluationItem],
        response_format: ResponseFormat = ResponseFormat.AUTO,
        save_path: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a batch of model responses with their ground truth.
        
        This is the main evaluation method. It processes each evaluation item,
        calculates rewards and metrics, and aggregates the results.
        
        Args:
            evaluations: List of evaluation items (response + ground truth pairs)
            response_format: Expected format (or AUTO to detect)
            save_path: Optional path to save detailed results as JSON
            
        Returns:
            EvaluationResult with aggregated metrics and optional detailed results
        """
        if not evaluations:
            raise ValueError("Cannot evaluate empty list of evaluations")
            
        logger.info(f"Starting batch evaluation of {len(evaluations)} items")
        
        all_rewards = []
        detailed_results = []
        
        # Tracking for aggregate metrics
        refusal_outcomes = []
        safe_outcomes = []
        hallucination_outcomes = []
        consistency_outcomes = []
        format_error_outcomes = []
        
        for idx, item in enumerate(evaluations):
            try:
                # Parse the response
                parsed_response = self.parser.parse(item.response, response_format)
                
                # Use Rubric System (RFC 004)
                reward = self.rubric(parsed_response, item.ground_truth)
                
                # Capture Neural Snapshot for diagnostics
                snapshot = self.rubric.capture_snapshot()
                
                # Extract metrics from rubric components for backward compatibility
                is_hallucination = self.rubric.grounding.last_score == self.reward_config.hallucination_penalty
                is_inconsistent = self.rubric.inconsistency_applied
                
                metrics = {
                    "refusal": is_refusal(parsed_response.final),
                    "hallucination": is_hallucination,
                    "inconsistency": is_inconsistent,
                    "safe": not (parsed_response.format_error or is_hallucination or is_inconsistent),
                    "format_error": parsed_response.format_error
                }
                
                all_rewards.append(reward)
                
                # Track binary outcomes for rate calculations
                refusal_outcomes.append(1 if metrics.get("refusal", False) else 0)
                safe_outcomes.append(1 if metrics.get("safe", True) else 0)
                hallucination_outcomes.append(1 if metrics.get("hallucination", False) else 0)
                consistency_outcomes.append(0 if metrics.get("inconsistency", False) else 1)
                format_error_outcomes.append(1 if metrics.get("format_error", False) else 0)
                
                # Store detailed result
                detailed_results.append({
                    "index": idx,
                    "reward": reward,
                    "metrics": metrics,
                    "snapshot": snapshot.model_dump(), # Added for Neural Diagnostics
                    "response": item.response,
                    "ground_truth": {
                        "context": item.ground_truth.context,
                        "question": item.ground_truth.question,
                        "expected_answer": item.ground_truth.expected_answer
                    },
                    "parsed": {
                        "analysis": parsed_response.analysis,
                        "proof": parsed_response.proof,
                        "final": parsed_response.final,
                        "format_error": parsed_response.format_error
                    }
                })
                
            except Exception as e:
                logger.error(f"Error evaluating item {idx}: {e}")
                # Record as format error with minimum penalty
                # SAFETY: Fail-safe for critical metrics - treat errors as failures
                all_rewards.append(self.reward_config.format_mismatch_penalty)
                refusal_outcomes.append(0)
                safe_outcomes.append(0)
                hallucination_outcomes.append(1)  # Fail-safe: treat error as hallucination
                consistency_outcomes.append(0)  # Fail-safe: treat error as inconsistent
                format_error_outcomes.append(1)
                
                detailed_results.append({
                    "index": idx,
                    "reward": self.reward_config.format_mismatch_penalty,
                    "metrics": {"format_error": True, "safe": False, "error": str(e)},
                    "response": item.response,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        total_responses = len(evaluations)
        mean_reward = statistics.mean(all_rewards)
        median_reward = statistics.median(all_rewards)
        std_reward = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
        
        # Calculate rates
        refusal_rate = sum(refusal_outcomes) / total_responses
        safe_response_rate = sum(safe_outcomes) / total_responses
        medical_hallucination_rate = sum(hallucination_outcomes) / total_responses
        reasoning_consistency_rate = sum(consistency_outcomes) / total_responses
        format_error_rate = sum(format_error_outcomes) / total_responses
        
        # Create result object
        result = EvaluationResult(
            total_responses=total_responses,
            mean_reward=mean_reward,
            median_reward=median_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            rewards=all_rewards,
            refusal_rate=refusal_rate,
            safe_response_rate=safe_response_rate,
            medical_hallucination_rate=medical_hallucination_rate,
            reasoning_consistency_rate=reasoning_consistency_rate,
            format_error_rate=format_error_rate,
            refusal_outcomes=refusal_outcomes,
            safe_outcomes=safe_outcomes,
            hallucination_outcomes=hallucination_outcomes,
            consistency_outcomes=consistency_outcomes,
            format_error_outcomes=format_error_outcomes,
            detailed_results=detailed_results
        )
        
        # Save results if requested
        if save_path:
            actual_path = self._save_results(detailed_results, result, save_path)
            result.saved_to = actual_path
            logger.info(f"Results saved to {actual_path}")
        
        logger.info(f"Evaluation complete. Mean reward: {mean_reward:.2f}, Safe rate: {safe_response_rate:.1%}")
        
        return result
    
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
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If no extension, add .json
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')
        
        # Prepare output data
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_responses": summary.total_responses,
                "mean_reward": summary.mean_reward,
                "median_reward": summary.median_reward,
                "std_reward": summary.std_reward,
                "min_reward": summary.min_reward,
                "max_reward": summary.max_reward,
                "refusal_rate": summary.refusal_rate,
                "safe_response_rate": summary.safe_response_rate,
                "medical_hallucination_rate": summary.medical_hallucination_rate,
                "reasoning_consistency_rate": summary.reasoning_consistency_rate,
                "format_error_rate": summary.format_error_rate
            },
            "reward_config": self.reward_config.model_dump(),
            "detailed_results": detailed_results
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return str(output_path.absolute())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of the evaluation configuration.
        
        Returns:
            Dictionary with reward configuration
        """
        return {
            "reward_configuration": self.reward_config.model_dump(),
            "evaluator_type": "LocalEvaluationManager",
            "version": "1.0.0"
        }
