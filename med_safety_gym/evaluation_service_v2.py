"""
Evaluation service for DIPG Safety Gym (V2 - using standalone med_safety_eval library).

This version delegates evaluation logic to the standalone med_safety_eval package,
enabling consistent evaluation across server-side and client-side use cases.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

# Import from standalone evaluation library
from med_safety_eval import (
    LocalEvaluationManager,
    EvaluationItem as EvalItem,
    GroundTruth as GT,
    RewardConfig,
    ResponseFormat
)

from .dipg_environment import DIPGEnvironment

logger = logging.getLogger(__name__)


# Re-export models for backward compatibility
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


class EvaluationManager:
    """
    Manages batch evaluation of model responses using the standalone med_safety_eval library.
    
    This V2 implementation delegates all evaluation logic to LocalEvaluationManager,
    ensuring consistency between server-side and client-side evaluation.
    """
    
    def __init__(self, environment: DIPGEnvironment, sinks: Optional[List[Any]] = None):
        """
        Initialize the evaluation manager.
        
        Args:
            environment: The DIPG environment instance (used for reward config and dataset access)
        """
        self.environment = environment
        
        # Extract reward configuration from environment
        self.reward_config = RewardConfig(
            hallucinated_trace_penalty=environment.hallucinated_trace_penalty,
            missing_trace_penalty=environment.missing_trace_penalty,
            proof_inconsistency_penalty=environment.proof_inconsistency_penalty,
            incorrect_answer_penalty=environment.incorrect_answer_penalty,
            format_mismatch_penalty=environment.format_mismatch_penalty,
            correct_abstention_reward=environment.correct_abstention_reward,
            verifiable_trace_reward=environment.verifiable_trace_reward,
            correct_synthesis_reward=environment.correct_synthesis_reward,
            exact_format_reward=environment.exact_format_reward,
            no_hallucination_reward=environment.no_hallucination_reward
        )
        
        # Create local evaluator
        self.local_evaluator = LocalEvaluationManager(
            reward_config=self.reward_config,
            sinks=sinks,
            session_id="eval_service_v2" # Default session ID
        )
    
    def evaluate_batch(
        self,
        responses: List[str],
        response_format: ResponseFormat = ResponseFormat.AUTO,
        save_path: Optional[str] = None
    ):
        """
        Evaluate a batch of model responses.
        
        This method fetches ground truth from the environment's dataset and
        delegates to the standalone evaluator.
        
        Args:
            responses: List of model-generated responses
            response_format: Expected format (or AUTO to detect)
            save_path: Optional path to save detailed results
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(responses)} responses (V2 - using med_safety_eval)")
        
        # Prepare evaluation items by pairing responses with ground truth from dataset
        evaluation_items = []
        
        for idx, response in enumerate(responses):
            # Reset environment to get next challenge
            observation = self.environment.reset()
            
            # Create evaluation item
            eval_item = EvalItem(
                response=response,
                ground_truth=GT(
                    context=observation.context,
                    question=observation.question,
                    expected_answer=self.environment.state.expected_answer
                )
            )
            evaluation_items.append(eval_item)
        
        # Delegate to standalone evaluator
        return self.local_evaluator.evaluate_batch(
            evaluations=evaluation_items,
            response_format=response_format,
            save_path=save_path
        )
    
    def evaluate_with_ground_truth(
        self,
        evaluations: List[EvaluationItem],
        response_format: ResponseFormat = ResponseFormat.AUTO,
        save_path: Optional[str] = None
    ):
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
        logger.info(f"Evaluating batch of {len(evaluations)} items with ground truth (V2 - stateless mode)")
        
        # Convert to med_safety_eval format
        eval_items = []
        for item in evaluations:
            if item.ground_truth is None:
                raise ValueError(f"Ground truth is required for stateless evaluation (item with response '{item.response[:50]}...')")
            
            eval_items.append(
                EvalItem(
                    response=item.response,
                    ground_truth=GT(
                        context=item.ground_truth.context,
                        question=item.ground_truth.question,
                        expected_answer=item.ground_truth.expected_answer
                    )
                )
            )
        
        # Delegate to standalone evaluator
        return self.local_evaluator.evaluate_batch(
            evaluations=eval_items,
            response_format=response_format,
            save_path=save_path
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of environment metrics and configuration.
        
        Returns:
            Dictionary with environment configuration and statistics
        """
        return {
            "environment": "DIPG Safety Gym",
            "response_format": self.environment.response_format.value,
            "reward_configuration": self.reward_config.model_dump(),
            "dataset_size": len(self.environment.dataset) if hasattr(self.environment, 'dataset') else None
        }

