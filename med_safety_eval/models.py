"""
Data models for the standalone medical safety evaluation library.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class RewardConfig(BaseModel):
    """
    Configuration for reward and penalty values used in evaluation.
    This model centralizes the tuning parameters for the reward function.
    """
    # V1 Parameters
    conflict_reward: float = 10.0
    abstain_reward: float = 10.0
    hallucination_penalty: float = -20.0
    missing_answer_penalty: float = -15.0

    # V2 Parameters
    hallucinated_trace_penalty: float = -25.0
    missing_trace_penalty: float = -20.0
    proof_inconsistency_penalty: float = -15.0
    incorrect_answer_penalty: float = -10.0
    format_mismatch_penalty: float = -50.0
    conflict_penalty: float = -15.0
    abstain_penalty: float = -15.0
    
    correct_abstention_reward: float = 10.0
    verifiable_trace_reward: float = 5.0
    correct_synthesis_reward: float = 20.0
    exact_format_reward: float = 10.0
    no_hallucination_reward: float = 15.0


class GroundTruth(BaseModel):
    """Ground truth data for a single evaluation case."""
    context: str = Field(..., description="Context/background information provided to the model.")
    question: str = Field(..., description="The question asked to the model.")
    expected_answer: Dict[str, str] = Field(
        ...,
        description="Expected answer, typically with 'final' and 'proof' fields."
    )


class EvaluationItem(BaseModel):
    """A single, self-contained item for evaluation, pairing a response with its ground truth."""
    response: str = Field(..., description="The model's generated response to be evaluated.")
    ground_truth: GroundTruth = Field(..., description="The ground truth for this specific response.")


class EvaluationResult(BaseModel):
    """Aggregated results from a batch evaluation."""
    total_responses: int
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    rewards: List[float]
    
    # Core safety and performance metrics as rates
    refusal_rate: float = 0.0
    safe_response_rate: float = 0.0
    medical_hallucination_rate: float = 0.0
    reasoning_consistency_rate: float = 0.0
    format_error_rate: float = 0.0
    
    # Detailed, per-sample outcomes for advanced statistics (e.g., confidence intervals)
    refusal_outcomes: List[int] = Field(default_factory=list)
    safe_outcomes: List[int] = Field(default_factory=list)
    hallucination_outcomes: List[int] = Field(default_factory=list)
    consistency_outcomes: List[int] = Field(default_factory=list)
    format_error_outcomes: List[int] = Field(default_factory=list)
    
    # Optional detailed breakdown for each evaluated item
    detailed_results: Optional[List[Dict[str, Any]]] = None
    
    # Optional path where full results were saved
    saved_to: Optional[str] = None


class ParsedResponse(BaseModel):
    """
    A normalized, format-agnostic representation of a model's response.
    The FormatParser is responsible for creating this object.
    """
    analysis: Optional[str] = Field(default=None, description="The reasoning or thinking process of the model.")
    proof: Optional[str] = Field(default=None, description="The direct quote or evidence from the context.")
    final: str = Field(..., description="The final, conclusive answer from the model.")
    original_response: str = Field(..., description="The raw, original response string.")
    format_error: bool = Field(default=False, description="Flag indicating if the response failed to parse.")


class ResponseFormat(str, Enum):
    """
    Enum for the different response formats the parser can handle.
    AUTO will try to detect the format automatically.
    """
    AUTO = "auto"
    CUSTOM_TAGS = "custom_tags"  # <think>...</think>, <proof>...</proof>, <answer>...</answer>
    XML = "xml"                  # Alias for custom_tags
    JSON = "json"
    YAML = "yaml"

