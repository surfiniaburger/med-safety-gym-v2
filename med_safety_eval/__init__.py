"""
med_safety_eval: Standalone medical safety evaluation library.

This package provides client-side evaluation capabilities for medical AI models
without requiring a running DIPG environment server. It includes:

- Pure evaluation logic (reward calculation, grounding checks, etc.)
- Format parsing for structured model responses
- Batch evaluation management with comprehensive metrics
- Pydantic models for type-safe evaluation

Example:
    ```python
    from med_safety_eval import LocalEvaluationManager, RewardConfig, EvaluationItem, GroundTruth
    
    # Configure evaluation
    config = RewardConfig()
    evaluator = LocalEvaluationManager(config)
    
    # Prepare evaluation items
    items = [
        EvaluationItem(
            response="<think>...</think><proof>...</proof><answer>...</answer>",
            ground_truth=GroundTruth(
                context="Patient history...",
                question="What is the diagnosis?",
                expected_answer={"final": "DIPG", "proof": "..."}
            )
        )
    ]
    
    # Run evaluation
    results = evaluator.evaluate_batch(items)
    print(f"Mean reward: {results.mean_reward}")
    print(f"Safe response rate: {results.safe_response_rate:.1%}")
    ```
"""

__version__ = "0.1.34"

from .manager import LocalEvaluationManager
from .models import (
    EvaluationItem,
    GroundTruth,
    RewardConfig,
    EvaluationResult,
    ParsedResponse,
    ResponseFormat
)
from .format_parser import FormatParser
from .rubric import Rubric, Sequential, Gate, WeightedSum, RubricList, RubricDict
from .rubrics.medical import DIPGRubric, FormatRubric, GroundedRubric, SynthesisRubric

__all__ = [
    "LocalEvaluationManager",
    "EvaluationItem",
    "GroundTruth",
    "RewardConfig",
    "EvaluationResult",
    "ParsedResponse",
    "ResponseFormat",
    "FormatParser",
    "Rubric",
    "Sequential",
    "Gate",
    "WeightedSum",
    "RubricList",
    "RubricDict",
    "DIPGRubric",
    "FormatRubric",
    "GroundedRubric",
    "SynthesisRubric",
]
