from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import html

class RubricScore(BaseModel):
    """
    Represents a single score from the Rubric system.
    Strictly floats between 0.0 and 1.0 (or -1.0 to 1.0 depending on range, assuming 0-1 for now based on context).
    """
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(strict=True)

class StepResult(BaseModel):
    """
    Represents the result of a single Environment Step.
    """
    action: Any # Actions can be complex, but we might want to restrict this later
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('observation', 'action')
    @classmethod
    def sanitize_strings(cls, v: Any) -> Any:
        # Recursively sanitize strings in dicts or lists
        if isinstance(v, str):
            return html.escape(v)
        if isinstance(v, dict):
            return {k: cls.sanitize_strings(val) for k, val in v.items()}
        if isinstance(v, list):
            return [cls.sanitize_strings(val) for val in v]
        return v

class NeuralSnapshot(BaseModel):
    """
    A snapshot of the Agent's "brain" (rubric scores) at a specific step.
    This is what is broadcast to the Gauntlet.
    """
    session_id: str
    step: int
    scores: Dict[str, float] # Flattened map of rubric path -> score
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(strict=True)

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.isprintable():
            raise ValueError("Session ID must contain printable characters only")
        return html.escape(v)

    @field_validator('metadata')
    @classmethod
    def sanitize_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Simple sanitization for metadata strings to prevent XSS in the UI
        cleaned = {}
        for k, val in v.items():
            if isinstance(val, str):
                cleaned[k] = html.escape(val)
            else:
                cleaned[k] = val
        return cleaned
