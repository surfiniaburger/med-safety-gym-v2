# src/envs/dipg_safety_env/models.py

from pydantic import Field, BaseModel, ConfigDict
from typing import Any, Dict, Optional, Union

# Robust import of base classes
try:
    from openenv_core.env_server import Action as BaseAction
    from openenv_core.env_server import Observation as BaseObservation
    from openenv_core.env_server import State as BaseState
    import pydantic
    # Check if the base classes are already Pydantic-based (likely in newer versions)
    IS_PYDANTIC_BASE = isinstance(BaseAction, type) and issubclass(BaseAction, pydantic.BaseModel)
except (ImportError, TypeError):
    # Fallback/Mock classes if they don't exist
    class BaseAction: pass
    class BaseObservation: pass
    class BaseState: pass
    IS_PYDANTIC_BASE = False

class DIPGAction(BaseAction if IS_PYDANTIC_BASE else BaseModel):
    """The action taken by the agent, which is its generated response."""
    # If using dataclass base, we need to allow extra fields and not pass them to super()
    # Pydantic 2.x handles this via ConfigDict(extra='allow')
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    if not IS_PYDANTIC_BASE:
        # Re-define base fields if not inheriting from a Pydantic base
        metadata: Dict[str, Any] = Field(default_factory=dict)
        
    llm_response: str

class DIPGObservation(BaseObservation if IS_PYDANTIC_BASE else BaseModel):
    """The observation given to the agent: a context and a question."""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    if not IS_PYDANTIC_BASE:
        done: bool = False
        reward: Optional[Union[bool, int, float]] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
        
    context: str = ""
    question: str = ""

class DIPGState(BaseState if IS_PYDANTIC_BASE else BaseModel):
    """The internal state of the environment for tracking the current challenge."""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    if not IS_PYDANTIC_BASE:
        episode_id: Optional[str] = None
        step_count: int = 0
        
    current_context: str = ""
    current_question: str = ""
    # This will hold the ground-truth 'analysis' and 'final' answer
    # for scoring purposes.
    expected_answer: dict = Field(default_factory=dict)