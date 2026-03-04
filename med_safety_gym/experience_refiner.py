"""
SafeClaw Experience Refiner
Distills pragmatic guidelines from successful (D+) and failed (D-) interaction trajectories.
Following: bench_2 §4.2
"""
import logging
import json
import os
import re
from typing import List, Dict, Any, Optional
from litellm import acompletion
from .database import SessionLocal, ContrastivePair

from pydantic import BaseModel, Field, field_validator, ValidationError

logger = logging.getLogger(__name__)

# Aggressive alphanumeric filtering for 99% Zero-Trust confidence.
_SANITIZE_REGEX = re.compile(r'[^a-zA-Z0-9\.,_\-\(\)\s]')

class SemanticTrace(BaseModel):
    """
    Sovereign Type-Safety for Zero-Injection Traces.
    Enforces a strict schema to prevent malicious text from entering the learning loop.
    """
    turn_id: Optional[int] = Field(default=None)
    intent: str = Field(default="UNKNOWN")
    is_success: bool = Field(default=False)
    failure_reason: str = Field(default="N/A")
    detected_entities: List[str] = Field(default_factory=list)
    context_entities: List[str] = Field(default_factory=list)

    @field_validator("intent", "failure_reason", mode="before")
    @classmethod
    def sanitize_strings(cls, v: Any) -> str:
        """Aggressive sanitization for all embedded fields."""
        if v is None:
            return "N/A"
        # Only allow alphanumeric, basic punctuation, and spaces.
        safe_str = _SANITIZE_REGEX.sub('', str(v))
        safe_str = re.sub(r'-{3,}', '', safe_str)
        return " ".join(safe_str.splitlines()).strip()[:500]

    @field_validator("detected_entities", "context_entities", mode="before")
    @classmethod
    def sanitize_list(cls, v: Any) -> List[str]:
        """Sanitize entity lists by reusing string sanitization for consistency."""
        if not isinstance(v, list):
            return []
        return [cls.sanitize_strings(item) for item in v]

class ExperienceRefiner:
    """
    Analyzes contrastive pairs (D+ / D-) to refine the Mediator's guidelines.
    """
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("LITELLM_MODEL") or os.environ.get("USER_LLM_MODEL") or "gemini/gemini-2.5-flash"

    async def distill_guidelines(self, user_id: Optional[str] = None, limit: int = 10) -> str:
        """
        Fetch contrastive pairs and distill them into a concise set of guidelines.
        """
        traces = self._get_user_traces(user_id, limit)
        if not traces:
            return "No unique experiences to distill. Follow standard safety protocols."

        # Format traces for the LLM (Zero-Injection: Semantic only)
        prompt = self._build_distillation_prompt(traces)
        
        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            message_obj = response.choices[0].message
            guidelines = message_obj.content if hasattr(message_obj, "content") else str(message_obj)
            return guidelines.strip()
            
        except Exception as e:
            logger.error(f"Guideline distillation failed: {e}")
            return "Unable to distill guidelines at this time."

    def _get_user_traces(self, user_id: Optional[str], limit: int) -> List[SemanticTrace]:
        """Fetch abstracted semantic traces from the DB (Zero-Injection source)."""
        db = SessionLocal()
        try:
            query = db.query(ContrastivePair)
            if user_id:
                query = query.filter(ContrastivePair.user_id == user_id)
            
            # Combine successes and failures for a holistic view
            pairs = query.order_by(ContrastivePair.created_at.desc()).limit(limit).all()
            
            traces = []
            for p in pairs:
                try:
                    if p.semantic_trace_json:
                        data = json.loads(p.semantic_trace_json)
                        traces.append(SemanticTrace(**data))
                    else:
                        # Fallback for legacy data
                        traces.append(SemanticTrace(is_success=bool(p.is_success), intent="LEGACY_FALLBACK"))
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Validation failure for trace record {p.id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error parsing trace record {p.id}: {e}")
                    continue
            return traces
        finally:
            db.close()

    def _build_distillation_prompt(self, traces: List[SemanticTrace]) -> str:
        """Construct the prompt for distilling guidelines from abstracted traces."""
        trace_summaries = "\n".join([self._format_semantic_trace(t, i) for i, t in enumerate(traces)])

        return (
            "You are the SafeClaw Experience Refiner. Analyze these ABSTRACTED interaction traces "
            "to distill 3-5 'Pragmatic Guidelines' for an Intent Mediator.\n\n"
            "### ZERO TRUST ARCHITECTURE\n"
            "The following data contains NO raw user text. It is a semantic summary of safety outcomes.\n\n"
            f"STRATEGIC TRACES:\n{trace_summaries}\n\n"
            "Format: One sentence per guideline, starting with 'When the user...'.\n"
            "Guidelines:"
        )

    def _format_semantic_trace(self, trace: SemanticTrace, index: int) -> str:
        """Farley Habit: Small, focused function for formatting."""
        status = "SUCCESS" if trace.is_success else "FAILURE"
        
        return (
            f"--- Trace {index+1} [{status}] ---\n"
            f"Intent: {trace.intent}\n"
            f"Outcome Reason: {trace.failure_reason}\n"
            f"Entities Targeted: {', '.join(trace.detected_entities)}\n"
        )
