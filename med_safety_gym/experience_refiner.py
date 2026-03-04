"""
SafeClaw Experience Refiner
Distills pragmatic guidelines from successful (D+) and failed (D-) interaction trajectories.
Following: bench_2 §4.2
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional
from litellm import acompletion
from .database import SessionLocal, ContrastivePair

logger = logging.getLogger(__name__)

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

    def _get_user_traces(self, user_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
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
                if p.semantic_trace_json:
                    traces.append(json.loads(p.semantic_trace_json))
                else:
                    # Fallback for legacy data (extract minimal semantic info if possible)
                    traces.append({
                        "is_success": bool(p.is_success),
                        "legacy": True
                    })
            return traces
        finally:
            db.close()

    def _build_distillation_prompt(self, traces: List[Dict[str, Any]]) -> str:
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

    def _format_semantic_trace(self, trace: Dict[str, Any], index: int) -> str:
        """Farley Habit: Small, focused function for formatting."""
        status = "SUCCESS" if trace.get("is_success") else "FAILURE"
        intent = trace.get("intent", "UNKNOWN")
        # Sanitize failure_reason to prevent trace spoofing or injection
        raw_reason = trace.get("failure_reason", "N/A")
        reason = str(raw_reason).replace("\n", " ").replace("---", "").strip()
        
        entities = ", ".join(trace.get("detected_entities", []))
        
        return (
            f"--- Trace {index+1} [{status}] ---\n"
            f"Intent: {intent}\n"
            f"Outcome Reason: {reason}\n"
            f"Entities Targeted: {entities}\n"
        )
