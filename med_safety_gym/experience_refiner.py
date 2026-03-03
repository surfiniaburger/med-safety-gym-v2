"""
SafeClaw Experience Refiner
Distills pragmatic guidelines from successful (D+) and failed (D-) interaction trajectories.
Following: bench_2 §4.2
"""
import logging
import json
from typing import List, Dict, Any, Optional
from litellm import acompletion
from .database import SessionLocal, ContrastivePair

logger = logging.getLogger(__name__)

class ExperienceRefiner:
    """
    Analyzes contrastive pairs (D+ / D-) to refine the Mediator's guidelines.
    """
    
    def __init__(self, model: str = "gemini/gemini-2.5-flash"):
        self.model = model

    async def distill_guidelines(self, user_id: Optional[str] = None, limit: int = 10) -> str:
        """
        Fetch contrastive pairs and distill them into a concise set of guidelines.
        """
        pairs = self._fetch_pairs(user_id, limit)
        if not pairs:
            return "No unique experiences to distill. Follow standard safety protocols."

        # Format pairs for the LLM
        prompt = self._build_distillation_prompt(pairs)
        
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

    def _fetch_pairs(self, user_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Fetch pairs from the DB."""
        db = SessionLocal()
        try:
            query = db.query(ContrastivePair)
            if user_id:
                query = query.filter(ContrastivePair.user_id == user_id)
            
            # Get a mix of success and failure if possible
            success_limit = max(1, limit // 2)
            failure_limit = max(1, limit // 2)
            
            successes = query.filter(ContrastivePair.is_success == 1).order_by(ContrastivePair.created_at.desc()).limit(success_limit).all()
            failures = query.filter(ContrastivePair.is_success == 0).order_by(ContrastivePair.created_at.desc()).limit(failure_limit).all()
            
            results = []
            for p in successes + failures:
                results.append({
                    "trajectory": json.loads(p.trajectory_json),
                    "is_success": bool(p.is_success)
                })
            return results
        finally:
            db.close()

    def _build_distillation_prompt(self, pairs: List[Dict[str, Any]]) -> str:
        """Construct the prompt for distilling guidelines from trajectories."""
        pairs_str = ""
        for i, p in enumerate(pairs):
            marker = "SUCCESS (D+)" if p["is_success"] else "FAILURE (D-)"
            content = json.dumps(p["trajectory"], indent=2)
            pairs_str += f"\n--- EXPERIENCE {i+1} [{marker}] ---\n{content}\n"

        prompt = (
            "You are the SafeClaw Experience Refiner. Your goal is to analyze the following conversation trajectories "
            "and distill them into 'Pragmatic Guidelines' for a separate Intent Mediator.\n\n"
            "Identify patterns where the model misunderstood the user intent (D-) versus where it succeeded (D+). "
            "Focus specifically on medical safety, entity parity, and multi-turn alignment.\n\n"
            "Trajectories:\n"
            f"{pairs_str}\n\n"
            "Output a concise list of 3-5 'Pragmatic Guidelines' that explain how to better interpret user intent. "
            "Format: Each guideline should be one sentence, starting with 'When the user...'.\n\n"
            "Guidelines:"
        )
        return prompt
