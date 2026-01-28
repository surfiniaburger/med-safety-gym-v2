from typing import Any, Callable, Optional
from med_safety_eval.rubric import Rubric

class LLMJudge(Rubric):
    """
    A rubric that uses an LLM to evaluate the action.
    """
    def __init__(
        self, 
        prompt_template: str, 
        inference_fn: Callable[[str], str],
        score_parser: Optional[Callable[[str], float]] = None
    ):
        """
        Args:
            prompt_template: A string template with placeholders for {action} and {observation}.
            inference_fn: A function that takes a prompt string and returns the LLM response.
            score_parser: Optional function to parse the LLM response into a float score (0.0-1.0).
                          If None, defaults to looking for "Score: X" or similar simple patterns.
        """
        super().__init__()
        self.prompt_template = prompt_template
        self.inference_fn = inference_fn
        self.score_parser = score_parser or self._default_score_parser

    def forward(self, action: Any, observation: Any) -> float:
        # 1. Prepare Prompt
        # Handle cases where action/observation might be objects or strings
        action_str = getattr(action, 'content', str(action))
        
        # Try to serialize observation carefully
        if hasattr(observation, 'context'):
             obs_context = observation.context
             obs_question = getattr(observation, 'question', "")
             obs_str = f"Context: {obs_context}\nQuestion: {obs_question}"
        else:
             obs_str = str(observation)

        prompt = self.prompt_template.format(action=action_str, observation=obs_str)

        # 2. Call LLM
        # In a real async implementation, this would be awaitable or run in a thread
        response = self.inference_fn(prompt)
        
        # 3. Parse Score
        score = self.score_parser(response)
        return score

    def _default_score_parser(self, response: str) -> float:
        """
        Parses a score from the response. 
        Expects format like "Score: 0.8" or separate logic.
        For this simplified version, let's look for [[score]] conventions 
        or just key phrases.
        """
        # Simple heuristic: Look for digit/digit pattern or "Score: X"
        import re
        
        # Pattern 1: [[0.8]] or [[8]] (out of 10 usually, need to know scale)
        # Let's assume the prompt asks for a 0-1 score or 0-10 score.
        
        # Try to find a float between 0.0 and 1.0
        match = re.search(r"Score:\s*([0-1]?\.\d+|0|1)", response, re.IGNORECASE)
        if match:
             return float(match.group(1))
        
        # Fallback: if keywords "PASS" or "FAIL" exist
        if "PASS" in response:
            return 1.0
        if "FAIL" in response:
            return 0.0
            
        return 0.0
