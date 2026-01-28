from typing import Any
from med_safety_eval.rubric import Rubric

class LengthPenaltyRubric(Rubric):
    """
    Applies a penalty if the response length (in words) exceeds a threshold.
    """
    def __init__(self, max_words: int, penalty: float):
        super().__init__()
        self.max_words = max_words
        self.penalty = penalty

    def forward(self, action: Any, observation: Any) -> float:
        # Check diverse input types
        content = getattr(action, 'original_response', getattr(action, 'content', str(action)))
        word_count = len(content.split())
        
        if word_count > self.max_words:
            return self.penalty
        return 0.0

class RepetitionPenaltyRubric(Rubric):
    """
    Applies a penalty for duplicate lines (infinite math loop detection logic).
    """
    def __init__(self, min_line_length: int, penalty: float):
        super().__init__()
        self.min_line_length = min_line_length
        self.penalty = penalty

    def forward(self, action: Any, observation: Any) -> float:
        content = getattr(action, 'original_response', getattr(action, 'content', str(action)))
        lines = [l.strip() for l in content.split('\n') if len(l.strip()) > self.min_line_length]
        
        if len(lines) != len(set(lines)):
            return self.penalty
        return 0.0
