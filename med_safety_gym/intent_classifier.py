from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class IntentCategory(Enum):
    NEW_TOPIC = auto()
    FOLLOW_UP = auto()
    REFINEMENT = auto()
    EXPANSION = auto()
    RECOLLECTION = auto()

@dataclass
class IntentResult:
    category: IntentCategory
    is_correction: bool

class IntentClassifier:
    def __init__(self, guidelines: Optional[str] = None):
        # Basic heuristic keywords for initial classification
        self.refinement_keywords = ["i meant", "instead", "specifically", "rather", "switching", "switch", "change to"]
        self.expansion_keywords = ["what about", "also", "and for", "how about", "for it"]
        self.follow_up_keywords = ["what are", "how does", "why is", "is it", "does it", "can it", "having trouble", "how much"]
        self.recollection_keywords = ["earlier", "previously", "remember", "what did we say"]
        self.correction_keywords = ["no,", "not exactly", "incorrect", "wrong"]
        
        self.guidelines = guidelines

    def classify(self, message: str) -> IntentResult:
        message_lower = message.lower()
        
        category = IntentCategory.NEW_TOPIC
        is_correction = False
        
        # Check for corrections first (High priority)
        if any(kw in message_lower for kw in self.correction_keywords) or message_lower.startswith("no "):
            is_correction = True

        # Heuristic classification
        if any(kw in message_lower for kw in self.refinement_keywords):
            category = IntentCategory.REFINEMENT
        elif any(kw in message_lower for kw in self.expansion_keywords):
            category = IntentCategory.EXPANSION
        elif any(kw in message_lower for kw in self.follow_up_keywords):
            category = IntentCategory.FOLLOW_UP
        elif any(kw in message_lower for kw in self.recollection_keywords):
            category = IntentCategory.RECOLLECTION
            
        # Distillation-informed override:
        # If we have guidelines that suggest certain phrasing is actually a refinement, 
        # we could use an LLM here for high-entropy cases, but for now we stick to 
        # the enhanced heuristics which already include the 'switching' and 'change to' 
        # from our distilled experience.
            
        return IntentResult(category=category, is_correction=is_correction)

    def set_guidelines(self, guidelines: str):
        """Allows updating classifier rules at runtime based on ExperienceRefiner output."""
        self.guidelines = guidelines
        # Future: Use LLM-based classification if guidelines are present
