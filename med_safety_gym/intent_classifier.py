from enum import Enum, auto
from dataclasses import dataclass

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
    def __init__(self):
        # Basic heuristic keywords for initial classification
        self.refinement_keywords = ["i meant", "instead", "specifically", "rather"]
        self.expansion_keywords = ["what about", "also", "and for", "how about"]
        self.follow_up_keywords = ["what are", "how does", "why is", "is it", "does it", "can it"]
        self.recollection_keywords = ["earlier", "previously", "remember", "what did we say"]
        self.correction_keywords = ["no,", "not exactly", "incorrect", "wrong"]

    def classify(self, message: str) -> IntentResult:
        message_lower = message.lower()
        
        category = IntentCategory.NEW_TOPIC
        is_correction = False
        
        if any(kw in message_lower for kw in self.correction_keywords) or message_lower.startswith("no "):
            is_correction = True

        if any(kw in message_lower for kw in self.refinement_keywords):
            category = IntentCategory.REFINEMENT
        elif any(kw in message_lower for kw in self.expansion_keywords):
            category = IntentCategory.EXPANSION
        elif any(kw in message_lower for kw in self.follow_up_keywords):
            category = IntentCategory.FOLLOW_UP
        elif any(kw in message_lower for kw in self.recollection_keywords):
            category = IntentCategory.RECOLLECTION
            
        return IntentResult(category=category, is_correction=is_correction)
