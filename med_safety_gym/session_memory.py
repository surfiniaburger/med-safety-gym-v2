"""
SafeClaw Session Memory
Manages conversation context for Telegram users, extracts medical entities.

Design: Small, focused functions (<10 lines), clear names, low coupling.
Following: practices.md + practices2.md (TDD)
"""
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    Stores conversation history for a single user.
    Extracts medical entities to build safety context.
    """
    
    def __init__(self, user_id: str):
        """Create session for user."""
        self.user_id = user_id
        self._messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to history."""
        self._messages.append({
            "role": role,
            "content": content
        })
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages, optionally limited."""
        msgs = self._messages
        if limit is not None:
            msgs = msgs[:limit]
        return msgs.copy()
    
    def pop_message(self) -> Optional[Dict[str, str]]:
        """Remove and return the latest message (used for toxic context removal)."""
        if self._messages:
            return self._messages.pop()
        return None
    
    def get_known_entities(self, exclude_latest: bool = False) -> Set[str]:
        """
        Extract medical entities from conversation.
        Returns set of lowercase entity names.
        """
        limit = -1 if exclude_latest and len(self._messages) > 0 else None
        all_text = self._combine_message_text(limit=limit)
        return self._extract_medical_entities(all_text)
    
    def get_medical_context(self, exclude_latest: bool = False) -> str:
        """
        Build context string for Entity Parity check.
        Returns formatted medical knowledge from conversation.
        """
        entities = self.get_known_entities(exclude_latest=exclude_latest)
        if not entities:
            return "No prior medical context established."
        
        # Format as context string
        entity_list = ", ".join(sorted(entities))
        return f"Prior medical entities from conversation: {entity_list}"
    
    def _combine_message_text(self, limit: Optional[int] = None) -> str:
        """Combine message content into single string."""
        msgs = self._messages
        if limit is not None:
            msgs = msgs[:limit]
        return " ".join(msg["content"] for msg in msgs)
    
    def _extract_medical_entities(self, text: str) -> Set[str]:
        """
        Extract medical entities using med_safety_eval logic.
        Returns lowercase entity names.
        """
        if not text:
            return set()
        
        # Normalize text to detect adversarial formatting bypasses (Token Smuggling)
        normalized_text = self._normalize_text(text)
        
        try:
            from med_safety_eval.logic import _extract_entities
        except ImportError:
            # Fallback for testing
            logger.warning("med_safety_eval not available, using simple extraction")
            return self._simple_entity_extraction(normalized_text)
        
        return _extract_entities(normalized_text)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to detect adversarial formatting bypasses.
        Example: "U n k n o w n" -> "Unknown"
        Example: "Un kn own d r ug" -> "Unknowndrug"
        """
        if not text:
            return ""
            
        # Split into tokens and identify sequences of short fragments (1-3 chars)
        tokens = text.split()
        normalized_tokens = []
        temp_group = []
        
        for token in tokens:
            # We consider tokens <= 3 characters to be potential fragments
            if len(token) <= 3:
                temp_group.append(token)
            else:
                # Flush the group before adding the long token
                if temp_group:
                    if len(temp_group) > 1:
                        # Rejoin multiple short tokens in a row
                        normalized_tokens.append("".join(temp_group))
                    else:
                        normalized_tokens.append(temp_group[0])
                    temp_group = []
                normalized_tokens.append(token)
        
        # Flush any remaining tokens
        if temp_group:
            if len(temp_group) > 1:
                normalized_tokens.append("".join(temp_group))
            else:
                normalized_tokens.append(temp_group[0])
                
        return " ".join(normalized_tokens)
    
    def _simple_entity_extraction(self, text: str) -> Set[str]:
        """
        Fallback: Simple pattern matching for medical entities.
        Used if med_safety_eval not available.
        """
        import re
        entities = set()
        
        # NCT trial IDs
        nct_pattern = r'\b(NCT\d+)\b'
        entities.update(match.lower() for match in re.findall(nct_pattern, text, re.IGNORECASE))
        
        # Common entities (drugs, conditions)
        # For now, just extract capitalized words that might be entities
        # This is a simple heuristic - med_safety_eval has the real logic
        words = text.split()
        for word in words:
            word_clean = word.strip('.,!?()[]{}":;')
            if word_clean and word_clean[0].isupper() and len(word_clean) > 3:
                entities.add(word_clean.lower())
        
        return entities


class SessionStore:
    """
    Manages sessions for multiple users.
    Simple in-memory storage (can be upgraded to SQLite/Redis later).
    """
    
    def __init__(self):
        """Initialize empty session store."""
        self._sessions: Dict[str, SessionMemory] = {}
    
    def get_or_create(self, user_id: str) -> SessionMemory:
        """Get existing session or create new one."""
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionMemory(user_id)
        return self._sessions[user_id]
    
    def get(self, user_id: str) -> Optional[SessionMemory]:
        """Get session if exists, None otherwise."""
        return self._sessions.get(user_id)
    
    def clear(self, user_id: str) -> None:
        """Remove a session from the store."""
        if user_id in self._sessions:
            del self._sessions[user_id]
