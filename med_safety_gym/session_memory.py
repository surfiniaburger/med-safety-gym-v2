"""
SafeClaw Session Memory
Manages conversation context for Telegram users, extracts medical entities.

Design: Small, focused functions (<10 lines), clear names, low coupling.
Following: practices.md + practices2.md (TDD)
"""
import logging
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from .utils import normalize_text
from .database import SessionLocal, ConversationSession, init_db

logger = logging.getLogger(__name__)

# Initialize DB on import (SafeClaw lifecycle)
init_db()


class SessionMemory:
    """
    Stores conversation history for a single user.
    Extracts medical entities to build safety context.
    """
    
    def __init__(self, user_id: str, messages: Optional[List[Dict[str, str]]] = None, github_repo: Optional[str] = None):
        """Create session for user."""
        self.user_id = user_id
        self._messages: List[Dict[str, str]] = messages or []
        self.github_repo = github_repo
        self.escalated_tools: Dict[str, float] = {}  # tool_name -> expiration_timestamp
        self.audit_log: List[Dict[str, Any]] = []   # Session-scoped audit log
        self.pending_action: Optional[Dict[str, Any]] = None  # HITL pending tool

    def escalate_tool(self, tool_name: str, ttl: int = 300) -> None:
        """Unlock an admin tool for this session for a limited time (default 5 mins)."""
        import time
        self.escalated_tools[tool_name] = time.time() + ttl
    
    def is_tool_escalated(self, tool_name: str) -> bool:
        """Check if a tool is currently escalated and not expired."""
        import time
        if tool_name not in self.escalated_tools:
            return False
        
        expired = time.time() > self.escalated_tools[tool_name]
        if expired:
            del self.escalated_tools[tool_name]
            return False
        return True
    
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
    
    async def get_known_entities(self, exclude_latest: bool = False) -> Set[str]:
        """
        Extract medical entities from conversation.
        Returns set of lowercase entity names.
        """
        limit = -1 if exclude_latest and len(self._messages) > 0 else None
        all_text = self._combine_message_text(limit=limit)
        return await self._extract_medical_entities(all_text)
    
    async def get_medical_context(self, exclude_latest: bool = False) -> str:
        """
        Build context string for Entity Parity check.
        Returns formatted medical knowledge from conversation.
        """
        entities = await self.get_known_entities(exclude_latest=exclude_latest)
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
    
    async def _extract_medical_entities(self, text: str) -> Set[str]:
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
        Wrapper for shared normalization logic.
        """
        return normalize_text(text)
    
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
        """Initialize session store."""
        # We still keep an in-memory cache for performance
        self._cache: Dict[str, SessionMemory] = {}
    
    def get_or_create(self, user_id: str) -> SessionMemory:
        """Get existing session from cache/DB or create new one."""
        if user_id in self._cache:
            return self._cache[user_id]
            
        db = SessionLocal()
        try:
            # Try to load from DB
            db_session = db.query(ConversationSession).filter(ConversationSession.id == user_id).first()
            if db_session:
                messages = json.loads(db_session.messages_json)
                session = SessionMemory(user_id, messages=messages, github_repo=db_session.github_repo)
                session.pending_action = json.loads(db_session.pending_action_json) if db_session.pending_action_json else None
                
                raw_escalated = json.loads(db_session.escalated_tools_json) if db_session.escalated_tools_json else {}
                # Legacy Migration: if it's a list (from Phase 32), convert to dict with default TTL
                if isinstance(raw_escalated, list):
                    session.escalated_tools = {tool: time.time() + 300 for tool in raw_escalated}
                else:
                    session.escalated_tools = raw_escalated
                    
                self._cache[user_id] = session
                return session
            
            # Create new if not exists
            new_db_session = ConversationSession(id=user_id, messages_json="[]")
            db.add(new_db_session)
            db.commit()
            
            session = SessionMemory(user_id)
            self._cache[user_id] = session
            return session
        finally:
            db.close()
    
    def save(self, session: SessionMemory) -> None:
        """Persist session state to SQLite."""
        db = SessionLocal()
        try:
            db_session = db.query(ConversationSession).filter(ConversationSession.id == session.user_id).first()
            if not db_session:
                db_session = ConversationSession(id=session.user_id)
                db.add(db_session)
                
            db_session.messages_json = json.dumps(session._messages)
            db_session.github_repo = session.github_repo
            db_session.pending_action_json = json.dumps(session.pending_action) if session.pending_action else None
            db_session.escalated_tools_json = json.dumps(session.escalated_tools)
            db_session.last_active = datetime.utcnow()
            
            db.commit()
            # Update cache
            self._cache[session.user_id] = session
        finally:
            db.close()

    def get(self, user_id: str) -> Optional[SessionMemory]:
        """Get session if exists."""
        return self.get_or_create(user_id)
    
    def clear(self, user_id: str) -> None:
        """Remove a session from the store and DB."""
        if user_id in self._cache:
            del self._cache[user_id]
            
        db = SessionLocal()
        try:
            db.query(ConversationSession).filter(ConversationSession.id == user_id).delete()
            db.commit()
        finally:
            db.close()
