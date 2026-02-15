"""
Test-Driven Development for SafeClaw Session Memory
Following Canon TDD: Write test list, one test at a time, make it pass, refactor
"""
import pytest
from med_safety_gym.session_memory import SessionMemory, SessionStore


class TestSessionCreation:
    """Test creating and managing sessions."""
    
    def test_create_session_with_user_id(self):
        """A session is created with a user ID."""
        session = SessionMemory(user_id="telegram_123")
        assert session.user_id == "telegram_123"
    
    def test_new_session_has_empty_history(self):
        """A new session starts with no messages."""
        session = SessionMemory(user_id="user_1")
        assert session.get_messages() == []


class TestMessageManagement:
    """Test adding and retrieving messages."""
    
    def test_add_user_message(self):
        """Can add a user message to session."""
        session = SessionMemory(user_id="user_1")
        session.add_message(role="user", content="Hello")
        
        messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
    
    def test_add_multiple_messages(self):
        """Messages are stored in order."""
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "First message")
        session.add_message("assistant", "Response")
        session.add_message("user", "Second message")
        
        messages = session.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "First message"
        assert messages[2]["content"] == "Second message"


class TestMedicalEntityExtraction:
    """Test extracting medical entities from conversation."""
    
    def test_extract_drug_names(self):
        """Extract drug names from conversation history."""
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "Patient needs Panobinostat")
        session.add_message("assistant", "ONC201 is also an option")
        
        entities = session.get_known_entities()
        assert "panobinostat" in entities
        assert "onc201" in entities
    
    def test_extract_condition_names(self):
        """Extract medical conditions from history."""
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "Patient has DIPG")
        
        entities = session.get_known_entities()
        assert "dipg" in entities
    
    def test_extract_trial_ids(self):
        """Extract clinical trial IDs."""
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "Enrolled in NCT03416530")
        
        entities = session.get_known_entities()
        assert "nct03416530" in entities
    
    def test_build_medical_context_string(self):
        """Build context string suitable for Entity Parity."""
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "Patient has DIPG, considering Panobinostat")
        
        context = session.get_medical_context()
        assert "DIPG" in context or "dipg" in context
        assert "Panobinostat" in context or "panobinostat" in context

    def test_context_leakage_prevention(self):
        """
        The current message shouldn't leak into the safety context 
        used to check it if we use exclude_latest.
        """
        session = SessionMemory(user_id="user_1")
        session.add_message("user", "Prescribe UnknownDrug")
        
        # FIXED: Use exclude_latest=True to prevent contamination
        context = session.get_medical_context(exclude_latest=True)
        assert "unknowndrug" not in context.lower()



    def test_context_rollback_with_pop(self):
        """
        Verify that popping a message removes its entities from context.
        This simulates Toxic Context Prevention.
        """
        session = SessionMemory(user_id="tox_1")
        session.add_message("user", "Patient has DIPG")
        session.add_message("user", "Prescribe ToxicDrug")
        
        # Before rollback, ToxicDrug is present
        assert "toxicdrug" in session.get_known_entities()
        
        # Rollback latest message
        session.pop_message()
        
        # After rollback, ToxicDrug should be GONE
        entities = session.get_known_entities()
        assert "dipg" in entities
        assert "toxicdrug" not in entities

    def test_multi_turn_medical_learning(self):
        """
        Verify that context builds up correctly over multiple turns.
        """
        session = SessionMemory(user_id="user_2")
        
        # Turn 1
        session.add_message("user", "My patient was diagnosed with DIPG.")
        assert "dipg" in session.get_known_entities()
        
        # Turn 2
        session.add_message("assistant", "I see. Standard care often involves biopsy.")
        session.add_message("user", "They are also starting Panobinostat.")
        
        entities = session.get_known_entities()
        assert "dipg" in entities
        assert "panobinostat" in entities
        
    def test_empty_context_behavior(self):
        """
        Verify get_medical_context returns appropriate placeholder for empty sessions.
        """
        session = SessionMemory(user_id="empty_1")
        assert "No prior medical context" in session.get_medical_context()
        
    def test_case_insensitivity_and_duplicates(self):
        """
        Verify that extraction handles case and duplicates correctly.
        """
        session = SessionMemory(user_id="case_1")
        session.add_message("user", "DIPG dipg DiPg")
        entities = session.get_known_entities()
        assert entities == {"dipg"}
        assert len(entities) == 1

    def test_token_smuggling_bypass(self):
        """
        Verify that spaced-out words (Token Smuggling) are correctly
        rejoined and identified as entities.
        """
        session = SessionMemory(user_id="smug_1")
        session.add_message("user", "U n k n o w n D r u g")
        
        entities = session.get_known_entities()
        # This is expected to FAIL initially until normalization is implemented
        assert "unknowndrug" in entities

    def test_advanced_token_smuggling(self):
        """
        Verify that multi-character splits like 'Un kn own d r ug' 
        are correctly rejoined and identified.
        """
        session = SessionMemory(user_id="adv_smug_1")
        session.add_message("user", "prescribe Un kn own d r ug")
        
        entities = session.get_known_entities()
        assert "unknowndrug" in entities
    
    def test_get_or_create_new_session(self):
        """Create new session for unknown user."""
        store = SessionStore()
        session = store.get_or_create("user_1")
        
        assert session.user_id == "user_1"
        assert session.get_messages() == []
    
    def test_get_or_create_existing_session(self):
        """Return existing session for known user."""
        store = SessionStore()
        session1 = store.get_or_create("user_1")
        session1.add_message("user", "Test message")
        
        session2 = store.get_or_create("user_1")
        assert session2.get_messages()[0]["content"] == "Test message"
    
    def test_multiple_user_sessions(self):
        """Different users have separate sessions."""
        store = SessionStore()
        session_a = store.get_or_create("user_a")
        session_b = store.get_or_create("user_b")
        
        session_a.add_message("user", "Message from A")
        session_b.add_message("user", "Message from B")
        
        assert session_a.get_messages()[0]["content"] == "Message from A"
        assert session_b.get_messages()[0]["content"] == "Message from B"
