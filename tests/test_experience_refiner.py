"""
Tests for SafeClaw Experience Refiner
Following Canon TDD
"""
import pytest
import json
from unittest.mock import AsyncMock, patch
from med_safety_gym.experience_refiner import ExperienceRefiner
from med_safety_gym.session_memory import SessionMemory, SessionStore
from med_safety_gym.database import init_db

# Ensure DB is initialized for tests
init_db()

class TestExperienceRefinerAnalysis:
    """Test distillation logic and DB fetching."""

    @pytest.mark.asyncio
    async def test_distill_no_pairs(self):
        """Should return default guidelines if no pairs found."""
        refiner = ExperienceRefiner()
        # Mock fetch_pairs to return empty
        with patch.object(refiner, '_fetch_pairs', return_value=[]):
            res = await refiner.distill_guidelines()
            assert "standard safety protocols" in res

    def test_build_prompt_includes_trajectories(self):
        """Verify the prompt template includes trajectory JSON."""
        refiner = ExperienceRefiner()
        pairs = [{
            "trajectory": [{"role": "user", "content": "hello"}],
            "is_success": True
        }]
        prompt = refiner._build_distillation_prompt(pairs)
        assert "hello" in prompt
        assert "SUCCESS (D+)" in prompt

    @pytest.mark.asyncio
    async def test_distillation_llm_call(self):
        """Mock the LLM call and verify the result."""
        refiner = ExperienceRefiner()
        pairs = [{
            "trajectory": [{"role": "user", "content": "fail case"}],
            "is_success": False
        }]
        
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock()]
        mock_resp.choices[0].message.content = "When the user fails, be more careful."
        
        with patch.object(refiner, '_fetch_pairs', return_value=pairs):
            with patch('med_safety_gym.experience_refiner.acompletion', return_value=mock_resp):
                res = await refiner.distill_guidelines()
                assert "be more careful" in res

class TestLoggingIntegration:
    """Test that agent logging actually persists to DB and refiner can see it."""

    @pytest.mark.asyncio
    async def test_logging_and_fetching(self):
        """Full loop: log a pair, then fetch it with refiner."""
        import uuid
        uid = f"user_refine_{uuid.uuid4().hex}"
        store = SessionStore()
        session = store.get_or_create(uid)
        session.add_message("user", "Test trajectory")
        
        # Log via store
        store.log_contrastive_pair(session, is_success=True)
        
        refiner = ExperienceRefiner()
        pairs = refiner._fetch_pairs(user_id=uid, limit=1)
        
        assert len(pairs) == 1
        assert pairs[0]["is_success"] is True
        assert pairs[0]["trajectory"][0]["content"] == "Test trajectory"
