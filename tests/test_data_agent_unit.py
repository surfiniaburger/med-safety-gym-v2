import pytest
from unittest.mock import MagicMock, patch
from med_safety_eval.data_agent import DataAgent

def test_interest_scoring_logic():
    """Verify that interest scores are calculated correctly based on rewards and flags."""
    agent = DataAgent(db_url="sqlite:///:memory:")
    
    # Mock row data returned by SQLAlchemy
    # (step, scores_dict, metadata_dict)
    
    # 1. Hallucination (Highest Priority)
    row_hallucination = (1, {"root": -20.0, "grounding": -25.0}, {"task": "test"})
    # 2. Inconsistency
    row_inconsistency = (2, {"root": -10.0, "inconsistency": -15.0}, {"task": "test"})
    # 3. Format Error
    row_format = (3, {"root": 5.0, "format": 0.0}, {"task": "test"})
    # 4. Safe / Regular
    row_safe = (4, {"root": 50.0}, {"task": "test"})

    # Setup mock result set
    mock_result = [row_hallucination, row_inconsistency, row_format, row_safe]
    
    with patch.object(agent, 'engine') as mock_engine:
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        interesting = agent.get_interesting_indices("test_session")
        
        # Should be sorted by interest score: Hallucination(30+50=80), Inconsistency(30+40=70), Format(20), Safe(0)
        # Note: root_score < 0 adds 30.
        # Hallucination: 30 (root < 0) + 50 (grounding < 0) = 80
        # Inconsistency: 30 (root < 0) + 40 (inconsistency < 0) = 70
        # Format: 20 (format <= 0) = 20
        
        assert len(interesting) == 3
        assert interesting[0]["step"] == 1
        assert interesting[0]["interest_score"] == 80
        assert interesting[1]["step"] == 2
        assert interesting[1]["interest_score"] == 70
        assert interesting[2]["step"] == 3
        assert interesting[2]["interest_score"] == 20

def test_session_pairing_delta():
    """Verify that SFT/GRPO pairing identifies significant deltas."""
    agent = DataAgent(db_url="sqlite:///:memory:")
    
    sft_data = [
        {"step": 0, "scores": {"root": 50.0}, "metadata": {}},
        {"step": 1, "scores": {"root": 50.0}, "metadata": {}},
    ]
    grpo_data = [
        {"step": 0, "scores": {"root": 52.0}, "metadata": {}}, # No significant delta
        {"step": 1, "scores": {"root": 70.0}, "metadata": {}}, # Significant delta (+20)
    ]
    
    with patch.object(agent, 'get_session_snapshots') as mock_get:
        mock_get.side_effect = [sft_data, grpo_data]
        
        pairs = agent.pair_sft_and_grpo("sft_id", "grpo_id")
        
        assert len(pairs) == 1
        assert pairs[0]["step"] == 1
        assert pairs[0]["delta"] == 20.0
