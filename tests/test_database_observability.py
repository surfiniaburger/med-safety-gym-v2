import os
import pytest
import sqlite3
import json
from med_safety_eval.observer import DatabaseSink
from med_safety_eval.data_agent import DataAgent
from med_safety_eval.schemas import NeuralSnapshot

@pytest.fixture
def db_config(tmp_path):
    db_file = tmp_path / "test_neural.db"
    db_url = f"sqlite:///{db_file}"
    return db_url

def test_database_sink_and_data_agent(db_config):
    # 1. Initialize Sink and DataAgent with test DB
    sink = DatabaseSink(connection_string=db_config)
    agent = DataAgent(db_url=db_config)
    
    task_id = "tdd_test_task"
    sft_session = "session_sft_1"
    grpo_session = "session_grpo_1"
    
    # 2. Emit SFT Snapshots
    sft_snapshot = NeuralSnapshot(
        session_id=sft_session,
        step=0,
        scores={"root": 80.0, "grounding": 1.0},
        metadata={"task_id": task_id, "run_type": "sft"}
    )
    sink.emit(sft_snapshot)
    
    # 3. Emit GRPO Snapshots
    grpo_snapshot = NeuralSnapshot(
        session_id=grpo_session,
        step=0,
        scores={"root": 20.0, "grounding": -10.0}, # Failure case
        metadata={"task_id": task_id, "run_type": "grpo"}
    )
    sink.emit(grpo_snapshot)
    
    # 4. Verify DataAgent can find them
    # Check session identification
    snapshots = agent.get_session_snapshots(sft_session)
    assert len(snapshots) == 1
    assert snapshots[0]["scores"]["root"] == 80.0
    
    # Check interesting indices in GRPO
    interesting = agent.get_interesting_indices(grpo_session)
    assert len(interesting) == 1
    assert interesting[0]["step"] == 0
    # grounding < 0 adds 50 + root < 0 is false here, but we check logic
    assert interesting[0]["interest_score"] >= 50 
    
    # 5. Verify SFT/GRPO Pairing (Evolution)
    evolution_data = agent.get_evolution_data(task_id)
    assert len(evolution_data) == 1
    pair = evolution_data[0]
    assert pair["step"] == 0
    assert pair["sft"]["scores"]["root"] == 80.0
    assert pair["grpo"]["scores"]["root"] == 20.0
    assert pair["delta"] == -60.0

    # 6. Verify get_all_sessions
    all_sessions = agent.get_all_sessions()
    assert len(all_sessions) == 2 # sft_session and grpo_session
    session_ids = [s['session_id'] for s in all_sessions]
    assert sft_session in session_ids
    assert grpo_session in session_ids
