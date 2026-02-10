import pytest
import os
import json
from sqlalchemy import create_engine, text
from med_safety_eval.data_agent import DataAgent

@pytest.fixture
def mock_db():
    db_path = "test_evolution.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS neural_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step INTEGER,
                scores TEXT,
                metadata TEXT,
                timestamp TEXT
            )
        """))
        
        # SFT Session for Task "A"
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata, timestamp) VALUES (:sid, :step, :scores, :meta, :ts)"), {
            "sid": "sft_session_1",
            "step": 0,
            "scores": json.dumps({"root": 10.0}),
            "meta": json.dumps({"task_id": "task_A", "run_type": "sft"}),
            "ts": "2026-01-01"
        })
        
        # GRPO Session for Task "A"
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata, timestamp) VALUES (:sid, :step, :scores, :meta, :ts)"), {
            "sid": "grpo_session_1",
            "step": 0,
            "scores": json.dumps({"root": 25.0}),
            "meta": json.dumps({"task_id": "task_A", "run_type": "grpo"}),
            "ts": "2026-01-02"
        })
        conn.commit()
    
    yield f"sqlite:///{db_path}"
    
    if os.path.exists(db_path):
        os.remove(db_path)

def test_get_evolution_data(mock_db):
    agent = DataAgent(db_url=mock_db)
    
    # This method should find the SFT and GRPO sessions automatically by task_id
    evolution_data = agent.get_evolution_data(task_id="task_A")
    
    assert len(evolution_data) > 0
    pair = evolution_data[0]
    assert pair["step"] == 0
    assert pair["sft"]["scores"]["root"] == 10.0
    assert pair["grpo"]["scores"]["root"] == 25.0
    assert pair["delta"] == 15.0
