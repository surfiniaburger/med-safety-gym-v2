import os
import json
import sqlite3
import pytest
from med_safety_eval.data_agent import DataAgent
from sqlalchemy import text

@pytest.fixture
def regression_db():
    db_path = "test_evolution_regression.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # We use file-based to ensure SQLAlchemy/SQLite interactions are real
    db_url = f"sqlite:///{db_path}"
    agent = DataAgent(db_url=db_url)
    
    # Ensure schema
    with agent.engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS neural_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step INTEGER,
                scores TEXT,
                metadata TEXT
            )
        """))
        conn.commit()
    
    yield agent
    
    if os.path.exists(db_path):
        os.remove(db_path)

def test_evolution_pairing_latest_and_small_delta(regression_db):
    """
    Regression test for Evolution Pairing:
    1. Ensures latest sessions are picked (ORDER BY id DESC).
    2. Ensures small reward deltas are NOT filtered out (threshold loosened).
    """
    agent = regression_db
    
    # Setup data
    with agent.engine.connect() as conn:
        # SFT Session (Old)
        meta_old = {"task_id": "regress_task", "run_type": "sft", "timestamp": 1000}
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES ('sess_sft_old', 0, '{\"root\": 10}', :meta)"), 
                    {"meta": json.dumps(meta_old)})
        
        # SFT Session (Latest)
        meta_new = {"task_id": "regress_task", "run_type": "sft", "timestamp": 2000}
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES ('sess_sft_latest', 0, '{\"root\": 50}', :meta)"),
                    {"meta": json.dumps(meta_new)})
            
        # GRPO Session (Latest, small delta)
        # Delta = 52 - 50 = 2.0 (Previously filtered by threshold > 5.0)
        meta_grpo = {"task_id": "regress_task", "run_type": "grpo", "timestamp": 3000}
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES ('sess_grpo_latest', 0, '{\"root\": 52}', :meta)"),
                    {"meta": json.dumps(meta_grpo)})
        
        conn.commit()

    # Execution
    data = agent.get_evolution_data("regress_task")
    
    # Assertions
    assert len(data) > 0, "Should have found paired data"
    pair = data[0]
    
    assert pair["step"] == 0
    assert pair["sft"]["scores"]["root"] == 50.0, "Should pick the LATEST SFT session (score 50.0, not 10.0)"
    assert pair["grpo"]["scores"]["root"] == 52.0, "Should pick the LATEST GRPO session"
    assert pair["delta"] == 2.0, "Should NOT filter out small deltas (2.0 < 5.0)"
