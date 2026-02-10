import pytest
from fastapi.testclient import TestClient
from med_safety_eval.observability_hub import app, data_agent
import json
import os
from sqlalchemy import create_engine, text

def test_evolution_api_endpoint():
    # Setup mock DB for the test instance of data_agent
    db_path = "test_hub_evolution.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    engine = create_engine(f"sqlite:///{db_path}")
    data_agent.engine = engine
    data_agent.db_url = f"sqlite:///{db_path}"
    
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS neural_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step INTEGER, scores TEXT, metadata TEXT, timestamp TEXT)"))
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata, timestamp) VALUES ('sft_1', 0, '{\"root\": 10}', '{\"task_id\": \"T1\", \"run_type\": \"sft\"}', 'now')"))
        conn.execute(text("INSERT INTO neural_snapshots (session_id, step, scores, metadata, timestamp) VALUES ('grpo_1', 0, '{\"root\": 20}', '{\"task_id\": \"T1\", \"run_type\": \"grpo\"}', 'now')"))
        conn.commit()

    client = TestClient(app)
    response = client.get("/gauntlet/evolution/T1")
    
    # This should fail because the endpoint doesn't exist yet
    assert response.status_code == 200
    data = response.json()
    assert "pairs" in data
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["delta"] == 10
    
    if os.path.exists(db_path):
        os.remove(db_path)
