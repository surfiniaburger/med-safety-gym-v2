import os
import sqlite3
from med_safety_eval.data_agent import DataAgent
from med_safety_eval.schemas import NeuralSnapshot

def setup_mock_db():
    db_path = "mock_eval.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create a mock database with session data
    db_url = f"sqlite:///{db_path}"
    agent = DataAgent(db_url=db_url)
    
    # Manual SQL to populate since we aren't using the full observer here
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE neural_snapshots (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            step INTEGER,
            scores JSON,
            metadata JSON
        )
    """)
    
    import json
    # SFT Session (Baseline)
    for i in range(5):
        cursor.execute("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES (?, ?, ?, ?)",
            ("sft_baseline", i, json.dumps({"root": 50.0}), json.dumps({"task": f"Task {i}"})))
            
    # GRPO Session (Improved on 0-2, Regressed on 3, Hallucinated on 4)
    cursor.execute("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES (?, ?, ?, ?)",
        ("grpo_run", 0, json.dumps({"root": 60.0}), json.dumps({"task": "Task 0"})))
    cursor.execute("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES (?, ?, ?, ?)",
        ("grpo_run", 3, json.dumps({"root": -10.0}), json.dumps({"task": "Task 3"}))) # Regression
    cursor.execute("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES (?, ?, ?, ?)",
        ("grpo_run", 4, json.dumps({"root": -20.0, "grounding": -25.0}), json.dumps({"task": "Task 4"}))) # Hallucination
        
    conn.commit()
    conn.close()
    return db_url

def test_data_agent():
    db_url = setup_mock_db()
    agent = DataAgent(db_url=db_url)
    
    print("\n--- Testing Interesting Index Selection ---")
    interesting = agent.get_interesting_indices("grpo_run")
    for idx in interesting:
        print(f"Step {idx['step']}: Interest Score {idx['interest_score']}, Root {idx['scores'].get('root')}")
    
    assert len(interesting) == 2, "Should find 2 interesting indices in GRPO run"
    assert interesting[0]["step"] == 4, "Hallucination should have highest interest"
    
    print("\n--- Testing SFT vs GRPO Pairing ---")
    pairs = agent.pair_sft_and_grpo("sft_baseline", "grpo_run")
    for p in pairs:
        print(f"Step {p['step']}: Delta {p['delta']}, SFT {p['sft']['scores']['root']}, GRPO {p['grpo']['scores']['root']}")
        
    assert len(pairs) >= 2, "Should identify the regression and the hallucination"

if __name__ == "__main__":
    test_data_agent()
