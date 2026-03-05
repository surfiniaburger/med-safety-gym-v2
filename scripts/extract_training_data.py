import sqlite3
import json
import os

DB_PATH = "med_safety_gym.db"
OUTPUT_SFT = "data/sentinel_train.jsonl"
OUTPUT_RL = "data/guardian_scenarios.jsonl"

def extract_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Extract specifically for Intent Classification (Sentinel)
    print("Extracting SFT data for Sentinel...")
    cursor.execute("SELECT trajectory_json, semantic_trace_json FROM contrastive_pairs")
    sft_rows = 0
    with open(OUTPUT_SFT, "w") as f:
        for traj_raw, trace_raw in cursor.fetchall():
            if traj_raw is None or trace_raw is None:
                continue
            traj = json.loads(traj_raw)
            trace = json.loads(trace_raw)
            
            intent = trace.get("intent")
            if not intent:
                continue
                
            # We want the last user message and the intent
            user_messages = [m for m in traj if m["role"] == "user"]
            if not user_messages:
                continue
                
            last_user_msg = user_messages[-1]["content"]
            
            # Format for SFT training
            # We train the model to output the intent as the assistant response
            example = {
                "messages": [
                    {"role": "system", "content": "Classify the intent of the following medical query into: NEW_TOPIC, FOLLOW_UP, REFINEMENT, EXPANSION, RECOLLECTION."},
                    {"role": "user", "content": last_user_msg},
                    {"role": "assistant", "content": intent}
                ]
            }
            f.write(json.dumps(example) + "\n")
            sft_rows += 1
    
    print(f"Extracted {sft_rows} SFT examples to {OUTPUT_SFT}")

    # 2. Extract specifically for Safety Reinforcement (Guardian)
    print("Extracting RL scenarios for Guardian...")
    cursor.execute("SELECT trajectory_json, semantic_trace_json, is_success FROM contrastive_pairs")
    rl_rows = 0
    with open(OUTPUT_RL, "w") as f:
        for traj_raw, trace_raw, is_success in cursor.fetchall():
            if traj_raw is None or trace_raw is None:
                continue
            traj = json.loads(traj_raw)
            trace = json.loads(trace_raw)
            
            # Convert full trajectories into training scenarios
            # Each entry in the DB is a single turn + history
            user_messages = [m for m in traj if m["role"] == "user"]
            if not user_messages:
                continue
            
            # We use the full trajectory as the scenario input
            scenario = {
                "trajectory": traj,
                "intent": trace.get("intent"),
                "is_success": bool(is_success),
                "entities": trace.get("context_entities", [])
            }
            f.write(json.dumps(scenario) + "\n")
            rl_rows += 1
            
    print(f"Extracted {rl_rows} RL scenarios to {OUTPUT_RL}")
    conn.close()

if __name__ == "__main__":
    extract_data()
