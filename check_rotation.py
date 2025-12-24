
import os
import sys
import logging

# Mock the environment or import it
from med_safety_gym.app import get_environment

# Set dummy env vars
os.environ["DIPG_DATASET_PATH"] = "tests/mock_dataset.jsonl"

def test_rotation():
    print("Testing rotation...")
    env1 = get_environment()
    obs1 = env1.reset()
    print(f"Env 1 Reset: {obs1.question}")
    
    env2 = get_environment()
    obs2 = env2.reset()
    print(f"Env 2 Reset: {obs2.question}")
    
    if obs1.question != obs2.question:
        print("✅ SUCCESS: Questions are different.")
    else:
        print("❌ FAILURE: Questions are identical.")

if __name__ == "__main__":
    test_rotation()
