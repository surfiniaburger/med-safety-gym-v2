import json
import sys
import os

def wrap(input_file, output_file, agent_id="purple_agent_id"):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    wrapped = {
        "participants": {
            "purple_agent": agent_id
        },
        "results": [
            data
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(wrapped, f, indent=2)
    print(f"✅ Wrapped {input_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python wrap_results.py <input.json> <output.json> [agent_id]")
    else:
        aid = sys.argv[3] if len(sys.argv) > 3 else "purple_agent_id"
        wrap(sys.argv[1], sys.argv[2], aid)
