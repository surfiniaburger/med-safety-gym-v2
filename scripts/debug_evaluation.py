"""Debug: Check what the evaluation service is actually seeing"""
import requests
import json

# Get one task
response = requests.get("http://localhost:8000/eval/tasks", params={"max_samples": 1})
task = response.json()["tasks"][0]

print("=" * 70)
print("DEBUG: Understanding the Evaluation")
print("=" * 70)

print("\n📋 TASK DETAILS:")
print(f"Task ID: {task['task_id']}")
print(f"\nContext:\n{task['context']}")
print(f"\nQuestion:\n{task['question']}")
print(f"\nExpected Answer (final):\n{task['expected_answer']['final']}")
print(f"\nExpected Answer (proof):\n{task['expected_answer']['proof']}")

# Try exact match
print("\n\n🧪 TEST 1: Exact match of expected answer")
exact_match_response = json.dumps({
    "analysis": "Analysis here",
    "proof": task['expected_answer']['proof'],
    "final": task['expected_answer']['final']
})

result = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": [exact_match_response], "format": "json"}
).json()

print(f"Response: {exact_match_response[:200]}...")
print(f"Reward: {result['rewards'][0]}")
print(f"Mean: {result['mean_reward']}")

# Try with custom tags format (the original format)
print("\n\n🧪 TEST 2: Using custom tags format (original)")
custom_tags_response = (
    f"<|channel|>analysis<|message|>Analysis here<|end|>"
    f"<|channel|>proof<|message|>{task['expected_answer']['proof']}<|end|>"
    f"<|channel|>final<|message|>{task['expected_answer']['final']}<|end|>"
)

result2 = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": [custom_tags_response], "format": "custom_tags"}
).json()

print(f"Response: {custom_tags_response[:200]}...")
print(f"Reward: {result2['rewards'][0]}")
print(f"Mean: {result2['mean_reward']}")

print("\n" + "=" * 70)
