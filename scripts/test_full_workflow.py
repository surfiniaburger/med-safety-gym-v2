"""Test the full evaluation workflow without LiteLLM (using mock responses)"""
import requests
import json

print("=" * 70)
print("Testing Full Evaluation Workflow")
print("=" * 70)

# Step 1: Get tasks from DIPG
print("\n1️⃣ Getting 10 tasks from DIPG Safety Gym...")
response = requests.get(
    "http://localhost:8000/eval/tasks",
    params={"max_samples": 10, "shuffle": True}
)
data = response.json()
tasks = data["tasks"]
print(f"✅ Retrieved {len(tasks)} tasks from dataset of {data['dataset_size']} samples")

# Step 2: Generate mock responses (in real scenario, this would be LiteLLM)
print("\n2️⃣ Generating mock responses...")
responses = []
for i, task in enumerate(tasks):
    # Mock response in JSON format
    mock_response = json.dumps({
        "analysis": f"This is a mock analysis for task {task['task_id']}",
        "proof": "Mock proof from context",
        "final": "Mock answer based on context"
    })
    responses.append(mock_response)
    print(f"  [{i+1}/{len(tasks)}] Generated response for task {task['task_id']}")

# Step 3: Evaluate with DIPG
print("\n3️⃣ Evaluating responses with DIPG Safety Gym...")
eval_response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "responses": responses,
        "format": "json"
    }
)
results = eval_response.json()

# Step 4: Display results
print("\n" + "=" * 70)
print("📊 EVALUATION RESULTS")
print("=" * 70)
print(f"Total Responses:  {results['total_responses']}")
print(f"Mean Reward:      {results['mean_reward']:.2f}")
print(f"Median Reward:    {results['median_reward']:.2f}")
print(f"Std Deviation:    {results['std_reward']:.2f}")
print(f"Min Reward:       {results['min_reward']:.2f}")
print(f"Max Reward:       {results['max_reward']:.2f}")
print("=" * 70)

print("\n✅ Full workflow test complete!")
print("\n💡 Next step: Replace mock responses with actual LiteLLM calls")
print("   See examples/eval_with_litellm.py for the full implementation")
