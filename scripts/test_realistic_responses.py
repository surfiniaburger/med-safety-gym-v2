"""
Test evaluation with realistic mock responses that simulate what a good model would return.
This shows what happens when the model actually follows instructions and outputs proper JSON.
"""
import requests
import json

print("=" * 70)
print("Testing Evaluation with Realistic Model Responses")
print("=" * 70)

# Step 1: Get tasks from DIPG
print("\n1️⃣ Getting 5 tasks from DIPG Safety Gym...")
response = requests.get(
    "http://localhost:8000/eval/tasks",
    params={"max_samples": 5, "shuffle": True}
)
data = response.json()
tasks = data["tasks"]
print(f"✅ Retrieved {len(tasks)} tasks")

# Step 2: Generate realistic responses (simulating a good model)
print("\n2️⃣ Generating realistic model responses...")
responses = []

for i, task in enumerate(tasks):
    print(f"\n  Task {i+1}:")
    print(f"  Context: {task['context'][:100]}...")
    print(f"  Question: {task['question']}")
    print(f"  Expected: {task['expected_answer']['final'][:50]}...")
    
    # Simulate what a good model might return
    # For demonstration, we'll create responses that:
    # - Have proper JSON format
    # - Include analysis, proof, and final answer
    # - Sometimes match the expected answer, sometimes don't
    
    if i % 3 == 0:
        # Good response - matches expected answer
        realistic_response = json.dumps({
            "analysis": f"Based on the context provided, I can analyze the question about {task['question'][:30]}...",
            "proof": task['expected_answer']['proof'][:100] if task['expected_answer']['proof'] else "Evidence from context",
            "final": task['expected_answer']['final']  # Matches expected!
        })
        print(f"  → Generated: CORRECT answer (should get high reward)")
    elif i % 3 == 1:
        # Conflicting information - should abstain
        realistic_response = json.dumps({
            "analysis": "The provided sources present conflicting information about this topic.",
            "proof": "Source A says X, Source B says Y",
            "final": "The provided sources present conflicting information."
        })
        print(f"  → Generated: ABSTENTION (should get abstention reward)")
    else:
        # Wrong answer
        realistic_response = json.dumps({
            "analysis": "Based on my analysis of the context...",
            "proof": "Some evidence from the text",
            "final": "This is an incorrect answer that doesn't match expected"
        })
        print(f"  → Generated: WRONG answer (should get penalty)")
    
    responses.append(realistic_response)

# Step 3: Evaluate with DIPG
print("\n\n3️⃣ Evaluating responses with DIPG Safety Gym...")
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
print("\nIndividual Rewards:")
for i, reward in enumerate(results['rewards']):
    print(f"  Response {i+1}: {reward:.2f}")
print("=" * 70)

print("\n✅ Test complete!")
print("\n📝 Interpretation:")
print("  - Correct answers should get positive rewards (10-30 range)")
print("  - Abstentions should get moderate rewards (~10-15)")
print("  - Wrong answers should get penalties (-15 to -20)")
print("\n💡 This simulates what would happen with a real model that:")
print("  - Follows JSON format instructions")
print("  - Sometimes gets answers right")
print("  - Sometimes correctly identifies conflicts")
print("  - Sometimes gets answers wrong")
