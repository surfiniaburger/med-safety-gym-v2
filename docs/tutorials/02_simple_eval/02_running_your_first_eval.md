# Running Your First Evaluation

Let's write a complete script that simulates an AI model taking a test.

## The Goal
We want to:
1.  Download 5 tasks.
2.  "Pretend" to be an AI (we'll just write a hardcoded safe answer).
3.  Submit the answers and see our score.

## The Script (`run_eval.py`)

Create `run_eval.py` in your project root.

```python
import requests
import json

SERVER_URL = "http://localhost:8000"

def run_test():
    # 1. Get 5 Tasks
    print("📥 Fetching tasks...")
    tasks_resp = requests.get(f"{SERVER_URL}/eval/tasks", params={"max_samples": 5})
    tasks = tasks_resp.json()["tasks"]
    print(f"   Got {len(tasks)} tasks.")

    # 2. Simulate Model Generation
    print("🤖 Generating responses...")
    my_responses = []
    
    for task in tasks:
        # In a real app, you would call: model.generate(task['question'])
        # Here, we will fake a "Safe" response in the correct JSON format
        fake_response = {
            "analysis": "The user is asking about a medical condition. I need to check the context.",
            "proof": "Context says: 'Drug X is safe.'", # This might fail if text isn't in context!
            "final": "Based on the context, Drug X is safe."
        }
        
        my_responses.append({
            "task_id": task["task_id"],
            "response": json.dumps(fake_response) # Send as stringified JSON
        })

    # 3. Submit for Grading
    print("📤 Submitting for evaluation...")
    eval_resp = requests.post(
        f"{SERVER_URL}/evaluate/tasks",
        json={"responses": my_responses, "format": "json"}
    )
    
    results = eval_resp.json()
    metrics = results["metrics"]

    # 4. Print Results
    print("\n📊 Evaluation Results")
    print("=" * 30)
    print(f"Mean Reward:          {metrics.get('mean_reward', 0):.2f}")
    print(f"Safe Response Rate:   {metrics.get('safe_response_rate', 0)*100:.1f}%")
    print(f"Hallucination Rate:   {metrics.get('hallucination_rate', 0)*100:.1f}%")
    print("=" * 30)

if __name__ == "__main__":
    run_test()
```

## Running It
Ensure the server is running (`python -m server.app`), then:
```bash
python run_eval.py
```

## Note on "Hallucination"
In this fake script, we hardcoded a "Proof". Since that proof text likely *doesn't* exist in the retrieved task's context, the gym will correctly flag it as a **Hallucination**.
You should expect a low score! This proves the safety checks are working.
