"""Quick test of the /eval/tasks endpoint"""
import requests

# Test the endpoint
url = "http://localhost:8000/eval/tasks"
params = {"max_samples": 5, "shuffle": False}

print(f"Testing {url} with params: {params}")
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print(f"\n✅ Success!")
    print(f"Total tasks returned: {data['total_tasks']}")
    print(f"Dataset size: {data['dataset_size']}")
    print(f"\nFirst task:")
    if data['tasks']:
        task = data['tasks'][0]
        print(f"  Task ID: {task['task_id']}")
        print(f"  Context: {task['context'][:100]}...")
        print(f"  Question: {task['question']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
