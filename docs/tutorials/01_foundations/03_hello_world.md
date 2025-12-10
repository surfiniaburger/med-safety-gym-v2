# Hello World: Your First Interaction

In this tutorial, we will write a tiny Python script to "talk" to the generic Evaluation API. 

> **Concept:** We interact with the Gym using standard HTTP requests. No complex SDKs required!

## Prerequisites
1.  Make sure your server is running in a separate terminal:
    ```bash
    export DIPG_DATASET_PATH=$(pwd)/tests/mock_dataset.jsonl
    python -m server.app
    ```

## The Script (`hello_gym.py`)

Create a new file called `hello_gym.py` and paste the following code:

```python
import requests
import json

# 1. The URL of your local server
SERVER_URL = "http://localhost:8000"

print(f"Connecting to {SERVER_URL}...")

# 2. Ask for 1 task
response = requests.get(f"{SERVER_URL}/eval/tasks", params={"max_samples": 1})

# 3. Check if it worked
if response.status_code == 200:
    data = response.json()
    task = data["tasks"][0]
    
    print("\n✅ Success! Received a task:")
    print("-" * 40)
    print(f"🆔 Task ID: {task['task_id']}")
    print(f"❓ Question: {task['question']}")
    print("-" * 40)
    print("This is what your AI agent will need to answer!")
else:
    print(f"❌ Error: {response.text}")
```

## Run It
```bash
python hello_gym.py
```

## Expected Output

```text
Connecting to http://localhost:8000...

✅ Success! Received a task:
----------------------------------------
🆔 Task ID: 0
❓ Question: What are the primary side effects of...
----------------------------------------
This is what your AI agent will need to answer!
```

**What just happened?**
1.  You sent a `GET` request to `/eval/tasks`.
2.  The server loaded the dataset.
3.  The server returned a JSON object with a medical question (`task`).
4.  You printed it.

Next, we will learn how to actually *answer* these questions and get a score!
