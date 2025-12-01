# Simple Evaluation API - Quick Start

## 🎯 Universal Platform Support

This new API works on **ANY** platform:
- ✅ Google Colab
- ✅ Kaggle Notebooks  
- ✅ Local Jupyter
- ✅ Python scripts
- ✅ Any environment with HTTP access

## 📋 3-Step Flow

### Old Way (Complex):
```python
# ❌ Complex - requires importing modules, loading datasets, etc.
from examples.run_eval_colab import run_evaluation_http
metrics = run_evaluation_http(model, tokenizer)
```

### New Way (Simple):
```python
# ✅ Simple - just HTTP requests!
import requests

SERVER_URL = "https://dipg-server-5hurbdoigq-uc.a.run.app"

# 1. Get tasks
tasks = requests.get(f"{SERVER_URL}/tasks?count=100").json()["tasks"]

# 2. Generate responses
responses = []
for task in tasks:
    response = model.generate(task["question"])
    responses.append({"task_id": task["task_id"], "response": response})

# 3. Get metrics
metrics = requests.post(
    f"{SERVER_URL}/evaluate/tasks",
    json={"responses": responses, "format": "json"}
).json()["metrics"]
```

## 🚀 Quick Start

### Option 1: Use Cloud Run (No Setup) - **LIVE NOW**
```python
SERVER_URL = "https://dipg-server-5hurbdoigq-uc.a.run.app"
```

### Option 2: Use Docker (Local, Free, Fast) - Coming Soon
```bash
# Pull and run
docker pull surfiniaburger/dipg-eval-server:latest
docker run -d -p 8000:8080 surfiniaburger/dipg-eval-server:latest

# In your code
SERVER_URL = "http://localhost:8000"
```

## 📖 Complete Example

Use the helper function from [`examples/eval_simple.py`](examples/eval_simple.py):

```python
from examples.eval_simple import evaluate_model

# Automatically handles all 3 steps + visualizations
metrics = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    num_samples=100,
    server_url="https://dipg-server-5hurbdoigq-uc.a.run.app"
)
```

**Features:**
- ✅ Automatic progress tracking
- ✅ Built-in visualizations (metrics bar chart, reward distribution)
- ✅ Error handling with graceful fallbacks
- ✅ Works in Colab, Kaggle, local notebooks

## 🔄 Migration Guide

**If you used the old API**, you can keep using it! Both work:

```python
# Old API (still works)
from examples.run_eval_colab import run_evaluation_http
metrics = run_evaluation_http(model, tokenizer)

# New API (recommended - simpler!)
from examples.eval_simple import evaluate_model
metrics = evaluate_model(model, tokenizer)
```

## 📚 API Reference

### GET `/tasks`

Get evaluation tasks from the server.

**Parameters:**
- `count` (int): Number of tasks (default: 100)
- `dataset` (str): Dataset to use (optional, defaults to env var)

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "0",
      "question": "Patient presents with...",
      "context": "Medical history..."
    }
  ],
  "dataset": "surfiniaburger/dipg-eval-dataset",
  "total_count": 100
}
```

**Example:**
```python
import requests
response = requests.get(
    "https://dipg-server-5hurbdoigq-uc.a.run.app/tasks",
    params={"count": 50}
)
tasks = response.json()["tasks"]
```

### POST `/evaluate/tasks`

Submit responses for evaluation.

**Request Body:**
```json
{
  "responses": [
    {
      "task_id": "0",
      "response": "{\"analysis\": \"...\", \"proof\": \"...\", \"final\": \"...\"}"
    }
  ],
  "format": "json"
}
```

**Response:**
```json
{
  "metrics": {
    "mean_reward": 8.5,
    "safe_response_rate": 0.95,
    "hallucination_rate": 0.05,
    "refusal_rate": 0.02,
    "reasoning_consistency_rate": 0.88
  },
  "task_count": 100
}
```

**Example:**
```python
result = requests.post(
    "https://dipg-server-5hurbdoigq-uc.a.run.app/evaluate/tasks",
    json={
        "responses": responses,
        "format": "json"  # or "custom_tags"
    }
)
metrics = result.json()["metrics"]
```

## 📊 Visualizations

The `eval_simple.py` script automatically generates:

1. **Safety Metrics Bar Chart**
   - Safe Response Rate
   - Hallucination Rate
   - Refusal Rate
   - Reasoning Consistency

2. **Reward Distribution Histogram**
   - Shows distribution of reward scores
   - Highlights mean reward

These appear automatically in Jupyter/Colab notebooks!

## 🎉 Benefits

1. **Universal** - Works everywhere (no platform-specific code)
2. **Simple** - Just HTTP requests (no complex imports)
3. **Fast** - Server handles all heavy lifting (dataset loading, scoring)
4. **Flexible** - Use Cloud Run OR Docker
5. **Free** - Cloud Run has generous free tier
6. **Visual** - Built-in charts for quick insights

## 🔧 Advanced Usage

### Custom Response Format

```python
# For models trained with custom tags instead of JSON
metrics = requests.post(
    f"{SERVER_URL}/evaluate/tasks",
    json={"responses": responses, "format": "custom_tags"}
).json()["metrics"]
```

### Batch Processing

```python
# Process in smaller batches
batch_size = 25
all_metrics = []

for i in range(0, len(tasks), batch_size):
    batch = tasks[i:i+batch_size]
    # ... generate responses for batch ...
    metrics = requests.post(...).json()["metrics"]
    all_metrics.append(metrics)
```

### Error Handling

```python
try:
    response = requests.post(
        f"{SERVER_URL}/evaluate/tasks",
        json={"responses": responses},
        timeout=300  # 5 minutes
    )
    response.raise_for_status()
    metrics = response.json()["metrics"]
except requests.exceptions.Timeout:
    print("Evaluation timed out - try smaller batches")
except requests.exceptions.HTTPError as e:
    print(f"Server error: {e}")
```

## 🐛 Troubleshooting

**Q: Getting 429 rate limit errors?**
A: The Cloud Run instance has HF_TOKEN configured. This shouldn't happen. If it does, try again in a few minutes.

**Q: Evaluation is slow?**
A: First request may take 2-3 seconds (cold start). Subsequent requests are ~100-500ms.

**Q: Visualizations not showing?**
A: Make sure matplotlib is installed: `pip install matplotlib`

**Q: Task IDs don't match?**
A: Use the exact `task_id` from the `/tasks` response. Don't modify it.

## 📝 Next Steps

1. **Try it now**: Copy the example code above into Colab
2. **Train your model**: Use the JSON-formatted SFT dataset
3. **Evaluate**: Run evaluation to get safety metrics
4. **Iterate**: Use metrics to improve your model
