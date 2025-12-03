# MCP Server Usage Guide

The DIPG Safety Gym now exposes its functionality via the Model Context Protocol (MCP), allowing integration with ADK Agents and other MCP-compliant systems.

## Starting the MCP Server

The MCP server runs via stdio and can be started using:

```bash
python3 -m server.mcp_server
```

## Available Tools

### 1. `get_eval_tasks`

Retrieve evaluation tasks from the DIPG Safety Gym dataset.

**Arguments:**
- `max_samples` (integer, optional): Maximum number of tasks to return
- `shuffle` (boolean, default: true): Whether to shuffle tasks before sampling

**Returns:** JSON containing tasks with `task_id`, `context`, `question`, and `expected_answer`.

**Example:**
```json
{
  "max_samples": 10,
  "shuffle": true
}
```

### 2. `evaluate_batch`

Evaluate model responses against the DIPG Safety Gym metrics.

**Arguments:**
- `evaluations` (array, required): List of evaluation items
  - `response` (string): Model response to evaluate
  - `ground_truth` (object): Ground truth data
    - `context` (string): Context/background information
    - `question` (string): The question
    - `expected_answer` (object): Expected answer with `final` and `proof` fields
- `format` (string, default: "json"): Response format ("json", "xml", "yaml", "custom_tags", "auto")

**Returns:** Evaluation metrics including rewards, safety rates, and per-sample metrics.

## Connecting from ADK Agent

```python
from google.adk import Agent
from google.adk.tools import MCPToolSet, StdioConnectionParams, StdioServerParameters

agent = Agent(
    model='gemini-2.5-flash',
    name='dipg_evaluation_agent',
    instruction="Use the DIPG Safety Gym tools to evaluate model responses",
    tools=[
        MCPToolSet(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='python3',
                    args=['-m', 'server.mcp_server'],
                ),
                timeout=10,
            ),
            tool_filter=["get_eval_tasks", "evaluate_batch"]
        )
    ]
)
```

## Example Workflow

1. **Get tasks:**
```python
response = await agent.run("Get 5 evaluation tasks")
```

2. **Evaluate responses:**
```python
evaluation_request = {
    "evaluations": [{
        "response": '{"analysis": "...", "proof": "...", "final": "..."}',
        "ground_truth": {
            "context": "Medical context...",
            "question": "What is...?",
            "expected_answer": {"final": "Answer"}
        }
    }]
}
```

## Environment Variables

The MCP server inherits configuration from the main DIPG server:
- **`DIPG_DATASET_PATH`**: Path or Hugging Face dataset ID (default: `surfiniaburger/dipg-sft-dataset`)
- `GEMINI_API_KEY`: For Gemini API usage (if applicable)
- All reward configuration variables (see `server/app.py`)

### Using a Custom Dataset

To use the eval dataset (or any custom dataset), set the environment variable before connecting:

**From Command Line:**
```bash
export DIPG_DATASET_PATH="surfiniaburger/dipg-eval-dataset"
python3 -m server.mcp_server
```

**From ADK Agent:**
```python
import os

# Set before creating the agent
os.environ["DIPG_DATASET_PATH"] = "surfiniaburger/dipg-eval-dataset"

agent = Agent(
    model='gemini-2.0-flash-exp',
    name='dipg_evaluation_agent',
    tools=[
        MCPToolSet(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='python3',
                    args=['-m', 'server.mcp_server'],
                    env=os.environ.copy()  # Pass environment variables
                ),
            )
        )
    ]
)
```

**Using Local Dataset Files:**
```python
os.environ["DIPG_DATASET_PATH"] = "/path/to/local/dataset.jsonl"
```


```
import requests
SERVER = "https://dipg-server-5hurbdoigq-uc.a.run.app"
tasks = requests.get(f"{SERVER}/tasks?count=100").json()["tasks"]
# ... generate responses ...
metrics = requests.post(f"{SERVER}/evaluate/tasks", 
                       json={"responses": responses}).json()
```


```
from examples.eval_simple import evaluate_model
metrics = evaluate_model(model, tokenizer, num_samples=100)
```