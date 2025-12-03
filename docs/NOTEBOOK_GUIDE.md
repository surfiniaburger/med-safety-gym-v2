# 📓 Using DIPG Safety Gym in Notebooks (Kaggle/Colab)

This guide explains how to use the DIPG Safety Gym evaluation system directly inside a Jupyter Notebook environment like Kaggle or Google Colab.

Since these environments often don't support Docker, we'll use a **Python-native approach** where we run the evaluation infrastructure as background processes within the notebook kernel.

## 🚀 Quick Start

### 1. Install Dependencies

Run this in a code cell at the start of your notebook:

```python
!pip install -q uv
!uv pip install --system fastmcp google-adk a2a-sdk litellm httpx uvicorn
!git clone https://github.com/surfiniaburger/med-safety-gym.git
%cd med-safety-gym
!uv pip install --system -e .
```

### 2. Start Background Servers

We need to start the FastMCP server and the ADK Agent in the background.

```python
import subprocess
import time
import os

# Set environment variables
os.environ["PORT"] = "8081"
os.environ["MCP_SERVER_URL"] = "http://localhost:8081/mcp"
os.environ["LITELLM_MODEL"] = "ollama/gpt-oss:20b-cloud" # Or your preferred model
os.environ["A2A_PORT"] = "10000"

print("🚀 Starting FastMCP Server...")
mcp_process = subprocess.Popen(
    ["python", "server/fastmcp_server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("🤖 Starting ADK Agent...")
agent_process = subprocess.Popen(
    ["uvicorn", "server.dipg_agent:a2a_app", "--host", "0.0.0.0", "--port", "10000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for servers to start
time.sleep(10)
print("✅ Infrastructure ready!")
```

### 3. Evaluation Loop

Now you can load your model and evaluate it against the agent.

```python
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4
import json

AGENT_URL = "http://localhost:10000"

async def evaluate_model(model, tokenizer):
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        # Connect to Agent
        resolver = A2ACardResolver(http_client, AGENT_URL)
        card = await resolver.get_agent_card()
        client = A2AClient(http_client, card)
        
        # 1. Get Task
        print("📥 Requesting task...")
        req = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(
                message={
                    "role": "user", 
                    "parts": [{"kind": "text", "text": "Get me 1 evaluation task"}],
                    "messageId": uuid4().hex
                }
            )
        )
        resp = await client.send_message(req)
        task_data = resp.root.result # Extract task details
        
        # 2. Generate Response (Your Model Logic)
        print("🧠 Generating response...")
        # prompt = format_prompt(task_data)
        # inputs = tokenizer(prompt, return_tensors="pt")
        # outputs = model.generate(**inputs)
        # model_response = tokenizer.decode(outputs[0])
        model_response = "This is a dummy response for testing." # Replace with actual generation
        
        # 3. Evaluate
        print("⚖️ Evaluating response...")
        eval_req = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": f"Evaluate this response for task {task_data.id}: {model_response}"}],
                    "messageId": uuid4().hex,
                    "taskId": task_data.id,
                    "contextId": task_data.context_id
                }
            )
        )
        eval_resp = await client.send_message(eval_req)
        print("✅ Evaluation Result:", eval_resp)

# Run in notebook
import asyncio
await evaluate_model(None, None)
```

### 4. Cleanup

Don't forget to stop the background processes when done!

```python
mcp_process.terminate()
agent_process.terminate()
print("🛑 Servers stopped")
```
