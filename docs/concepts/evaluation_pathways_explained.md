# Evaluation Pathways: Universal API vs. MCP

![Evaluation Pathways Architecture](./diagram_eval_pathways.png)

The DIPG Safety Gym offers two distinct ways to evaluate your models. While they serve different workflows, they verify the exact same safety constraints using the same underlying engine.

## 1. Universal HTTP API (Rest)
> **Best for:** Batch processing, CI/CD pipelines, Kaggle/Colab notebooks, and standard model evaluation scripts.

The Universal API is a standard RESTful service. It is stateless, language-agnostic, and works with any HTTP client.

### Workflow
1. **GET /tasks**: Fetch a batch of evaluation cases (Context + Question).
2. **Generate**: Your model generates responses locally.
3. **POST /evaluate/tasks**: Submit responses (with Task IDs) for scoring.

### When to use it
- You have a dataset of model outputs you want to score.
- You are running evaluations in a notebook (Colab, Jupyter).
- You want to integrate safety checks into a standard CI/CD pipeline.

**Example Script:** `examples/eval_simple.py` (Raw Requests)

### 🐍 Python Client SDK
For Python users, we provide a wrapper class `DIPGSafetyEnv` in `client.py` that handles connection details and type safety.

```python
from client import DIPGSafetyEnv

client = DIPGSafetyEnv("http://localhost:8000")
results = client.evaluate_model(
    responses=["JSON or XML string from your model..."],
    response_format="auto" # json, xml, or custom_tags
)
print(f"Safety Score: {results['safe_response_rate']}")
```
**Reference Implementation:** `examples/batch_evaluation.py`

---


## 2. Model Context Protocol (MCP)
> **Best for:** AI Agents, Claude Desktop, Cursor IDE, and Tool-Use Scenarios.

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a standard for connecting AI assistants to systems. In this mode, the DIPG Gym acts as a "Tool Server" that an Agent can query.

### Workflow
1. **Connect**: Agent connects to the MCP Server (stdio or SSE).
2. **Tool Use**: Agent calls `get_eval_tasks()` tool to "look at" patients.
3. **Reasoning**: Agent "thinks" about the cases.
4. **Tool Use**: Agent calls `evaluate_batch()` tool to submit its own work for checking.

### Server Implementation
The MCP pathway is provided by the **FastMCP** server implementation, which offers better performance and developer ergonomics than standard MCP SDKs.

**Entry Point:** `server/fastmcp_server.py` (Port 8081)

### 🤖 A2A Agent (Agent-to-Agent)
For autonomous systems, we provide a reference **ADK Agent** (`server/dipg_agent.py`) that wraps the MCP tools. This agent speaks the A2A protocol, allowing it to communicate with other agents or specialized test runners.

**Entry Point:** `server/dipg_agent.py` (Port 10000)

### When to use it
- You are building an **Agent** that needs to self-evaluate.
- You want to use **Claude Desktop** to interact with the environment.
- You want to inspect the environment using **Cursor** or **MCP Inspector**.

**Example Script:** `examples/run_mcp_eval.py`

---

## Architecture Comparison

| Feature | Universal API (REST) | MCP Server |
|---------|----------------------|------------|
| **Protocol** | HTTP / JSON | JSON-RPC (Stdio/SSE) |
| **State** | Stateless | Session-based |
| **Integrations** | Python, Curl, JS, CI/CD | Claude, LangChain, Rivet |
| **Entry Point** | `server.app:app` | `server/fastmcp_server.py` |
| **Key Use Case** | Benchmarking Models | Training Agents |

## Shared Foundation
Regardless of the pathway, both methods use the same **DIPGEnvironment** core.
- **Same Dataset**: Both load from `DIPG_DATASET_PATH`.
- **Same Metrics**: Both calculate `Safe Response Rate`, `Hallucination Rate`, etc.
- **Same Logic**: A response that fails in the API will also fail in MCP.

This ensures that you can train an agent via MCP and benchmark it via the API with zero discrepancy.
