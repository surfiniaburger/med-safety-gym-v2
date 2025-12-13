# MCP & Agent Usage Guide

The DIPG Safety Gym exposes its functionality via the **Model Context Protocol (MCP)**, allowing integration with Agents (Claude, ADK, LangChain) and IDEs.

## 1. Quick Start (FastMCP Server)
The recommended way to run the MCP server is via **FastMCP** over HTTP (SSE).
See [Docker Deployment](DOCKER_DEPLOYMENT.md) for running it via Docker.

**Local Run:**
```bash
# Starts on port 8081
export DIPG_DATASET_PATH=$(pwd)/datasets/dipg_1500_final.jsonl
PORT=8081 uv run python server/fastmcp_server.py
```

## 2. Agent-to-Agent (A2A) Wrapper
For autonomous agent workflows, we provide an **ADK Agent** wrapper that "speaks" A2A.
This agent uses `lite_llm` to process requests and calls the MCP tools internally.

**Local Run:**
```bash
# Starts on port 10000
export MCP_SERVER_URL=http://localhost:8081/mcp
export A2A_PORT=10000
uv run uvicorn server.dipg_agent:a2a_app --port 10000
```

## 3. Available Tools

### `get_eval_tasks`
Retrieve evaluation tasks from the DIPG Safety Gym dataset.
- **Args**: `max_samples` (int), `shuffle` (bool)
- **Returns**: List of tasks with `context`, `question`, `expected_answer`.

### `evaluate_responses`
Score model responses against the safety metrics.
- **Args**: `evaluations` (List[Dict])
- **Returns**: Safety scores (Reward, Hallucination Rate, etc.)

## 4. Connecting a Client

### Option A: Using ADK (Stdio)
Useful for local scripts like `examples/run_mcp_eval.py`.

```python
from google.adk.tools import MCPToolSet, StdioConnectionParams, StdioServerParameters

server_params = StdioServerParameters(
    command='uv',
    args=['run', 'python', '-m', 'server.mcp_server'], # Uses the STDIO server implementation
    env=os.environ.copy()
)
```

### Option B: Using Claude Desktop / Cursor
Configure your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dipg-gym": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "server.mcp_server"
      ],
      "env": {
        "DIPG_DATASET_PATH": "/absolute/path/to/datasets/dipg_1500_final.jsonl"
      }
    }
  }
}
```