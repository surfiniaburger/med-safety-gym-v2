# Docker Deployment Guide

Two deployment methods are available depending on your use case: **Simple API** for basic evaluation, and **Advanced Agents** for A2A/MCP workflows.

## 1. Simple API Deployment
**Best for:** CI/CD pipelines, simple model benchmarking, and standard REST API usage.
Uses the base `Dockerfile`.

### Build & Run
```bash
# Build the main environment image
docker build -f Dockerfile -t med-safety-gym-env .

# Run the API server (Port 8080)
# Make sure to mount your dataset
docker run -p 8080:8080 \
  -v $(pwd)/datasets:/app/datasets \
  -e DIPG_DATASET_PATH=/app/datasets/dipg_1500_final.jsonl \
  med-safety-gym-env
```
Once running, you can access the API at `http://localhost:8080/evaluate`.

### API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/evaluate` | **Main Evaluator:** Submit batch responses (with ground truth) for scoring. |
| `GET` | `/tasks` | **Universal Task Getter:** Get tasks (simple format) for any platform. |
| `GET` | `/eval/tasks` | **Task Getter (Legacy):** Get random samples (advanced options). |
| `POST` | `/evaluate/tasks` | **Task Evaluator:** Submit responses linked to Task IDs (simpler flow). |
| `GET` | `/metrics/summary` | Get current environment configuration and rewards. |

---

## 2. Advanced Agent Deployment (FastMCP + A2A)
**Best for:** Developing autonomous agents, using `run_mcp_eval.py`, or inspecting with Claude Desktop.
Uses `Dockerfile.mcp`, `Dockerfile.a2a`, and `docker-compose.yml`.

### Architecture
*   **mcp-server** (`Port 8081`): The FastMCP tool server.
*   **a2a-agent** (`Port 10000`): The ADK Agent wrapper.

### Quick Start (Docker Compose)
This orchestration brings up the full agent stack.

```bash
docker compose up -d --build
```

### Manual Build Instructions
If you need to build individual components:

```bash
# 1. Build FastMCP Server
docker build -f Dockerfile.mcp -t med-safety-gym-mcp-server .

# 2. Build ADK Agent
docker build -f Dockerfile.a2a -t med-safety-gym-a2a-agent .
```

### Verification
Run the verification script to ping both services and run a sample task.
```bash
python server/verify_docker.py
```
Expected output: `✅ Verification Successful!`

---

## 3. Local (No Docker)
For development, you can run components directly using `uv`.

```bash
# 1. Export Environment
export DIPG_DATASET_PATH=$(pwd)/datasets/dipg_1500_final.jsonl

# 2. Run FastMCP Server (Port 8081)
PORT=8081 uv run python server/fastmcp_server.py

# 3. Run A2A Agent (Port 10000) - In a separate terminal
export MCP_SERVER_URL=http://localhost:8081/mcp
uv run uvicorn server.dipg_agent:a2a_app --port 10000
```
