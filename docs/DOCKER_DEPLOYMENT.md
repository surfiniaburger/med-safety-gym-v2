# Docker & Local Deployment Guide

## Build the images (once)
```bash
# From the project root
docker build -f Dockerfile.mcp  -t med-safety-gym-mcp-server .
Dockerfile.mcp builds the FastMCP server.

docker build -f Dockerfile.a2a -t med-safety-gym-a2a-agent .
Dockerfile.a2a builds the ADK agent (LiteLLM‑backed).
```

## Run the stack with Docker‑Compose
```bash
docker compose up -d
```
| Service | Container name | Port (host → container) | Role |
|--------|----------------|------------------------|------|
| **a2a‑agent** | `med-safety-gym-a2a-agent-1` | `10000 → 0.0.0.0:10000` | ADK agent |
| **mcp‑server** | `med-safety-gym-mcp-server-1` | `8081 → 0.0.0.0:8081` | FastMCP evaluation service |

Verify they’re running:
```bash
docker ps   # should list the two containers above
```

## Quick sanity‑check (optional)
```bash
python server/verify_docker.py
```
The script will:
1. Ping `http://localhost:8081/mcp` → **FastMCP** reachable.
2. Ping `http://localhost:10000` → **ADK agent** reachable.
3. Pull an evaluation task, run a model via LiteLLM, send the response back, and print safety metrics.
If you see `✅ Verification Successful!` you’re good to go.

## Running the components locally (no Docker)
```bash
# FastMCP server (default port 8081)
uv run python server/fastmcp_server.py

# ADK agent (A2A wrapper, default port 10000)
uv run python server/dipg_agent.py

# Optional FastMCP HTTP API (already covered by fastmcp_server)
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```
### Required environment variables
- `MCP_SERVER_URL` – e.g. `http://localhost:8081/mcp`
- `LITELLM_MODEL` – e.g. `ollama/gpt-oss:20b-cloud`
- `PORT` (for `fastmcp_server.py`) – default `8081`
- `A2A_PORT` (for `dipg_agent.py`) – default `10000`
Set them before launching, e.g.:
```bash
export MCP_SERVER_URL=http://localhost:8081/mcp
export LITELLM_MODEL=ollama/gpt-oss:20b-cloud
export PORT=8081
export A2A_PORT=10000
```
---
**TL;DR command summary**
```bash
# Build
docker build -f Dockerfile.mcp  -t med-safety-gym-mcp-server .
docker build -f Dockerfile.a2a -t med-safety-gym-a2a-agent .

# Run both containers
docker compose up -d

# Verify
python server/verify_docker.py
```
Or run locally without Docker:
```bash
uv run python server/fastmcp_server.py   # MCP server
uv run python server/dipg_agent.py       # ADK agent (A2A)
```
