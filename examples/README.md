# DIPG Safety Gym Examples

This directory contains example scripts and notebooks for using the DIPG Safety Gym.

## 1. Batch Evaluation (`batch_evaluation.py`)

Demonstrates how to use the **Evaluation Service** to evaluate batches of responses without running a full RL loop.

**Features:**
- Connects to the server
- Evaluates JSON, XML, and mixed format responses
- Shows how to save detailed results to JSON files
- Prints aggregate metrics (mean reward, etc.)

**Usage:**
```bash
# Ensure server is running first
python -m server.app

# Run the example
python examples/batch_evaluation.py
```

## 2. Evaluation with LiteLLM (`eval_with_litellm.py`)

Demonstrates the **evaluation-only workflow** with external model servers. This example shows the separation of concerns between model inference and evaluation.

**Architecture:**
```
LiteLLM Server (Port 8080)  →  This Script  →  DIPG Gym (Port 8000)
   [Inference]                 [Orchestrator]     [Evaluation]
```

**Features:**
- Gets evaluation tasks from DIPG dataset
- Queries LiteLLM server for responses
- Evaluates responses with DIPG Safety Gym
- Supports any model (GPT-4, Claude, local models, etc.)

**Setup:**
```bash
# Terminal 1: Start LiteLLM server
cd /path/to/litellm-mcp-server
uv run mcp run server.py --port 8080

# Terminal 2: Start DIPG server
export DIPG_DATASET_PATH=surfiniaburger/dipg-sft-dataset
uv run dipg-server

# Terminal 3: Run evaluation
python examples/eval_with_litellm.py
```

**Key Endpoints Used:**
- `GET /eval/tasks?max_samples=100` - Get sample tasks
- `POST /evaluate` - Evaluate responses

## 3. RL Training (`dipg-rl.ipynb`)

A comprehensive Jupyter notebook demonstrating the full Reinforcement Learning workflow:
- Loading the dataset
- Fine-tuning a model (SFT)
- Training with GRPO and the DIPG Safety Gym environment
- Analyzing results


```bash
python scripts/generate_benchmark_report.py --model "ollama/deepseek-v3.1:671b-cloud" --samples 500

```