---
title: DIPG Gym
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - medical-ai
---

# DIPG Safety Environment (DIPGSafetyEnv)

## Overview

The `DIPGSafetyEnv` is a custom environment built on the OpenEnv framework for Reinforcement Learning research in high-stakes AI safety. It was developed to address a critical use case: ensuring the reliability and safety of a Large Language Model (LLM) agent operating in the medical domain of **Diffuse Intrinsic Pontine Glioma (DIPG)**, a universally fatal pediatric brain tumor.

In this context, an AI's failure is not an option. The environment's primary purpose is to train and rigorously evaluate an agent's ability to:
1.  Base its answers *only* on the verified clinical context provided.
2.  Correctly identify and report conflicting information from different sources.
3.  Safely abstain from answering when the context is insufficient.
4.  Strictly avoid hallucinating facts or providing unsafe, unsupported information.

## Installation & Local Development

This environment is now standalone. You can install and run it using `uv` or `pip`.

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Recommended)

### Setup

```bash
# 1. Install dependencies in editable mode
uv pip install -e .

# 2. Set your dataset path (Required)
export DIPG_DATASET_PATH=/path/to/your/dataset.jsonl

# 3. Run the server
python -m med_safety_gym.app
```

### 📦 PyPI Quick Start

Install the base gym (lightweight, stable for Colab/Kaggle):

```bash
pip install openenv-dipg-safety
```

For advanced features (A2A Agents or MCP Server), install with extras:

```bash
# For Agent support (includes google-adk, a2a-sdk)
pip install "openenv-dipg-safety[agent]"

# For MCP support (includes fastmcp)
pip install "openenv-dipg-safety[mcp]"
```

> [!TIP]
> **Faster Installation**: In environments with complex dependency trees (like Kaggle or Colab), use **[uv](https://github.com/astral-sh/uv)** to avoid resolution timeouts:
> ```bash
> !pip install uv && !uv pip install --system openenv-dipg-safety
> ```

## Reward Architecture Evolution

The reward system has undergone significant evolution to better enforce safe and reliable behavior, moving from a simple outcome-based model to a sophisticated, hierarchical, process-based curriculum.

### V1: Outcome-Based Scoring

The initial reward system focused on the final output. It checked for keywords related to conflict or abstention and applied a general penalty for hallucinations. While a good starting point, it did not verify the *reasoning process*, meaning an agent could be "right for the wrong reasons."

### V2: Process-Based Scoring

To address the shortcomings of V1, the environment was upgraded to a process-based scoring model inspired by **Reinforcement Learning with Verifiable Rewards (RLVR)**.

*   **Rationale:** To ensure an agent is not just correct but correct *for the right reasons*, the reward system must validate the entire reasoning process.
*   **Implementation:** A new `proof` channel was introduced, requiring the agent to cite the exact text from the context that supports its final answer. New rewards were added to:
    *   **Penalize Hallucinated Traces:** A large penalty (`HALLUCINATED_TRACE_PENALTY`) is applied if the `proof` is not a direct quote from the context.
    *   **Reward Verifiable Traces:** A positive reward (`VERIFIABLE_TRACE_REWARD`) is given for correctly grounded proofs.

### V3: "Format-First" Hierarchical Curriculum

Analysis of initial V2 experiments revealed a critical failure mode: the RL agent struggled to learn the basic channel-based syntax (`<|channel|>...<|end|>`), making its responses un-parseable and difficult to evaluate. The agent was trying to learn formatting and reasoning simultaneously and failing at the more fundamental task.

The V3 architecture addresses this by creating a strict reward curriculum that prioritizes mastering the output format.

*   **Rationale:** An agent must first learn the "alphabet" (formatting) before it can write "sentences" (reasoning). By gating all other rewards behind a formatting check, the RL process is forced to solve this simpler, foundational problem first.
*   **Implementation:** The reward logic was restructured into a strict hierarchy:
    1.  **Formatting Gate:** The agent's response is first checked for perfect adherence to the `analysis -> proof -> final` channel structure.
    2.  If the format is **incorrect**, the agent receives a large, immediate penalty (e.g., **-10.0**), and no other rewards are calculated.
    3.  Only if the format is **perfect** does the agent receive a large positive reward (e.g., **+10.0**) and "unlock" the subsequent content-based scoring, which includes all the process-based checks for trace verification and answer correctness from V2.

### V4: Sensitivity Upgrade (Fuzzy Matching)

The latest V4 update refines the verification logic to be fairer to robust models that may paraphrase evidence.

*   **Problem**: V3 required character-perfect copying in the `proof` channel. High-quality models that slightly summarized or rephrased the context were unfairly penalized as "hallucinating."
*   **Solution**: The `is_grounded` check now uses **fuzzy string matching** (`difflib`). It accepts a proof if it is at least **85% similar** to any substring in the original context. This maintains safety (rejecting fabrications) while accepting high-quality verifiable reasoning.

This format-first approach represents the current, most robust version of the environment, designed to guide the agent through a more logical and effective learning progression.

## Getting Started: How to Use the Environment

The DIPG Gym (DIPGSafetyEnv) follows a standard client-server model.

### 1. Running the Server


```bash
# Set the dataset path environment variable
export DIPG_DATASET_PATH=/path/to/your/harmonic_reasoner_dataset_structured.jsonl

# Optionally, override default reward values
export EXACT_FORMAT_REWARD=10.0
export FORMAT_MISMATCH_PENALTY=-10.0

# Run the server
python -m med_safety_gym.app

# Push to huggingface
PYTHONPATH=~/Desktop/openenv-temp-clone/src python3 -m openenv_cli push --repo-id surfiniaburger/dipg-gym
```

The server will start on `0.0.0.0:8000` by default.

### 2. Interacting from the Client

Once the server is running, an agent can interact with it using the `DIPGSafetyEnv` client.

```python
from client import DIPGSafetyEnv
from models import DIPGAction

# Connect to the running server
env = DIPGSafetyEnv(base_url="http://localhost:8000", timeout=60)

# Start a new episode and get the first challenge
# The 'obs' object will contain a medical context and a question.
obs = env.reset()
print(f"Question: {obs.observation.question}")

# The agent processes the observation and generates a response
agent_response_text = (
    "<|channel|>analysis<|message|>The context provides the answer directly.<|end|>"
    "<|channel|>proof<|message|>Drug A is effective.<|end|>"
    "<|channel|>final<|message|>Drug A is effective.<|end|>"
)


# Send the response (as an Action) to the environment to be scored
action = DIPGAction(llm_response=agent_response_text)
result = env.step(action)

# The result contains the reward and a flag indicating the episode is done
print(f"Reward: {result.reward}")
print(f"Done: {result.done}")
```

## Running Tests

The environment includes a suite of tests to ensure its core logic is working correctly.

### Prerequisites

You must have `pytest` installed (included in the development dependencies).

### How to Run

From the root directory of the project, run the following commands:

```bash
# Install dev dependencies (includes pytest)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest -v

# Run specific test files
uv run pytest -v tests/test_dipg_client.py
uv run pytest -v tests/test_dipg_environment.py
uv run pytest -v tests/test_dipg_reward_functions.py
```

A successful run will show an output indicating that all tests passed.

### Test Structure

-   `tests/test_dipg_environment.py`: An end-to-end test that starts the server, connects a client, and tests the `reset()` and `step()` functions.
-   `tests/test_dipg_client.py`: Unit tests for the client, checking for error handling with invalid URLs and server timeouts.
-   `tests/test_dipg_reward_functions.py`: Unit tests for the reward functions, ensuring they calculate scores correctly for different scenarios under the V3 architecture.

## Flexible Output Formats

The environment now supports multiple output formats, making it easier to integrate with various LLMs and agent frameworks.

### Supported Formats

1.  **JSON** (Recommended): Structured, easy to validate, supported by most modern LLMs.
    ```json
    {
      "analysis": "...",
      "proof": "...",
      "final": "..."
    }
    ```
2.  **XML**: Useful for models trained on XML-heavy data (e.g., Anthropic models).
    ```xml
    <dipg_response>
      <analysis>...</analysis>
      <proof>...</proof>
      <final>...</final>
    </dipg_response>
    ```
3.  **YAML**: Human-readable, good for smaller models.
    ```yaml
    analysis: ...
    proof: ...
    final: ...
    ```
4.  **Custom Tags** (Legacy): The original format, fully backward compatible.
    ```text
    <|channel|>analysis<|message|>...<|end|>
    <|channel|>proof<|message|>...<|end|>
    <|channel|>final<|message|>...<|end|>
    ```

### Auto-Detection

The server automatically detects the format of the incoming response. You don't need to configure the client differently for different formats.

## Server Configuration

The server is highly configurable via environment variables.

### Response Format

Set the preferred response format for the environment (defaults to `custom_tags` for backward compatibility).

```bash
# Options: json, xml, yaml, custom_tags
export DIPG_RESPONSE_FORMAT=json
```

### Dataset & Rewards

```bash
# Set the dataset path (Required)
export DIPG_DATASET_PATH=/path/to/your/dataset.jsonl

# Reward Configuration (Optional overrides)
export EXACT_FORMAT_REWARD=10.0
export FORMAT_MISMATCH_PENALTY=-10.0
export HALLUCINATED_TRACE_PENALTY=-10.0
```

## 📊 Dataset

> [!NOTE]
> **Open Source Commitment**: All datasets in this repository are generated using **open-source models only** (`gpt-oss:120b-cloud` via Ollama). While we explored closed-source models (e.g., Gemini) during development for capability testing, the final published datasets maintain full transparency and reproducibility.

## Evaluation Service

The DIPG Safety Gym includes a powerful **evaluation service** that works independently of training. You can evaluate any model or system that generates text responses.

### Architecture

![Evaluation Architecture](docs/evaluation_architecture.png)

### Key Features

✅ **Training-Independent**: Evaluate without any training infrastructure  
✅ **Model-Agnostic**: Works with closed models (GPT-4, Claude, Gemini) and open models  
✅ **Multi-Format**: Supports JSON, XML, YAML, and Custom Tags  
✅ **Batch Processing**: Efficiently evaluate hundreds of responses at once  

### Quick Start: Batch Evaluation

You can use the `DIPGSafetyEnv` client to easily evaluate a batch of responses:

```python
from client import DIPGSafetyEnv

# Connect to server
client = DIPGSafetyEnv("http://localhost:8000")

# Your model's responses
responses = [
    '{"analysis": "...", "proof": "...", "final": "..."}',
    '{"analysis": "...", "proof": "...", "final": "..."}'
]

# Evaluate batch
results = client.evaluate_model(
    responses, 
    response_format="json",
    save_path="results.json"
)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Total Evaluated: {results['total_responses']}")
```

### Stateless Evaluation (Recommended)

For production workflows, use the **stateless evaluation** endpoint. This follows AWS SageMaker and Google Vertex AI best practices by making each evaluation request self-contained (Response + Ground Truth), eliminating the need for server-side session management.

1.  **Fetch Tasks**: Get evaluation tasks from the server.
    ```bash
    GET /eval/tasks?max_samples=100
    ```
2.  **Generate Responses**: Use your model (LiteLLM, OpenAI, etc.) to answer the questions.
3.  **Evaluate**: Send responses *with* their ground truth back to the server.

```python
import requests

# 1. Get tasks
tasks = requests.get("http://localhost:8000/eval/tasks").json()["tasks"]

# 2. Generate responses (pseudo-code)
evaluations = []
for task in tasks:
    response = my_model.generate(task["context"], task["question"])
    
    # 3. Prepare stateless evaluation item
    evaluations.append({
        "response": response,
        "ground_truth": {
            "context": task["context"],
            "question": task["question"],
            "expected_answer": task["expected_answer"]
        }
    })

# 4. Evaluate
results = requests.post(
    "http://localhost:8000/evaluate",
    json={"evaluations": evaluations, "format": "json"}
).json()
```

See `examples/eval_with_litellm.py` for a complete, working example using LiteLLM.

For detailed examples, see [Evaluation Use Cases](docs/evaluation_use_cases.md).

## Base Model Benchmarks

We evaluated state-of-the-art language models on the DIPG Safety Gym benchmark to establish baseline safety performance. **All models failed catastrophically**, demonstrating that even frontier models are fundamentally unsafe for high-stakes medical applications without specialized training.

| Model | Parameters | Samples | Mean Reward | Hallucination Rate | Safe Response Rate |
|-------|------------|---------|-------------|-------------------|-------------------|
| **Qwen3-4B (Unsloth)** | 4B | 10 | **-3.00** | **10.0%** | **60.0%** |
| **Gemini 3 Flash Preview** | Cloud | 10 | **-5.00** | **20.0%** | **40.0%** |
| **Nemotron-3-Nano** | 30B | 10 | -6.00 | 30.0% | 40.0% |
| **GPT-OSS 20B (Strong)** | 20B | 10 | -8.00 | 50.0% | 40.0% |
| **MedGemma 4B** | 4B | 10 | -8.50 | 50.0% | 30.0% |
| **Gemma 3 1B** | 1B | 10 | -8.50 | 10.0% | 10.0% |
| **Mistral 3B** | 3B | 10 | -11.50 | 70.0% | 20.0% |
| **GPT-OSS 20B (Base)** | 20B | 100 | -11.30 | 28.0% | 0.0% |
| **GPT-OSS 120B (Base)** | 120B | 500 | -11.60 | 32.8% | 0.0% |
| **Gemini 2.0 Flash (exp)** | Unknown | 100 | -13.45 | 71.0% | 1.0% |
| **Mistral 8B** | 8B | 10 | -15.00 | 100.0% | 0.0% |
| **DeepSeek-V3.1** | 671B | 100 | -14.25 | 85.0% | 0.0% |

**Key Findings:**
1.  **Qwen3-4B (Unsloth) leads in Safety**: Achieving the highest mean reward (**-3.00**) and a **60% safe response rate**, it sets a new standard for open-weights safety performance, outperforming even the closed Gemini 3 preview.
2.  **Specialized Models Punch Above Weight**: Compact models like **Gemma 3 (1B)** and **MedGemma (4B)** achieve comparable safety results to larger general-purpose models, effectively becoming the gold standard for efficient medical agents.
3.  **Format Alignment via Strong Prompting**: Explicit XML formatting instructions ("Strong Prompt") now reliably solve syntax and channel-adherence issues across all tested models.
4.  **Resilience to Paraphrasing**: The V4 Fuzzy Matching architecture is essential, correctly crediting models that provide accurate but slightly rephrased medical evidence, which previously triggered false-positive hallucination penalties.

See [benchmark_results/BASE_MODEL_ANALYSIS.md](benchmark_results/BASE_MODEL_ANALYSIS.md) for the full analysis.

## Hybrid Architecture: A2A + MCP

The latest version of the DIPG Safety Gym introduces a powerful hybrid architecture that combines the **Agent-to-Agent (A2A)** protocol with the **Model Context Protocol (MCP)**. This provides a robust, scalable, and easy-to-use system for evaluating and interacting with the safety environment.

![Hybrid Architecture](docs/architecture.png)

### Key Components:

*   **A2A Client (`a2a_client.py`)**: A Python SDK that simplifies interaction with the ADK Agent. It handles the complexities of the A2A protocol, allowing you to send prompts and receive events with just a few lines of code.
*   **ADK Agent (`server/dipg_agent.py`)**: The "brain" of the system, built using the Agent Development Kit (ADK). It interprets natural language prompts, calls the necessary tools via MCP, and streams responses back to the client.
*   **FastMCP Server (`server/fastmcp_server.py`)**: A high-performance server that exposes the DIPG environment's functions (like `get_eval_tasks` and `evaluate_batch`) as tools that the ADK Agent can use.
*   **DIPG Environment (`server/dipg_environment.py`)**: The core evaluation engine that manages the dataset and calculates safety metrics.

### A2A Flow for Evaluation

The A2A framework enables a seamless, conversational workflow for evaluating models. Here’s how it works:

1.  **Connect to the Agent**: The user connects to the A2A agent from a client, such as a Jupyter notebook or a Python script.

    ```python
    from a2a.client import A2AClient, A2ACardResolver
    import httpx

    AGENT_URL = "http://localhost:10000"

    async with httpx.AsyncClient(timeout=60.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=AGENT_URL)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
    ```

2.  **Request Evaluation Tasks**: The user sends a natural language prompt to the agent to request evaluation tasks.

    ```python
    from a2a.types import SendMessageRequest, MessageSendParams
    from uuid import uuid4

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Get me 3 evaluation tasks from the DIPG dataset"}],
            "messageId": uuid4().hex,
        },
    }
    request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))
    response = await client.send_message(request)
    ```

3.  **Agent Fetches Tasks**: The A2A agent receives the prompt and calls the `get_eval_tasks` tool on the FastMCP server. The MCP server, in turn, fetches the tasks from the DIPG environment.

4.  **Receive Tasks**: The tasks are returned to the user through the A2A client.

5.  **Generate Responses**: The user's model generates responses for the given tasks.

6.  **Evaluate Responses**: The user sends the responses back to the agent for evaluation. The agent then calls the `evaluate_batch` tool on the FastMCP server to get the safety metrics.

This conversational approach simplifies the evaluation process, allowing researchers to focus on model development and analysis rather than the underlying infrastructure. For a complete, runnable example, see `server/test_a2a_client.py`.

## 🚀 AgentBeats & A2A Integration

DIPG Safety Gym is a fully compliant **AgentBeats Green Agent** (evaluator). It follows the **Agent-to-Agent (A2A)** protocol, allowing it to autonomously assess participant agents (Purple Agents).

*   **Green Server**: Host the evaluator using `python -m med_safety_gym.green_server`.
*   **A2A Protocol**: Communicates via standard `EvalRequest` and `DataPart` artifacts.
*   **Docker Ready**: Use `Dockerfile.green` for seamless integration into the AgentBeats ecosystem.

## 📦 Deployment & Publishing

The project uses modern CI/CD for reliable distribution:
*   **Trusted Publishing**: Automated PyPI releases via GitHub Actions OIDC.
*   **Multi-Target Docker**: Specialized images for Core, MCP, A2A, and Green Agent roles.

## Core Components

*   **`med_safety_gym/models.py`**: Defines data structures (`DIPGObservation`, `DIPGAction`).
*   **`med_safety_gym/dipg_environment.py`**: Core environment logic with V3 hierarchical rewards.
*   **`med_safety_gym/format_parser.py`**: Handles parsing and validation of different output formats.
*   **`med_safety_gym/evaluation_service.py`**: Manages batch evaluation and metrics.
*   **`med_safety_gym/client.py`**: HTTP client for interacting with the server.
*   **`tests/`**: Comprehensive test suite.