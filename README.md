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
python -m server.app
```

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
python -m server.app

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

## Core Components

*   **`models.py`**: Defines data structures (`DIPGObservation`, `DIPGAction`).
*   **`server/dipg_environment.py`**: Core environment logic with V3 hierarchical rewards.
*   **`server/format_parser.py`**: Handles parsing and validation of different output formats.
*   **`server/evaluation_service.py`**: Manages batch evaluation and metrics.
*   **`client.py`**: HTTP client for interacting with the server.
*   **`tests/`**: Comprehensive test suite.
