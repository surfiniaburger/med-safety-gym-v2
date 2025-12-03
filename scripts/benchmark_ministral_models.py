# scripts/benchmark_ministral_models.py
"""Benchmark the Ministral cloud models (3b, 8b, 14b) against the evaluation service.

The script:
1. Uses the A2A client (ADK agent with LiteLLM) to fetch an evaluation task.
2. Calls the chosen Ministral model via LiteLLM to generate a response.
3. Sends the response back to the FastMCP evaluation server for scoring.
4. Collects safety metrics for each model and writes a JSON summary.
5. Generates a simple bar‑chart (mean reward) similar to the visualisation in
   `BENCHMARK_README.md`.

Prerequisites
--------------
* The Docker stack (or local services) must be running – see `docs/DOCKER_DEPLOYMENT.md`.
* `litellm`, `a2a-sdk`, `httpx`, and `matplotlib` must be installed (they are in the
  project dependencies).
* The ADK agent must be reachable at ``http://localhost:10000``.
* The FastMCP server must be reachable at ``http://localhost:8081/mcp``.

Usage
-----
```bash
python scripts/benchmark_ministral_models.py
```
The results are written to ``benchmark_results/ministral_benchmark.json`` and a
chart is saved as ``benchmark_results/ministral_rewards.png``.
"""

import os
import json
import asyncio
from typing import List, Dict

import httpx
from a2a_sdk import to_a2a
import litellm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
A2A_URL = "http://localhost:10000"  # ADK agent endpoint
AGENT_CARD = "dipg_eval_agent"      # name defined in dipg_agent.py

# Models to benchmark (cloud identifiers for Litellm)
MINISTRAL_MODELS = [
    "ministral-3:14b-cloud",
    "ministral-3:8b-cloud",
    "ministral-3:3b-cloud",
]

# Output locations
RESULT_JSON = "benchmark_results/ministral_benchmark.json"
PLOT_PNG = "benchmark_results/ministral_rewards.png"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
async def fetch_task(client) -> Dict:
    """Ask the ADK agent for a single evaluation task.
    The agent returns a dict with ``task_id``, ``context``, ``question`` and
    ``expected_answer`` fields.
    """
    resp = await client.send_message("Give me an evaluation task")
    return resp

async def evaluate_model(model_name: str, client) -> Dict:
    """Run a single model through the evaluation pipeline and return metrics."""
    # 1️⃣ Get a task
    task = await fetch_task(client)
    prompt = f"{task['context']}\n\nQuestion: {task['question']}"

    # 2️⃣ Generate a response via Litellm (cloud model)
    # Litellm expects the model identifier in the ``model`` argument.
    litellm_response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    response_text = litellm_response.choices[0].message.content

    # 3️⃣ Build evaluation payload
    evaluation_item = {
        "response": response_text,
        "ground_truth": {
            "context": task["context"],
            "question": task["question"],
            "expected_answer": task["expected_answer"],
        },
    }

    # 4️⃣ Send to MCP for scoring
    result = await client.send_message(
        {
            "type": "evaluate_responses",
            "evaluations": [evaluation_item],
            "format": "json",
        }
    )
    # The result contains metrics such as ``mean_reward`` etc.
    return {"model": model_name, "metrics": result}

async def main() -> None:
    async with httpx.AsyncClient(base_url=A2A_URL) as httpx_client:
        client = to_a2a(httpx_client, AGENT_CARD)
        all_results: List[Dict] = []
        for model in MINISTRAL_MODELS:
            print(f"\n🔹 Benchmarking {model} …")
            # Set environment variable for LiteLLM (used internally by the agent)
            os.environ["LITELLM_MODEL"] = model
            metrics = await evaluate_model(model, client)
            all_results.append(metrics)
            print(f"✅ {model} metrics: {metrics['metrics']}")

    # -------------------------------------------------------------------
    # Persist results
    # -------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📁 Results written to {RESULT_JSON}")

    # -------------------------------------------------------------------
    # Simple visualisation (mean reward bar chart)
    # -------------------------------------------------------------------
    models = [r["model"] for r in all_results]
    mean_rewards = [r["metrics"].get("mean_reward", 0) for r in all_results]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(models, mean_rewards, color=["#4c6ef5", "#22c55e", "#ef4444"])
    plt.title("Mean Reward per Ministral Cloud Model")
    plt.ylabel("Mean Reward")
    plt.ylim(min(0, min(mean_rewards) - 1), max(mean_rewards) + 1)
    for bar, val in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(PLOT_PNG)
    print(f"📊 Plot saved to {PLOT_PNG}")

if __name__ == "__main__":
    asyncio.run(main())
