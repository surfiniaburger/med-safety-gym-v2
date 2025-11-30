# ==================================================================================
# Colab Evaluation with Cloud Run
# ==================================================================================
# This cell shows how to use the deployed Cloud Run server for evaluation

import os

# Set your Cloud Run URL (get this from deployment output)
# Example: https://dipg-server-xxxxx-uc.a.run.app
os.environ["DIPG_SERVER_URL"] = "YOUR_CLOUD_RUN_URL_HERE"

# Import the evaluation function
from examples.run_eval_colab import run_evaluation_http

# Run evaluation
metrics = run_evaluation_http(
    model=model,
    tokenizer=tokenizer,
    num_samples=100
)

# Results will be displayed automatically
print("\n" + "="*60)
print("📊 EVALUATION RESULTS")
print("="*60)
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Safe Response Rate: {metrics['safe_response_rate']:.2%}")
print(f"Hallucination Rate: {metrics['hallucination_rate']:.2%}")
print("="*60)

# ==================================================================================
# Alternative: Direct URL specification
# ==================================================================================
# If you don't want to use environment variables:

metrics = run_evaluation_http(
    model=model,
    tokenizer=tokenizer,
    num_samples=100,
    server_url="https://dipg-server-xxxxx-uc.a.run.app"  # Your Cloud Run URL
)
