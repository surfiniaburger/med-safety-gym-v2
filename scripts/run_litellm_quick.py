"""Quick test with 3 samples using local Ollama model"""
from examples.eval_with_litellm import EvaluationOrchestrator

# Use local Ollama model with proper LiteLLM format
orchestrator = EvaluationOrchestrator(
    litellm_url="http://localhost:8080",
    dipg_url="http://localhost:8000",
    model_name="ollama/gpt-oss:120b-cloud"  # LiteLLM requires "ollama/" prefix
)

# Run on just 2 samples for quick test
print("Testing with 2 samples using Ollama gpt-oss:120b-cloud...")
results = orchestrator.run_evaluation(max_samples=2)
