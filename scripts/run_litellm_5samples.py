"""Test with 5 samples to see variety of rewards"""
from examples.eval_with_litellm import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator(
    litellm_url="http://localhost:8080",
    dipg_url="http://localhost:8000",
    model_name="ollama/gpt-oss:120b-cloud"
)

print("Testing with 5 samples to see reward distribution...")
results = orchestrator.run_evaluation(max_samples=5)

# Show reward breakdown
print("\n📊 Reward Breakdown:")
for i, reward in enumerate(results['rewards'], 1):
    print(f"  Sample {i}: {reward:.2f}")
