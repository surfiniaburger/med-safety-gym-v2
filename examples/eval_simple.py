"""
Simple Evaluation Script - Works on ANY Platform

This script provides a clean, minimal evaluation flow that works on:
- Google Colab
- Kaggle
- Local Jupyter
- Any Python environment with HTTP access

No complex imports needed - just requests and your model!
"""

import requests

# Configuration
SERVER_URL = "https://dipg-server-5hurbdoigq-uc.a.run.app"  # Cloud Run
# SERVER_URL = "http://localhost:8000"  # Local Docker

def evaluate_model(model, tokenizer, num_samples=100, server_url=SERVER_URL):
    """
    Evaluate a model using the simple 3-step flow.
    
    Args:
        model: Your model (any framework)
        tokenizer: Your tokenizer
        num_samples: Number of samples to evaluate
        server_url: DIPG server URL (Cloud Run or local Docker)
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Step 1: Get tasks from med_safety_gym
    print(f"📥 Fetching {num_samples} evaluation tasks...")
    response = requests.get(f"{server_url}/tasks", params={"count": num_samples})
    response.raise_for_status()
    tasks = response.json()["tasks"]
    print(f"✅ Received {len(tasks)} tasks")
    
    # Step 2: Generate responses locally
    print(f"🤖 Generating responses...")
    responses = []
    
    for i, task in enumerate(tasks):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(tasks)}")
        
        # Create prompt (customize this for your model)
        prompt = f"{task['context']}\n\n{task['question']}"
        
        # Generate response (customize this for your model)
        # Example for transformers:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        responses.append({
            "task_id": task["task_id"],
            "response": response_text
        })
    
    print(f"✅ Generated {len(responses)} responses")
    
    # Step 3: Submit for evaluation
    print(f"📊 Evaluating responses...")
    eval_response = requests.post(
        f"{server_url}/evaluate/tasks",
        json={
            "responses": responses,
            "format": "json"  # or "custom_tags" depending on your training
        }
    )
    eval_response.raise_for_status()
    metrics = eval_response.json()["metrics"]
    
    # Display results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    def safe_format(val, fmt=".2f"):
        try:
            return f"{float(val):{fmt}}"
        except (ValueError, TypeError):
            return str(val)

    print(f"Mean Reward: {safe_format(metrics.get('mean_reward', 'N/A'))}")
    print(f"Safe Response Rate: {safe_format(metrics.get('safe_response_rate', 'N/A'), '.2%')}")
    print(f"Hallucination Rate: {safe_format(metrics.get('hallucination_rate', 'N/A'), '.2%')}")
    print("="*60 + "\n")
    
    # Generate visualizations if matplotlib is available
    try:
        plot_results(metrics)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
        print("Install matplotlib to see visualizations: pip install matplotlib")
    
    return metrics


def plot_results(metrics):
    """
    Generate simple visualizations for the evaluation results.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️ Matplotlib not installed. Skipping visualizations.")
        return

    # 1. Metrics Bar Chart
    plt.figure(figsize=(10, 5))
    
    # Extract key metrics (coupled keys and labels for maintainability)
    metrics_to_plot = {
        'safe_response_rate': {'label': 'Safe Response', 'color': 'green'},
        'hallucination_rate': {'label': 'Hallucination', 'color': 'red'},
        'refusal_rate': {'label': 'Refusal', 'color': 'orange'},
        'reasoning_consistency_rate': {'label': 'Consistency', 'color': 'blue'}
    }
    labels = [v['label'] for v in metrics_to_plot.values()]
    values = [metrics.get(k, 0) for k in metrics_to_plot.keys()]
    
    colors = [v['color'] for v in metrics_to_plot.values()]
    bars = plt.bar(labels, values, color=colors, alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')
    
    plt.title('Safety Metrics Overview')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 2. Reward Distribution (if available)
    if 'rewards' in metrics and metrics['rewards']:
        plt.figure(figsize=(10, 5))
        rewards = metrics['rewards']
        
        plt.hist(rewards, bins=20, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        
        plt.title('Reward Distribution')
        plt.xlabel('Reward Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


# Example usage in Colab/Kaggle:
# Example usage (standalone test):
if __name__ == "__main__":
    print("⚠️ Running in STANDALONE simulation mode.")
    print("   (Actual model evaluation requires importing this script)")
    
    # Mock Model & Tokenizer for demonstration
    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            return {} # Dummy
        def decode(self, *args, **kwargs):
            return ""

    class MockModel:
        def generate(self, **kwargs):
            return [""] # Dummy output

    # Override generation loop for simulation
    # We will patch the evaluate_model function locally for this run only
    # to return valid dummy responses without a real model.
    
    import random
    
    def simulate_generation(prompt):
        # Generate a semi-valid dummy response to test the API
        return {
            "think": "<think>I need to analyze the patient data.</think>",
            "proof": "<proof>Patient has H3K27M mutation</proof>",
            "answer": "<answer>The diagnosis is DIPG.</answer>"
        }

    # Monkey-patch the loop in a real scenario, or just duplicate logic for the test?
    # Better: Update evaluate_model to handle "mock" mode? 
    # Let's just create a test wrapper here.
    
    print("   Connecting to Dev Server: http://localhost:8000")
    # Using local server for verification
    TEST_SERVER = "http://localhost:8000" 
    
    try:
        # Step 1: Fetch
        print("📥 [Mock] Fetching tasks...")
        res = requests.get(f"{TEST_SERVER}/tasks", params={"count": 5})
        res.raise_for_status()
        tasks = res.json()["tasks"]
        
        # Step 2: Generate (Simulated)
        responses = []
        for t in tasks:
            # We generate a valid-looking XML response to pass the server format check
            # This proves the PIPELINE works, even if the "model" is dumb.
            sim_content = (
                "<think>\nAnalysis of context...\n</think>\n\n"
                "<proof>\nQuote from text\n</proof>\n\n"
                "<answer>\nSimulated Answer\n</answer>"
            )
            responses.append({"task_id": t["task_id"], "response": sim_content})
            
        # Step 3: Submit
        print("📤 [Mock] Submitting responses...")
        eval_res = requests.post(f"{TEST_SERVER}/evaluate/tasks", json={"responses": responses})
        eval_res.raise_for_status()
        print("✅ SUCCESS: Pipeline Verified!")
        print(eval_res.json()["metrics"])
        
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to localhost:8000.")
        print("   Please make sure the DIPG Server is running locally via Docker.")
        exit(1)
    except Exception as e:
        print(f"❌ FAILED: {e}")
        exit(1)
