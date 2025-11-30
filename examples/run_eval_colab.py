"""
Colab-Compatible Evaluation Script

This script uses the HTTP API instead of MCP stdio client,
which is compatible with Jupyter/Colab notebooks.
"""

import os
import sys
import json
import logging
import requests
import matplotlib.pyplot as plt
from IPython.display import display, Image
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default server URL - can be overridden by environment variable
DEFAULT_SERVER_URL = os.getenv("DIPG_SERVER_URL", "http://localhost:8000")

# Import visualization utilities
try:
    from scripts.visualizations import save_all_visualizations
except ImportError:
    logger.warning("⚠️ Could not import visualizations module")
    save_all_visualizations = None


def run_evaluation_http(model, tokenizer, num_samples=100, server_url=None):
    """
    Run evaluation using HTTP API instead of MCP stdio.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        num_samples: Number of evaluation samples
        server_url: URL of the DIPG server (default: from DIPG_SERVER_URL env var or http://localhost:8000)
    
    Returns:
        dict: Evaluation metrics
    """
    # Use provided URL or fall back to default
    if server_url is None:
        server_url = DEFAULT_SERVER_URL
    
    logger.info("📊 Running Post-SFT Benchmark via HTTP API...")
    logger.info(f"Server URL: {server_url}")
    
    # 1. Load evaluation dataset directly
    logger.info(f"Loading evaluation dataset...")
    try:
        eval_dataset = load_dataset("surfiniaburger/dipg-eval-dataset", split="train")
        logger.info(f"✅ Loaded {len(eval_dataset)} evaluation samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return None
    
    # Limit to num_samples
    tasks = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    logger.info(f"Using {len(tasks)} samples for evaluation")
    
    # 2. Generate Responses
    logger.info("Generating responses...")
    evaluations = []
    
    # Enable inference mode for unsloth model if available
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        logger.info("✅ Enabled inference mode for unsloth model")
    except (ImportError, NameError):
        logger.warning("⚠️ Could not enable unsloth inference mode")
    
    for i, task in enumerate(tasks):
        if i % 10 == 0:
            logger.info(f"  Processing {i}/{len(tasks)}...")
        
        # Create prompt from dataset
        context = task.get("context", "")
        question = task.get("question", "")
        expected_answer = task.get("expected_answer", "")
        
        # Create messages for chat template
        messages = [{"role": "user", "content": context + "\n\n" + question}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Extract response (handle different chat templates)
        response_text = decoded[0]
        
        # Try to extract assistant response
        for marker in ["<start_of_turn>model\n", "<|start|>assistant<|message|>", "assistant\n"]:
            if marker in response_text:
                parts = response_text.split(marker)
                if len(parts) > 1:
                    response_text = parts[-1].split("<end_of_turn>")[0].split("<|end|>")[0].strip()
                    break
        
        evaluations.append({
            "response": response_text,
            "ground_truth": {
                "context": context,
                "question": question,
                "expected_answer": expected_answer
            }
        })
        
        # Log first 5 samples
        if i < 5:
            logger.info(f"\n{'='*60}")
            logger.info(f"SAMPLE OUTPUT #{i}")
            logger.info(f"{'='*60}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"Response Length: {len(response_text)} chars")
            logger.info(f"Response Preview: {response_text[:200]}...")
            logger.info(f"{'='*60}\n")
    
    # 3. Evaluate via HTTP API
    logger.info(f"Evaluating {len(evaluations)} responses via HTTP API...")
    try:
        response = requests.post(
            f"{server_url}/evaluate",
            json={
                "evaluations": evaluations,
                "format": "json"
            },
            timeout=300
        )
        response.raise_for_status()
        metrics = response.json()
        logger.info("✅ Evaluation complete")
        
        # 4. Display Results
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Mean Reward: {metrics.get('mean_reward', 'N/A'):.2f}")
        logger.info(f"Safe Response Rate: {metrics.get('safe_response_rate', 'N/A'):.2%}")
        logger.info(f"Hallucination Rate: {metrics.get('hallucination_rate', 'N/A'):.2%}")
        logger.info("="*60 + "\n")
        
        # 5. Generate Visualizations
        if save_all_visualizations:
            try:
                logger.info("Generating visualizations...")
                viz_dir = save_all_visualizations(metrics, output_dir="benchmark_results")
                logger.info(f"✅ Visualizations saved to: {viz_dir}")
                
                # Display visualizations in notebook
                for viz_file in ["reward_distribution.png", "metrics_overview.png"]:
                    viz_path = os.path.join(viz_dir, viz_file)
                    if os.path.exists(viz_path):
                        display(Image(filename=viz_path))
            except Exception as e:
                logger.warning(f"⚠️ Could not generate visualizations: {e}")
        
        return metrics
        
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Could not connect to server at {server_url}")
        logger.error("💡 Make sure the DIPG server is running:")
        logger.error(f"   uvicorn server.app:app --host 0.0.0.0 --port 8000")
        return None
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Example usage when model and tokenizer are available.
    """
    logger.info("This script is intended to be run within a notebook where 'model' and 'tokenizer' are defined.")
    logger.info("\nExample usage:")
    logger.info("  metrics = run_evaluation_http(model, tokenizer, num_samples=100)")


if __name__ == "__main__":
    main()
