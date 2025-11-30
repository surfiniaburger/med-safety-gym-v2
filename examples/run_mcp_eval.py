import os
import sys
import json
import asyncio
import logging
import nest_asyncio
import matplotlib.pyplot as plt
from IPython.display import display, Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure we can import scripts from root
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('..'))

try:
    from scripts.visualizations import save_all_visualizations
except ImportError:
    # Fallback if running from root
    from scripts.visualizations import save_all_visualizations

# Apply nest_asyncio to allow async in notebook/script
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def run_evaluation(model, tokenizer, num_samples=100):
    logger.info("📊 Running Post-SFT Benchmark via MCP Server...")

    # 1. Setup MCP Server Parameters (Pointing to EVAL dataset)
    eval_env = os.environ.copy()
    eval_env["DIPG_DATASET_PATH"] = "surfiniaburger/dipg-eval-dataset"
    
    # Ensure PYTHONPATH includes the current directory so server modules can be found
    if "PYTHONPATH" in eval_env:
        eval_env["PYTHONPATH"] += f":{os.getcwd()}"
    else:
        eval_env["PYTHONPATH"] = os.getcwd()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.mcp_server"],
        env=eval_env
    )

    logger.info(f"Starting MCP server with dataset: {eval_env['DIPG_DATASET_PATH']}")
    logger.info(f"Server command: {sys.executable} -m server.mcp_server")
    logger.info(f"PYTHONPATH: {eval_env.get('PYTHONPATH', 'not set')}")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("Initializing MCP session...")
                await session.initialize()
                logger.info("✅ MCP session initialized successfully")
                
                # 2. Fetch Tasks
                logger.info(f"Fetching {num_samples} tasks...")
                result = await session.call_tool("get_eval_tasks", arguments={"max_samples": num_samples})
                tasks_data = json.loads(result.content[0].text)
                tasks = tasks_data["tasks"]
                logger.info(f"✅ Retrieved {len(tasks)} tasks.")
                
                # 3. Generate Responses
                logger.info("Generating responses...")
                evaluations = []
                
                # Enable inference mode for unsloth model if available
                try:
                    from unsloth import FastLanguageModel
                    FastLanguageModel.for_inference(model)
                    logger.info("✅ Enabled inference mode for unsloth model")
                except ImportError:
                    logger.warning("⚠️ unsloth not available, skipping inference mode optimization")
                except NameError:
                    logger.warning("⚠️ 'model' not defined, skipping inference mode optimization")
                
                for i, task in enumerate(tasks):
                    if i % 10 == 0: 
                        logger.info(f"  Processing {i}/{len(tasks)}...")
                    
                    # Create prompt
                    messages = [{"role": "user", "content": task["context"] + "\n\n" + task["question"]}]
                    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
                    
                    # Generate
                    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    decoded = tokenizer.batch_decode(outputs)
                    
                    # Extract response, handling cases where the template might be missing.
                    assistant_token = "<|start|>assistant<|message|>"
                    response_parts = decoded[0].split(assistant_token)
                    if len(response_parts) > 1:
                        response_text = response_parts[-1].replace("<|end|>", "").strip()
                    else:
                        response_text = "" # Or handle as an error
                        logger.warning(f"⚠️ Assistant start token not found in output for task {i}")
                    
                    evaluations.append({
                        "response": response_text,
                        "ground_truth": {
                            "context": task["context"],
                            "question": task["question"],
                            "expected_answer": task["expected_answer"]
                        }
                    })
                    
                    # Log first 5 samples for inspection
                    if i < 5:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"SAMPLE OUTPUT #{i}")
                        logger.info(f"{'='*60}")
                        logger.info(f"Question: {task['question'][:100]}...")
                        logger.info(f"Raw Output Length: {len(decoded[0])} chars")
                        logger.info(f"Extracted Response Length: {len(response_text)} chars")
                        logger.info(f"Response Preview: {response_text[:200]}...")
                        logger.info(f"{'='*60}\n")
                
                # 4. Evaluate with JSON format
                logger.info("Evaluating responses with JSON format...")
                eval_result = await session.call_tool("evaluate_batch", arguments={
                    "evaluations": evaluations,
                    "format": "json"
                })
                metrics = json.loads(eval_result.content[0].text)
                logger.info("✅ Evaluation complete")
                return metrics

    except Exception as e:
        logger.error(f"\n❌ Error during MCP evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Additional diagnostics
        logger.error("\n🔍 Diagnostics:")
        logger.error(f"  - Current directory: {os.getcwd()}")
        logger.error(f"  - Python executable: {sys.executable}")
        logger.error(f"  - Dataset path: {eval_env.get('DIPG_DATASET_PATH')}")
        logger.error(f"  - PYTHONPATH: {eval_env.get('PYTHONPATH')}")
        
        # Try to run the server manually to see the error
        logger.error("\n🔧 Attempting to run server manually for diagnostics...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "server.mcp_server"],
            env=eval_env,
            capture_output=True,
            text=True,
            timeout=5,
            input=""
        )
        if result.returncode != 0:
            logger.error(f"Server stderr: {result.stderr}")
            logger.error(f"Server stdout: {result.stdout}")
        
        return None

def main():
    # This script is designed to be run where 'model' and 'tokenizer' are already defined
    # or to be imported.
    # If run standalone, it would need to load the model.
    logger.info("This script is intended to be run within the notebook or where 'model' is available.")
    logger.info("If running standalone, please uncomment the model loading section.")
    
    # Example standalone usage (commented out):
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    # metrics = asyncio.run(run_evaluation(model, tokenizer, num_samples=10))

if __name__ == "__main__":
    main()
