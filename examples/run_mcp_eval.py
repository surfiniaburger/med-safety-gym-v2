import os
import sys
import json
import asyncio
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

async def run_evaluation(model, tokenizer, num_samples=100):
    print("📊 Running Post-SFT Benchmark via MCP Server...")

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

    print(f"Starting MCP server with dataset: {eval_env['DIPG_DATASET_PATH']}")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 2. Fetch Tasks
                print(f"Fetching {num_samples} tasks...")
                result = await session.call_tool("get_eval_tasks", arguments={"max_samples": num_samples})
                tasks_data = json.loads(result.content[0].text)
                tasks = tasks_data["tasks"]
                print(f"✅ Retrieved {len(tasks)} tasks.")
                
                # 3. Generate Responses
                print("Generating responses...")
                evaluations = []
                
                # Enable inference mode for unsloth model if available
                try:
                    from unsloth import FastLanguageModel
                    FastLanguageModel.for_inference(model)
                except ImportError:
                    pass
                except NameError:
                    pass
                
                for i, task in enumerate(tasks):
                    if i % 10 == 0: print(f"  Processing {i}/{len(tasks)}...")
                    
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
                        print(f"Warning: Assistant start token not found in output for task {i}.")
                    
                    evaluations.append({
                        "response": response_text,
                        "ground_truth": {
                            "context": task["context"],
                            "question": task["question"],
                            "expected_answer": task["expected_answer"]
                        }
                    })
                
                # 4. Evaluate
                print("Evaluating responses...")
                eval_result = await session.call_tool("evaluate_batch", arguments={"evaluations": evaluations})
                metrics = json.loads(eval_result.content[0].text)
                return metrics

    except Exception as e:
        print(f"\n❌ Error during MCP evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # This script is designed to be run where 'model' and 'tokenizer' are already defined
    # or to be imported.
    # If run standalone, it would need to load the model.
    print("This script is intended to be run within the notebook or where 'model' is available.")
    print("If running standalone, please uncomment the model loading section.")
    
    # Example standalone usage (commented out):
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    # metrics = asyncio.run(run_evaluation(model, tokenizer, num_samples=10))

if __name__ == "__main__":
    main()
