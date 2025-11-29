import json
import glob
import os

def refactor_notebooks():
    notebook_path = "examples/dipg-rl-with-benchmarks-01.ipynb"
    print(f"Refactoring {notebook_path}...")
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    setup_updated = False
    eval_updated = False
    
    # New Evaluation Code with Robust Parsing
    eval_code = [
        "import os\n",
        "import sys\n",
        "import json\n",
        "import asyncio\n",
        "import nest_asyncio\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import display, Image\n",
        "from mcp import ClientSession, StdioServerParameters\n",
        "from mcp.client.stdio import stdio_client\n",
        "# Ensure we can import scripts from root\n",
        "sys.path.append(os.getcwd())\n",
        "sys.path.append(os.path.abspath('..'))\n",
        "from scripts.visualizations import save_all_visualizations\n",
        "\n",
        "# Apply nest_asyncio to allow async in notebook\n",
        "nest_asyncio.apply()\n",
        "\n",
        "print(\"📊 Running Post-SFT Benchmark via MCP Server...\")\n",
        "\n",
        "# 1. Setup MCP Server Parameters (Pointing to EVAL dataset)\n",
        "eval_env = os.environ.copy()\n",
        "eval_env[\"DIPG_DATASET_PATH\"] = \"surfiniaburger/dipg-eval-dataset\"\n",
        "\n",
        "server_params = StdioServerParameters(\n",
        "    command=sys.executable,\n",
        "    args=[\"-m\", \"server.mcp_server\"],\n",
        "    env=eval_env\n",
        ")\n",
        "\n",
        "async def run_evaluation(num_samples=100):\n",
        "    print(f\"Starting MCP server with dataset: {eval_env['DIPG_DATASET_PATH']}\")\n",
        "    \n",
        "    async with stdio_client(server_params) as (read, write):\n",
        "        async with ClientSession(read, write) as session:\n",
        "            await session.initialize()\n",
        "            \n",
        "            # 2. Fetch Tasks\n",
        "            print(f\"Fetching {num_samples} tasks...\")\n",
        "            result = await session.call_tool(\"get_eval_tasks\", arguments={\"max_samples\": num_samples})\n",
        "            tasks_data = json.loads(result.content[0].text)\n",
        "            tasks = tasks_data[\"tasks\"]\n",
        "            print(f\"✅ Retrieved {len(tasks)} tasks.\")\n",
        "            \n",
        "            # 3. Generate Responses\n",
        "            print(\"Generating responses...\")\n",
        "            evaluations = []\n",
        "            \n",
        "            # Enable inference mode for unsloth model\n",
        "            try:\n",
        "                FastLanguageModel.for_inference(model)\n",
        "            except NameError:\n",
        "                print(\"⚠️ 'model' not defined. Assuming testing mode or model loaded elsewhere.\")\n",
        "            \n",
        "            for i, task in enumerate(tasks):\n",
        "                if i % 10 == 0: print(f\"  Processing {i}/{len(tasks)}...\")\n",
        "                \n",
        "                # Create prompt\n",
        "                messages = [{\"role\": \"user\", \"content\": task[\"context\"] + \"\\n\\n\" + task[\"question\"]}]\n",
        "                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=\"pt\").to(\"cuda\")\n",
        "                \n",
        "                # Generate\n",
        "                outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, pad_token_id=tokenizer.eos_token_id)\n",
        "                decoded = tokenizer.batch_decode(outputs)\n",
        "                \n",
        "                # Extract response, handling cases where the template might be missing.\n",
        "                assistant_token = \"<|start|>assistant<|message|>\"\n",
        "                response_parts = decoded[0].split(assistant_token)\n",
        "                if len(response_parts) > 1:\n",
        "                    response_text = response_parts[-1].replace(\"<|end|>\", \"\").strip()\n",
        "                else:\n",
        "                    response_text = \"\" # Or handle as an error\n",
        "                    print(f\"Warning: Assistant start token not found in output for task {i}.\")\n",
        "                \n",
        "                evaluations.append({\n",
        "                    \"response\": response_text,\n",
        "                    \"ground_truth\": {\n",
        "                        \"context\": task[\"context\"],\n",
        "                        \"question\": task[\"question\"],\n",
        "                        \"expected_answer\": task[\"expected_answer\"]\n",
        "                    }\n",
        "                })\n",
        "            \n",
        "            # 4. Evaluate\n",
        "            print(\"Evaluating responses...\")\n",
        "            eval_result = await session.call_tool(\"evaluate_batch\", arguments={\"evaluations\": evaluations})\n",
        "            metrics = json.loads(eval_result.content[0].text)\n",
        "            return metrics\n",
        "\n",
        "# Run the evaluation\n",
        "metrics = asyncio.run(run_evaluation(num_samples=100))\n",
        "\n",
        "# 5. Display Results\n",
        "print(\"\\n\" + \"=\"*40)\n",
        "print(\"BENCHMARK RESULTS\")\n",
        "print(\"=\"*40)\n",
        "print(f\"Mean Reward: {metrics['mean_reward']:.2f}\")\n",
        "print(f\"Safe Response Rate: {metrics['safe_response_rate']:.1%}\")\n",
        "print(f\"Hallucination Rate: {metrics['medical_hallucination_rate']:.1%}\")\n",
        "\n",
        "# 6. Generate Visualizations\n",
        "output_dir = \"benchmark_results_sft\"\n",
        "saved_files = save_all_visualizations(metrics, output_dir, \"SFT_Model\")\n",
        "\n",
        "print(f\"\\nVisualizations saved to {output_dir}/\")\n",
        "for file in saved_files:\n",
        "    display(Image(filename=file))\n"
    ]

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # 1. Replace Server Setup Script
            if "Server Setup Script for Google Colab" in source and "subprocess.run" in source:
                cell["source"] = [
                    "# Run the server setup script\n",
                    "!python scripts/setup_env.py"
                ]
                setup_updated = True
                print("  ✅ Replaced server setup script with call to scripts/setup_env.py")

            # 2. Update Evaluation Cell with Robust Parsing
            if "server.mcp_server" in source and "export DIPG_DATASET_PATH" in source:
                # This check might fail if I already replaced it with python code.
                # Let's check for the python code signature instead.
                pass
            
            if "StdioServerParameters" in source and "run_evaluation" in source:
                 cell["source"] = eval_code
                 eval_updated = True
                 print("  ✅ Updated evaluation cell with robust parsing")
    
    if setup_updated or eval_updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  💾 Saved changes to {notebook_path}")
    else:
        print(f"  ⚠️ No changes needed for {notebook_path}")

if __name__ == "__main__":
    refactor_notebooks()
