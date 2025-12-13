# 📚 DIPG Safety Gym: The Tutorials

Welcome to the "Zero to Hero" guide for the DIPG Safety Gym.

## 🟢 Level 1: Foundations
*Start here if you are new.*
1.  [**What is the Safety Gym?**](tutorials/01_foundations/01_what_is_dipg_gym.md) - The "Safe by Default" philosophy.
2.  [**Installation Guide**](tutorials/01_foundations/02_installation_guide.md) - Get it running on your machine.
3.  [**Hello World**](tutorials/01_foundations/03_hello_world.md) - Your first script.

## 🟡 Level 2: The Universal API
*For researchers who just want to test a model.*
1.  [**The Universal API Concept**](tutorials/02_simple_eval/01_universal_api_concept.md) - Why we use HTTP.
2.  [**Running Your First Evaluation**](tutorials/02_simple_eval/02_running_your_first_eval.md) - Code walkthrough.
3.  [**Understanding Metrics**](tutorials/02_simple_eval/03_understanding_metrics.md) - Hallucination Rate, Safe Rate, etc.
4.  [**Visualizing Results**](tutorials/02_simple_eval/04_visualizing_results.md) - Charts and graphs.
5.  [**External Models (LiteLLM)**](examples/eval_with_litellm.py) - Example script for connecting to LiteLLM.

## 🟠 Level 3: Environment Logic
*For engineers building agents.*
1.  [**Anatomy of a Task**](tutorials/03_environment_logic/01_anatomy_of_a_task.md) - JSON breakdown.
2.  [**Strict Format Curriculum**](tutorials/03_environment_logic/02_strict_format_curriculum.md) - The 3 Gates of Safety.
3.  [**Supported Formats**](tutorials/03_environment_logic/03_supported_formats.md) - JSON vs XML vs Custom Tags.

## 🔴 Level 4: Advanced Architecture
*For system architects.*
1.  [**The Hybrid Model (A2A + MCP)**](tutorials/04_architecture/01_the_hybrid_model.md) - Separation of Brain and Body.
2.  [**Using MCP Tools**](tutorials/04_architecture/02_using_mcp_tools.md) - `get_eval_tasks` and `evaluate_batch`.
3.  [**Connecting an Agent**](tutorials/04_architecture/03_connecting_an_agent.md) - ADK configuration.
4.  [**How-to: Run an MCP Eval**](tutorials/04_architecture/04_running_mcp_eval.md) - Python script walkthrough.

## ⚫ Level 5: Expert Customization
*For power users.*
1.  [**How We Generate Data**](tutorials/05_data_and_customization/01_how_we_generate_data.md) - Reasoning, Refusal, and Traps.
2.  [**Using Custom Datasets**](tutorials/05_data_and_customization/02_using_custom_datasets.md) - How to use your own JSONL.
3.  [**Process Supervision Deep Dive**](tutorials/05_data_and_customization/03_process_supervision.md) - How the Proof verifier works.
4.  [**Data Generation Scripts**](tutorials/05_data_and_customization/04_data_generation_scripts.md) - Guide to `scripts/`.

## 📘 Reference Concepts
*Deep dives into specific topics.*
1.  [**Evaluation Pathways: REST vs MCP**](concepts/evaluation_pathways_explained.md) - Choosing the right tool for the job.
2.  [**Server Architectures**](concepts/architecture_server_implementations.md) - FastMCP vs Standard API.
3.  [**Data Generation Architecture**](concepts/data_generation_architecture.md) - OSS vs Gemini pipelines.
4.  [**MCP & Agent Usage Guide**](MCP_GUIDE.md) - Reference for FastMCP, A2A, and Tools.
