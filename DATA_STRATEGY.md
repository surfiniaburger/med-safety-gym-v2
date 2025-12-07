# DIPG Safety Gym: Data Strategy & Milestones
> **Goal**: Build a "Gold Standard" medical safety dataset for SFT and GRPO, moving from simple templates to high-fidelity, verifiable clinical reasoning.

## 🎯 Strategic Objectives 
1.  **Foundational Alignment (SFT)**: Teach the "Safety Prime" – both *what* to answer (groundedness) and *when* to abstain (safety).
2.  **Policy Refinement (GRPO)**: Use Reinforcement Learning to incentivize reasoning and strict adherence to safety constraints.
3.  **MedGR² Framework**: Use Generative Reward Learning to overcome data scarcity.

---

## 🗺️ Milestones

### Milestone 1: Schema & Format Standardization
**Objective**: Define the exact data structure for SFT and GRPO to ensure compatibility with Unsloth and modern reasoning models.

*   **Decision**: Move from JSON-in-string to **XML-style Reasoning Tags** (e.g., `<think>`, `<evidence>`, `<answer>`) for the SFT target. This aligns better with Chain-of-Thought (CoT) training and GRPO.
*   **Format**: Standard **ShareGPT** (multi-turn friendly).
*   **Deliverable**: `DATA_SCHEMA.md` defining the exact JSONL structure.

### Milestone 2: High-Fidelity Synthetic Data Pipeline ("MedGR² Lite")
**Objective**: Replace the current random slot-filling generator with a semantic generation pipeline using a high-quality "Teacher" Model.

*   **Action**: Created `scripts/generate_med_data.py`.
*   **Status**: ✅ Implemented & Verified.
*   **Models**:
    *   **Primary**: `gpt-oss:120b-cloud` (Selected ✅).
    *   **Challenger**: `qwen3-coder:480b-cloud` (Verified).
*   **Method**:
    1.  **Seed**: Use real medical concepts.
    2.  **Generate**: Prompt Teacher Model to create complex clinical vignettes with `<think>` and `<proof>` tags.
    3.  **Synthesize Negatives**: Explicitly generate "trap" questions.
*   **Deliverable**: Robust generation script + `DATA_GENERATION_REPORT.md` (GPT-OSS selected).

### "MedGR² Lite": The Synthetic Data Pipeline Flow

The `scripts/generate_med_data.py` script implements the "MedGR² Lite" pipeline, a sophisticated process for generating high-fidelity synthetic data. This pipeline uses a powerful "teacher" model (e.g., `gpt-oss:120b-cloud`) to create a diverse and challenging dataset for training and evaluating the safety of medical AI agents.

The flow consists of the following steps:

1.  **Initialization**: A `MedicalDataGenerator` is initialized with a specified teacher model. This model is responsible for generating the content of the synthetic data.

2.  **Prompt Engineering**: The script employs a variety of specialized prompts to generate different types of training examples. This ensures the dataset covers a wide range of scenarios, including:
    *   **Positive Reasoning**: Generates complex clinical vignettes that require multi-step reasoning to arrive at a correct answer.
    *   **Negative Refusal**: Creates "trap" scenarios where critical information is missing or ambiguous, forcing the model to learn to safely abstain from answering.
    *   **Needle in a Haystack**: Generates a specific, verifiable fact that is later hidden within a large volume of irrelevant text, testing the model's ability to extract information from noisy contexts.
    *   **Anti-Knowledge**: Creates scenarios where the provided context is entirely irrelevant to the question, testing the model's ability to recognize when it cannot answer a question based on the given information.

3.  **Concurrent Generation**: The script uses asynchronous programming to generate multiple examples concurrently, significantly speeding up the data generation process. For each example, it sends the appropriate prompt to the teacher model and requests a response in a structured JSON format.

4.  **ShareGPT Formatting**: The raw JSON output from the teacher model is then transformed into the **ShareGPT** format, which is a standard for training conversational AI models. This involves:
    *   Structuring the data into a series of "user" and "assistant" messages.
    *   Wrapping the reasoning steps, evidence, and final answer in XML-style tags (e.g., `<think>`, `<proof>`, `<answer>`).
    *   For "Needle in a Haystack" examples, injecting the specific fact (the "needle") into a large block of irrelevant medical axioms (the "haystack").

5.  **Output to JSONL**: The formatted examples are collected, shuffled to ensure a random distribution of example types, and then written to a JSONL file. Each line in the file represents a complete training example, ready to be used for Supervised Fine-Tuning (SFT) or other training methods.

This pipeline provides a robust and flexible way to generate high-quality, synthetic data that is essential for training and evaluating the safety and reliability of medical AI agents. By using a powerful teacher model and a variety of prompt engineering techniques, the "MedGR² Lite" pipeline produces a dataset that is far more realistic and challenging than what can be achieved with simple template-based methods.


### Milestone 3: SFT Dataset Generation (The "Base" Corpus)
**Objective**: Generate the dataset for the initial Supervised Fine-Tuning.

*   **Composition**:
    *   **50% Positive Reasoning**: Complex questions $\rightarrow$ Correct CoT $\rightarrow$ Correct Answer.
    *   **40% Refusal/Abstention**: "Trap" questions $\rightarrow$ CoT identifying missing info $\rightarrow$ Refusal.
    *   **10% General Safety**: General medical safety constraints.
*   **Target Size**: ~1k - 5k high-quality examples (quality > quantity).
*   **Deliverable**: `datasets/dipg_sft_v1.jsonl`.

### Milestone 4: GRPO Reward Definition & Dataset
**Objective**: Prepare the environment for Group Relative Policy Optimization.

*   **Dataset**: A set of *prompts only* (questions + context) without the target answers (the model generates them during RL).
*   **Reward Functions**:
    1.  **Format Reward**: Strict adherence to `<think>` tags.
    2.  **Groundedness Reward**: Checking if cited facts exist in context (using a lightweight evaluator or string matching).
    3.  **Safety Reward**: Penalty for answering "trap" questions.
*   **Deliverable**: `datasets/dipg_grpo_prompts.jsonl` and `server/rewards.py`.

### Milestone 5: The "Green Agent" Benchmark (AgentX-AgentBeats)
**Objective**: Formalize the evaluation set as a standalone benchmark.

*   **Action**: Select a held-out set of "Hard" and "Trap" questions.
*   **Implementation**: Package this as an A2A "Green Agent" that:
    1.  Sends a task.
    2.  Receives the response.
    3.  Scores it using the Reward Model.
*   **Deliverable**: A deployable Docker container acting as the Evaluation Benchmark.

---

## 🛠️ Immediate Next Steps (Execution)

1.  **Refactor `sharegpt_data_generator.py`** to use the new **XML-Tag Schema** and **ShareGPT** format.
2.  **Implement the "Teacher" generation logic** (using LiteLLM to call a strong model) instead of random strings.


# model to use
unsloth/medgemma-4b-it-unsloth-bnb-4bit