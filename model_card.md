# Model Card: Gemma-3-1B-IT for DIPG Medical Safety Reasoning

This model card documents the `Gemma-3-1B-IT` model fine-tuned for the Google Tunix Hackathon. The model is trained to generate safe and verifiable reasoning traces for questions related to Diffuse Intrinsic Pontine Glioma (DIPG).

# Model Summary

This is a `Gemma-3-1B-IT` model fine-tuned using a two-phase process: Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO) with the Tunix library. The model is specialized for the `DIPG-Safety-Gym` environment, a safety-critical medical reasoning task.

The model's key characteristic is its ability to "show its work" by generating a response in a structured XML format, including a `<think>` process, a `<proof>` section with direct quotes from the source context, and a final `<answer>`. The training process focused heavily on a penalty annealing curriculum to maximize safety and minimize hallucinations.

The final model (at 900 GRPO steps) achieves an **88% safe response rate** and a **4% medical hallucination rate** on the evaluation dataset, a significant improvement from its post-SFT baseline of 30% safety and 50% hallucination.

## Usage

This model is intended to be used for generating structured, verifiable reasoning for medical questions *within the context of the DIPG-Safety-Gym environment*. It should be prompted with a specific system message and a user query containing both context and a question.

**Code Snippet for Inference (using Tunix):**

```python
import os
import jax
import kagglehub
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.generate import sampler as sampler_lib
from flax import nnx

# 1. Load the trained model from Kaggle Models
# (Replace with your final model handle)
KAGGLE_MODEL_HANDLE = "[YOUR-KAGGLE-USERNAME]/gemma-3-1b-tunix-grpo/jax/dipg-safety-900steps"
local_model_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)

# 2. Setup Model and Tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=os.path.join(local_model_path, "tokenizer.model")
)
model_config = gemma_lib.ModelConfig.gemma3_1b()
mesh = jax.make_mesh((1,), ('fsdp',)) # Simple mesh for inference

with mesh:
    # Create the model structure (including LoRA adapters if used)
    model = params_safetensors_lib.create_model_from_safe_tensors(
        local_model_path, model_config, mesh=mesh
    )

# 3. Create a Sampler
cache_config = sampler_lib.CacheConfig(
    cache_size=4096,
    num_layers=model_config.num_layers,
    num_kv_heads=model_config.num_kv_heads,
    head_dim=model_config.head_dim,
)
sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

# 4. Define Input Shape and Prompt
SYSTEM_PROMPT = "You are a strict safety assistant..." # Use the full system prompt from training
CONTEXT = "[Paste the medical context here]"
QUESTION = "[Ask the question here]"

# Input is a single string with a specific structure
prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n<context>{CONTEXT}</context>\n<question>{QUESTION}</question><end_of_turn>\n<start_of_turn>model\n"

# 5. Generate a Response
output = sampler(
    input_strings=[prompt],
    max_generation_steps=512,
    temperature=0.7 # Use temperature > 0 for generation
)

# Output is a string with the model's structured reasoning
final_response = output.text[0]
print(final_response)
```

**Known Failures:**
The model's primary failure mode is generating an incorrect or irrelevant answer when the provided context is ambiguous or does not contain the necessary information. While trained to abstain, it can still fail to do so. It should not be used for any real-world medical decision-making.

## System

This is a standalone model but is designed to be part of a larger human-in-the-loop system.

*   **Input Requirements:** The model requires a structured prompt that includes a system message defining the task, and the user query formatted with `<context>` and `<question>` tags.
*   **Downstream Dependencies:** A downstream system would need an XML parser to extract the content of the `<think>`, `<proof>`, and `<answer>` tags. The `<proof>` section is intended to be used for automated or human verification against the original context.

## Implementation requirements

*   **Hardware:** All training was conducted on a single Kaggle TPU v5e-8 environment.
*   **Software:** The training stack was built on JAX, using the `tunix` library for the GRPO implementation, `Flax` for model representation, and `optax` for optimization.
*   **Compute Requirements:**
    *   **Training Time:** Each 300-step training block took approximately 1 hour. The final 900-step model represents about **3-4 hours** of total training time on a TPU v5e-8.
    *   **Inference:** Inference is efficient on a single TPU core, with responses generated in a few seconds.

# Model Characteristics

## Model initialization

The model was **fine-tuned** from the official pre-trained `google/gemma-3-1b-it` model. The SFT and GRPO phases both started from these weights.

## Model stats

*   **Size:** 1 Billion parameters.
*   **Architecture:** It is a decoder-only Transformer based on the Gemma 3 architecture.
*   **Weights:** The final model consists of the base model weights plus trained LoRA adapters for the attention and MLP layers.
*   **Latency:** Latency was not formally benchmarked, but is low enough for interactive use cases when running on a TPU.

## Other details

The model is **not pruned** and **not quantized**. No differential privacy techniques were used during training.

# Data Overview

## Training data

*   **Dataset:** `surfiniaburger/dipg-safety-instruction-1500` on Hugging Face.
*   **Description:** The dataset contains 1,500 examples. Each example consists of a medical `context` (a clinical vignette about DIPG), a `question` related to the context, and a ground-truth `answer` that includes a reasoning trace.
*   **Collection & Pre-processing:** The data was synthetically generated to cover a range of scenarios, including questions that can be answered directly, questions requiring synthesis, and questions that *cannot* be answered from the context (to train abstention). The `scripts/train_grpo_tpu.py` script performs the final pre-processing step, which involves assembling the context and question into the final prompt structure with the system message.

## Demographic groups

The data pertains to clinical vignettes of pediatric patients with DIPG. No other specific demographic data (e.g., race, gender) is present in the dataset or was used for analysis. The focus is on medical safety rather than demographic fairness.

## Evaluation data

*   **Dataset:** `surfiniaburger/med-safety-gym-eval` on Hugging Face.
*   **Split:** The training process used the entire 1,500-example training set. Evaluation was performed against a separate, held-out set of 50 examples hosted by the `openenv-dipg-safety` environment server. There was no traditional train/validation/test split of the training data itself.

# Evaluation Results

## Summary

The key finding of our evaluation is that a patient penalty curriculum during GRPO training is critical for achieving both safety and performance. Our best model (trained for 900 steps with a soft-penalty recovery phase) achieved:

*   **Mean Reward:** +0.58 (the first positive mean reward achieved)
*   **Safe Response Rate:** 88%
*   **Medical Hallucination Rate:** 4%
*   **Reasoning Consistency:** 88%

These results are a dramatic improvement over the post-SFT baseline (30% safety, 50% hallucination) and the failed "medium penalty" model (which had a max reward of only 1.0). For a full narrative and block-by-block results, please see our `kaggle_writeup.md`.

## Subgroup evaluation results

Our primary subgroup analysis was the evaluation of the model at different stages of the training curriculum (300, 600, 900, and 1200 steps). This analysis revealed the "penalty cliff": when penalties were increased prematurely at step 600, the model's performance collapsed, and it refused to generate correct answers. Reverting to soft penalties allowed the model to recover and achieve its peak performance at 900 steps. This demonstrates a key, preventable failure mode in training: **aggressive penalty scheduling is counterproductive.**

## Fairness

For this safety-critical task, we define fairness as the model's ability to apply its safety protocols (groundedness, abstention, consistency) uniformly across all questions in the evaluation set, regardless of their complexity or type. The final model's high safety (88%) and consistency (88%) rates suggest it behaves reliably across the diverse evaluation scenarios.

## Usage limitations

*   **CRITICAL: This model is a research prototype and MUST NOT be used for real-world medical advice or in any clinical or diagnostic capacity.**
*   The model is highly specialized for the DIPG-Safety-Gym task and prompt format. It is not expected to generalize to other medical domains or different question formats without further fine-tuning.
*   Performance is dependent on the quality and clarity of the provided context.

## Ethics

*   **Ethical Considerations:** The primary ethical consideration was the minimization of harm in a safety-critical domain. The entire training process was designed to create a model that is fundamentally cautious. We aimed to build a model that would rather abstain from answering than provide a potentially harmful, incorrect, or hallucinated response.
*   **Risks Identified:** The foremost risk is misuse of the model as a real medical authority. Despite training, the model can still make mistakes (4% hallucination rate).
*   **Mitigations:**
    1.  **Training for Abstention:** The reward function explicitly incentivizes correct abstention, teaching the model to say "I don't know" when appropriate.
    2.  **Verifiable Reasoning:** The required `<proof>` tag forces the model to provide direct evidence for its claims, allowing a human user to quickly verify the source of the information and build trust.
    3.  **Explicit Disclaimers:** This model card and other documentation clearly state that the model is for research purposes only and not for medical use.
