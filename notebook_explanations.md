# Med-Safety-Gym: SFT and GRPO Training Notebook Guide

This notebook is divided into two main parts, following a state-of-the-art training strategy:

1.  **Supervised Fine-Tuning (SFT)**: First, we teach the base model the "language" of our task—how to structure its responses and reason about medical questions.
2.  **Group Relative Policy Optimization (GRPO)**: Then, we use reinforcement learning to train the model for *safety*. We reward good behavior (like refusing to answer when unsure) and penalize bad behavior (like making things up).

Let's begin.

---

## Part 1: Supervised Fine-Tuning (SFT) - Teaching the Model the Rules

The goal of this first phase is to take a general-purpose model and make it a specialist. We're not focused on maximizing safety yet. Instead, we want to teach the model how to follow our very specific instructions and output format. This provides a solid foundation for the safety training that comes next.

### Cell: Environment Setup

```python
%%capture
!pip install "google-tunix[prod]==0.1.3"
!pip install wandb
!pip install uv
!uv pip install --system openenv-dipg-safety
```

Here, we install all the necessary libraries.
-   `tunix`: Google's library for training large models on TPUs.
-   `wandb`: A tool for logging our experiments so we can track progress.
-   `openenv-dipg-safety`: Our custom medical safety environment, which contains the dataset and evaluation logic.

### Cell: Weights & Biases Login

```python
import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("wandb_api_key")
wandb.login(key=wandb_api_key)
```

This cell securely logs us into Weights & Biases using a Kaggle secret. This allows us to save and visualize our training results without exposing our private API key.

### Cell: TPU Hardware Check

```python
import jax
# ... (TPU check code) ...
```

This is a quick but important sanity check. We confirm that our notebook is running on a TPU (Tensor Processing Unit). TPUs are specialized hardware from Google that can train large models like Gemma 3 much, much faster than traditional CPUs or GPUs. If this check fails, training would be too slow to be practical.

### Cell: SFT Training Configuration

```python
KAGGLE_MODEL_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 64
LEARNING_RATE = 2e-5
# ... (other configs) ...
```

This is where we set the "knobs" for our SFT training run.

-   **`KAGGLE_MODEL_HANDLE`**: We're using `gemma-3-1b-it`, a powerful and efficient 1-billion-parameter model from Google. The "it" means it's already been instruction-tuned, making it a great starting point.
-   **`MAX_SEQ_LENGTH`**: This is the maximum amount of text (in tokens) the model can handle at once. We set it to 1024 to balance detail with memory capacity.
-   **`LORA_RANK`**: We use LoRA (Low-Rank Adaptation), a clever technique that freezes most of the model and only trains a tiny fraction of its parameters. This makes training dramatically faster and more memory-efficient. A rank of 64 provides a good balance between training speed and model quality.

### Cell: Dataset Formatting and System Prompt

```python
SYSTEM_PROMPT = """You are a strict safety assistant..."""
def format_dipg_example(ex):
    # ... (formatting code) ...
```

This is the most critical part of the SFT phase. We are defining the exact rules the model must follow.

The **`SYSTEM_PROMPT`** is the model's prime directive. It strictly defines its persona and the required output structure. We enforce a three-part XML format:

1.  **`<think>`...`</think>`**: This forces the model to "show its work." It must first write down its reasoning process. This gives us a window into its "mind" and is crucial for debugging and ensuring it's on the right track.

2.  **`<proof>`...`</proof>`**: This is our primary defense against hallucination. The model **must** copy-paste direct quotes from the provided context to support its answer. If it cannot find a supporting quote, it is not allowed to answer.

3.  **`<answer>`...`</answer>`**: The final, conclusive answer, which should be based *only* on the information in the `<proof>` tag.

By training the model on examples formatted this way, we teach it to be structured, transparent, and evidence-based from the very beginning.

### Cell: Data Pipeline and Token Masking

```python
def tokenize_and_mask(ex, tokenizer, max_length):
    # ... (tokenizing and masking code) ...
```

This cell sets up the data pipeline that feeds examples to the model. It also contains a clever trick for efficient training.

When we train, we want the model to learn to generate the *assistant's response*, not the user's question. The `tokenize_and_mask` function handles this by creating a "loss mask." This mask tells the training algorithm to ignore the user's part of the text and only calculate the learning error on the model's own output. In simple terms, we're only grading the model on its answer, not the question it was given.

### Cell: Running the SFT Trainer

```python
trainer.train(train_loader)
```

This is where the magic happens. We launch the `PeftTrainer`, which feeds the formatted data to the LoRA-adapted model on the TPU. The model will see thousands of examples and learn to mimic the desired XML format and reasoning structure. After this step, we will have a model that is specialized for our task and ready for safety tuning.

### Cell: Saving the SFT Model

```python
checkpointer.save(save_path, state)
```

Finally, we save the fine-tuned model's LoRA weights. This checkpoint is the starting point for our next, most important phase: GRPO reinforcement learning.

---

## Part 2: GRPO Reinforcement Learning - Making the Model Safe

Now that our model understands the *format* of the task, we will use reinforcement learning to teach it *good behavior*. The GRPO process will reward the model for being safe and helpful, and penalize it for being dangerous or making things up. The configurations here are based on the key findings from our training report.

### Cell: GRPO Configuration

```python
MAX_STEPS = 300
NUM_GENERATIONS = 4
BETA = 0.08
# ... (other configs) ...
```

We adjust our configuration for reinforcement learning:

-   **`NUM_GENERATIONS = 4`**: In GRPO, the model generates multiple possible answers for each prompt (in this case, 4). It then internally compares them to see which ones lead to better rewards. Our training report found that using 4 generations was more stable than 2, giving the model enough variety to learn robustly.
-   **`BETA = 0.08`**: This parameter acts as a safety tether. It prevents the policy model (the one we're training) from straying too far from the original SFT model we just built. This encourages stable learning and prevents the model from "forgetting" its initial training.

### Cell: The Reward Function - The Heart of Safety

```python
class DIPGRaxReward:
    def __init__(self):
        self.env = DIPGEnvironment(
            # ... (reward values) ...
        )
```

This class is the heart of our entire safety system. It acts as the "judge" that scores every single one of the model's responses. Based on the extensive experiments documented in our training report, we engineered a "high-stakes, high-reward" system with carefully tuned penalties.

#### **Positive Rewards (The Carrots)** 🥕

We heavily incentivize good behavior:

-   **`correct_abstention_reward = +30.0`**: This is our largest reward. We give the model a huge bonus for correctly identifying when an answer is not in the context and safely refusing to answer. This is the single most important behavior for preventing harmful, made-up advice.
-   **`correct_synthesis_reward = +20.0`** and **`verifiable_trace_reward = +15.0`**: We give significant points for providing the right answer and backing it up with a valid, verifiable proof.
-   **`no_hallucination_reward = +5.0`**: We give a small but consistent bonus for every response that is free of hallucination.

#### **Negative Penalties (The Sticks)** 몽둥이

As our training report revealed, the penalty values are critical. **Prematurely harsh penalties caused the model to stop answering questions entirely.** The key to our success was using "soft" initial penalties:

-   **`hallucination_penalty = -5.0`** and **`hallucinated_trace_penalty = -10.0`**: These are our soft penalties for making things up. They are just punishing enough to discourage hallucination, but not so severe that they scare the model away from attempting to answer at all. This balance was essential for allowing the model to learn and explore, ultimately leading to our **88% safety rate**.
-   **`format_mismatch_penalty = -10.0`**: We keep a stricter penalty for failing to use the XML format, as the model should have already mastered this during the SFT phase.

### Cell: Model Loading with Checkpoint Logic

```python
if os.path.exists(GRPO_CHECKPOINT):
    RESUME_PATH = GRPO_CHECKPOINT
elif os.path.exists(SFT_CHECKPOINT):
    RESUME_PATH = SFT_CHECKPOINT
# ... (restore code) ...
```

This logic creates our two-stage pipeline. It first looks for a GRPO checkpoint to continue a previous reinforcement learning run. If it doesn't find one, it loads the weights from the SFT model we trained in Part 1. This ensures we are always building upon our previous work, starting the RL phase with a model that already understands the task's structure.

### Cell: Running the GRPO Trainer

```python
grpo_trainer.train(dataset)
```

This command kicks off the reinforcement learning loop. For each step:
1.  The model (the "actor") generates 4 responses to a prompt.
2.  Our `DIPGRaxReward` function scores each response.
3.  The `GRPOLearner` analyzes the rewards and updates the model's weights, encouraging it to produce responses that will earn higher scores in the future.

This loop, repeated for 300 steps, is what refines the model's behavior and aligns it with our safety goals.

### Cell: Final Evaluation and Model Save

```python
metrics = evaluate_dipg_model(sampler, 50)
# ... (save code) ...
```

After the GRPO training is complete, we run a final evaluation against 50 unseen test questions. This is where we measure our final success metrics, such as the **88% safe response rate** and **4% hallucination rate** documented in the report.

Finally, we save the fully trained model. This `grpo_900` checkpoint represents our best and final model, optimized for both accuracy and safety.
