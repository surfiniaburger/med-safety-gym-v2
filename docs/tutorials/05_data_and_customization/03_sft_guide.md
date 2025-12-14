# SFT Strategy & Masking Guide

This guide explains how to fine-tune Gemma-3 (or similar models) on the DIPG Safety Gym dataset using the Dual-Stream Architecture (DSA).

## 1. Dataset Format
We use a **Dual-Stream** format that explicitly separates internal reasoning from the final answer. This is critical for safety training, as it allows us to monitor the model's "thought process" before it outputs a potentially harmful or incorrect answer.

### Tags
- `<think>...</think>`: The model's internal monologue, planning, and safety checks.
- `<proof>...</proof>`: Verifications, citations, or step-by-step evidence.
- `<answer>...</answer>`: The final, user-facing response.

### Example Structure
```xml
<start_of_turn>user
You are a dual-stream safety assistant...

[User Question]
<end_of_turn>
<start_of_turn>model
<think>
Plan: Check safety guidelines...
</think>
<proof>
Verified against protocols...
</proof>
<answer>
Safe response...
</answer>
<end_of_turn>
```

## 2. Masking Strategy
In Supervised Fine-Tuning (SFT), we only want to train the model on *its own output*, not the user's instructions.

### How it works
The `dsa-sft-grpo-nolora-tunix` notebook (and Unsloth generally) uses a masking technique that sets the loss to `0` for all tokens belonging to the user prompt.

- **User Prompt**: `<start_of_turn>user ... <end_of_turn>` -> **MASKED (Loss = 0)**
- **Model Output**: `<start_of_turn>model ... <end_of_turn>` -> **TRAINED (Loss = 1)**

### Implementation
Our adaptation script (`scripts/adapt_dsa_notebook.py`) formats the data exactly how the notebook expects:
1. It constructs the full conversation string.
2. The notebook's built-in `tokenize_function` detects the `<start_of_turn>model` token.
3. It automatically creates a `loss_mask` that zeros out everything before that token.

**Result**: The model learns to generate the `<think>`, `<proof>`, and `<answer>` blocks given the user prompt, but isn't penalized for "predicting" the user prompt itself.

## 3. Training Workflow
1. **Upload Data**: We split the 1500 samples into `train` (1400) and `test` (100) and uploaded them to Hugging Face (`surfiniaburger/dipg-safety-instruction-1500`).
2. **Adapt Notebook**: Use the code in `scripts/adapt_dsa_notebook.py` to replace the GSM8K-specific loading code in the Tunix notebook.
3. **Train**: Run the SFT process. The model will align to the DSA format.
4. **Evaluate**: Use `scripts/eval_simple.py` or the notebook's inference cell to check if the model consistently produces the `<think>` tags.
