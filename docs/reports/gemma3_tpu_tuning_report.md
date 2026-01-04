# Tunix & Gemma 3: A TPU Fine-Tuning Retrospective

**Date:** December 2025
**Model:** Gemma 3 1B-it
**Framework:** Tunix (JAX/Flax) on Kaggle TPU v5e-8

## 1. Executive Summary
This report documents the successful adaptation of Google's Tunix framework to fine-tune Gemma 3 for the DIPG Safety benchmarks. Unlike standard GPU workflows (like Unsloth), TPU training requires specific "Hardware Heuristics" focused on compilation and fixed-shape tensors rather than dynamic quantization. We successfully stabilized training, implemented custom loss masking, and achieved a **20% Safety Response Rate** with strong context grounding.

## 2. The Challenge: TPU vs. GPU Heuristics

Most open-source fine-tuning wisdom comes from the GPU world (Unsloth, bitsandbytes). We found that these heuristics do not directly translate to TPUs.

| Feature | GPU Strategy (e.g., Unsloth) | TPU Strategy (Tunix) | Why? |
| :--- | :--- | :--- | :--- |
| **Quantization** | 4-bit / 8-bit (QLoRA) is standard to save VRAM. | **bfloat16 (16-bit)** is preferred. | TPUs have high HBM and specific matrix units optimized for bf16 XLA operations. 4-bit dequantization overhead can be costly on TPU. |
| **Batching** | maximize batch size dynamically. | **Fixed "Micro Batches"**. | TPUs use XLA (Accelerated Linear Algebra) which compiles the computation graph. Dynamic shapes trigger expensive re-compilation. Fixed, small micro-batches with accumulation are essential. |
| **Steps vs. Epochs** | Often count raw "Steps" (e.g., 500 steps). | Count **Epochs**. | Data loaders on TPU are sharded. We cycle the dataset multiple times (e.g., 13 epochs) to achieve equivalent training volume. |

## 3. Critical Technical Interventions

We encountered and solved three major blockers:

### A. The "Masking Hijack" (SFT Correctness)
**Problem:** The default Tunix data pipeline did not correctly apply loss masking to the user instructions, risking the model "learning to speak like a user" rather than a helpful assistant.
**Solution:** We implemented a custom `grain` transformation in `gemma_data_pipeline.py` that pre-calculates the mask (0 for prompt, 1 for response) and injects it into the `input_mask` field, effectively "hijacking" the field Tunix expects for padding to serve as a loss mask.

### B. OOM (Resource Exhausted) on TPU
**Problem:** Initial attempts with `MAX_SEQ_LENGTH=2048` and default batch sizes caused immediate Out Of Memory errors.
**Solution:** The "Fit to Tile" Strategy.
1.  **Micro Batch = 2:** Reduced per-device batch to the absolute minimum.
2.  **Gradient Accumulation = 8:** Compensated for the small batch by accumulating gradients over 8 steps, yielding an effective global batch of 128 (2 * 8 chips * 8 accumulation).
3.  **Seq Len = 1024:** Halved sequence length, which allows for significantly larger batch throughput without sacrificing relevant context for safety prompts.

### C. Stability (The Step 80 Crash)
**Problem:** Training crashed reliably around step 80 due to memory spikes from intermediate evaluation or checkpointing running concurrently with training loops.
**Solution:** We disabled intermediate hooks (`eval_every_n_steps=1000`) and implemented a robust **Post-Training Manual Save** using `orbax.checkpoint`. This isolated the memory-intensive saving operation from the training loop.

## 4. Final Configuration

```python
# Production Config for Kaggle TPU v5e-8
MAX_SEQ_LENGTH = 1024
TRAIN_MICRO_BATCH_SIZE = 2      # Per-chip batch
GRADIENT_ACCUMULATION_STEPS = 8 # Global Batch = 128
MAX_STEPS = 300                 # ~27 Epochs for 1400 examples
LEARNING_RATE = 2e-5            # Standard LoRA rate
```

## 5. Results

The model was evaluated against the DIPG Safety Gym protocol:

*   **Baseline (Pre-Tuning):** 0% Safety Rate (Frequent refusals or hallucinations).
*   **Post-Tuning (300 Steps):** **20% Safe Response Rate**.
*   **Grounding:** The model successfully learned to refuse out-of-domain queries (e.g., "What is the capital of France?" -> "Context does not contain this information"), demonstrating that the SFT process effectively taught it to respect the RAG context window.

## 6. Recommendations
For future iterations:
1.  **Scale Up:** Increase `MAX_STEPS` to 500-1000 and `MAX_SEQ_LENGTH` to 2048 if higher memory TPUs (v4/v5p) become available.
2.  **Dataset:** Expand the 1400-example safety dataset to include more diversity, which will help generalize the 20% score.
