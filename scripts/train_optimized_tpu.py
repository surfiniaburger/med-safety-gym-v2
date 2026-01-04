# ==============================================================================
# GEMMA 3 TPU TRAINING: GOLDEN CONFIGURATION 🧪
# ==============================================================================
# "From Alchemy to Chemistry" - The Parameters that Worked.
# ==============================================================================

import os
import gc
import sys
import shutil
import traceback
import optax
import jax
import numpy as np
import grain.python as grain
from tunix import PeftTrainer, TrainingConfig
from tunix.sft import utils as sft_utils

# ------------------------------------------------------------------------------
# 1. HYPERPARAMETERS (The "Science")
# ------------------------------------------------------------------------------
# Why these values?
# - MAX_SEQ_LENGTH = 1024: Prevents OOM on TPU v5e-8 (2048 causes crash).
# - TRAIN_MICRO_BATCH_SIZE = 2: Fits in per-chip HBM (16GB).
# - GRADIENT_ACCUMULATION_STEPS = 8: 
#      Global Batch = 2 * 8 (chips) * 8 (accum) = 128.
#      High batch size stabilizes the gradients for safety tuning.
# - MAX_STEPS = 300: 
#      For 1400 examples with Batch 128, this is ~27 Epochs.
#      Necessary to "burn in" the refusal behavior (Safety First).
# ------------------------------------------------------------------------------

MAX_SEQ_LENGTH = 1024
TRAIN_MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = 300
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# ------------------------------------------------------------------------------
# 2. DATA PIPELINE (The "Masking Hijack")
# ------------------------------------------------------------------------------
# Embedded here to ensure standalone execution.

def tokenize_and_mask(example, tokenizer, max_seq_len):
    """
    Tokenizes the input and CREATES A LOSS MASK so we don't train on User Prompts.
    
    Structure:
    [User Tokens] -> Mask = 0.0 (Do not learn)
    [Model Tokens] -> Mask = 1.0 (Learn this!)
    """
    
    # 1. Tokenize Parts
    # Note: Adjust logic if your dataset structure differs (e.g. 'messages' list)
    # This assumes 'prompt' and 'completion' or 'instruction' fields.
    # Fallback to standard "formatted_train" list of dicts structure:
    # {'input_tokens': ..., 'input_mask': ...} if already pre-processed.
    
    # If using raw text (preferred for Grain):
    # prompt = example['prompt']
    # completion = example['completion']
    # But for this script, we assume the dataset object passed to loader 
    # provides pre-tokenized arrays OR raw strings.
    # Let's assume standard Tunix inputs for safety:
    
    # ... Implementation Detail: Since we don't have the raw dataset shape here,
    # we assume the user's notebook has 'dataset' which is a HuggingFace dataset
    # or list of dicts.
    
    # SIMPLIFIED LOADER for Kaggle Notebook Context:
    # We rely on 'input_ids' being present or we tokenize on the fly.
    
    pass # Real logic is handled by grain map below

def create_grain_loader(dataset_rows, tokenizer, max_seq_len, batch_size, shuffle=True):
    """
    Creates a robust, Infinite-Repeating Data Loader for TPUs.
    CRITICAL: Adds .repeat() to prevent 'Step 87 Stop'.
    """
    
    # 1. Define Transformation (Tokenization + Masking)
    def _process_fn(ex):
        # Extract Text
        # Handle 'messages' format (common in SFT)
        if 'messages' in ex:
            # Simple concatenation for Tunix prompt/response
            user_text = ex['messages'][0]['content'] # User
            model_text = ex['messages'][1]['content'] # Assistant
        else:
            # Fallback
            user_text = ex.get('prompt', '') or ex.get('instruction', '')
            model_text = ex.get('completion', '') or ex.get('response', '')
            
        # Tokenize (using model's tokenizer)
        # Note: We use the global 'tokenizer' object passed in
        user_tokens = tokenizer.encode(user_text, add_bos=True, add_eos=False)
        model_tokens = tokenizer.encode(model_text, add_bos=False, add_eos=True)
        
        # Concatenate
        full_tokens = user_tokens + model_tokens
        
        # Create Loss Weights (0 for User, 1 for Model)
        # We start with all zeros (masked)
        weights = [0.0] * len(user_tokens) + [1.0] * len(model_tokens)
        
        # Truncate / Pad
        if len(full_tokens) > max_seq_len:
            full_tokens = full_tokens[:max_seq_len]
            weights = weights[:max_seq_len]
        else:
            pad_len = max_seq_len - len(full_tokens)
            full_tokens += [0] * pad_len # Pad token ID (usually 0 or tokenizer.pad_token_id)
            weights += [0.0] * pad_len   # Mask padding
            
        return {
            'input_tokens': np.array(full_tokens, dtype=np.int32),
            'input_mask': np.array(weights, dtype=np.float32) # HIJACKED FIELD for Tunix
        }

    # 2. Build Pipeline
    # Convert HF Dataset or List to Grain Source
    if hasattr(dataset_rows, 'to_list'):
        source_data = dataset_rows.to_list()
    else:
        source_data = list(dataset_rows) # Assume list-like
        
    loader = grain.python.MapDataset.source(source_data)
    
    if shuffle:
        loader = loader.shuffle(seed=42)
        
    loader = loader.map(_process_fn)
    
    # CRITICAL FIX: Repeat indefinitely so we hit MAX_STEPS (300)
    # Without this, it stops at Step 87 (1 epoch).
    loader = loader.repeat() 
    
    loader = loader.batch(batch_size=batch_size, drop_remainder=True)
    return loader


# ------------------------------------------------------------------------------
# 3. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
CHECKPOINT_DIR = "/kaggle/working/outputs_sft_final/checkpoints"
LOG_FILE = "training_error.log"

# Clean Slate: Wipe old checkpoints to prevent shape mismatches
if os.path.exists(CHECKPOINT_DIR):
    print(f"🧹 Cleaning old checkpoints at {CHECKPOINT_DIR}...")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

# Free Memory
gc.collect()

print(f"🚀 STARTING OPTIMIZED RUN")
print(f"👉 Configuration: SeqLen={MAX_SEQ_LENGTH}, Batch={TRAIN_MICRO_BATCH_SIZE}x8x{GRADIENT_ACCUMULATION_STEPS}, Steps={MAX_STEPS}")
print(f"👉 Safety Mode: Error Logging Enabled")

# ------------------------------------------------------------------------------
# 4. ROBUST TRAINING LOOP
# ------------------------------------------------------------------------------
try:
    # A. Configuration
    training_config = TrainingConfig(
        max_steps=MAX_STEPS,
        # CRITICAL: Disable intermediate eval to prevent memory spikes (The "Step 80 Crash")
        eval_every_n_steps=1000, 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        checkpoint_root_directory=CHECKPOINT_DIR,
    )

    # B. Optimizer (Cosine Decay)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value=LEARNING_RATE, 
        warmup_steps=int(MAX_STEPS * 0.1), # 10% Warmup
        decay_steps=MAX_STEPS, 
        end_value=LEARNING_RATE * 0.1,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.scale_by_adam(b1=0.9, b2=0.999),
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    # C. Data & Model
    # Note: Assumes `tokenizer` and `gemma3_model` and `dataset['train']` are defined in notebook
    train_loader = create_grain_loader(
        dataset['train'], 
        tokenizer, 
        MAX_SEQ_LENGTH, 
        TRAIN_MICRO_BATCH_SIZE, 
        shuffle=True
    )
    
    # D. Trainer Initialization
    trainer = PeftTrainer(model=gemma3_model, optimizer=optimizer, training_config=training_config)
    
    # E. Input Pipeline (The "Masking Hijack" for Correctness)
    trainer = trainer.with_gen_model_input_fn(lambda x: {
        'input_tokens': x['input_tokens'],
        'input_mask': x['input_mask'], # Passing pre-calculated loss mask here
        'positions': sft_utils.build_positions_from_mask(x['input_tokens'] != 0),
        'attention_mask': sft_utils.make_causal_attn_mask(x['input_tokens'] != 0),
    })

    # F. Execute
    print("🔥 outputting logs... (this may take a few minutes for XLA compile)")
    trainer.train(train_loader)
    
    print("✅ Training Complete!")
    
    # G. Manual Save (Reliability)
    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    state = nnx.state(gemma3_model)
    save_path = os.path.join(CHECKPOINT_DIR, f"golden_step_{MAX_STEPS}")
    checkpointer.save(save_path, state)
    print(f"📦 Model Saved to: {save_path}")

except Exception as e:
    print("\n❌ TRAINING CRASHED!")
    print(f"Error: {str(e)}")
    with open(LOG_FILE, "w") as f:
        traceback.print_exc(file=f)
    print(f"🔍 Traceback saved to {LOG_FILE}.")
