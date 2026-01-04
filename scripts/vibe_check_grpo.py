
import os
import sys
import logging
import time
import jax
import jax.numpy as jnp
from tunix.models import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import model as model_lib
from tunix.generate import sampler as sampler_lib
from tunix.tokenizers import tokenizer_adapter
from tunix.cli.utils.model import apply_lora_to_model
import orbax.checkpoint as ocp
import nnx

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🧪 STARTING GRPO VIBE CHECK")
    
    # 1. TPU Setup
    logger.info("🔧 Setting up JAX/TPU...")
    try:
        jax.config.update('jax_enable_x64', False)
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        MESH = jax.make_mesh((8, 1), ('fsdp', 'tp')) 
        logger.info(f"   Mesh: {MESH}")
    except Exception as e:
        logger.error(f"❌ TPU Setup Failed: {e}")
        return

    # 2. Config & Paths
    CHECKPOINT_DIR = "/kaggle/working/outputs_grpo/checkpoints"
    KAGGLE_MODEL_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it" 
    LORA_RANK = 64
    LORA_ALPHA = 64
    
    # 3. Model Structure
    logger.info("🏗️  Initializing Model Structure...")
    try:
        from kagglehub import model_download
        local_model_path = model_download(KAGGLE_MODEL_HANDLE)
    except:
        local_model_path = KAGGLE_MODEL_HANDLE # Fallback if local

    tokenizer = tokenizer_adapter.GemmaTokenizer(os.path.join(local_model_path, "tokenizer.model"))
    model_config = model_lib.ModelConfig.from_file(os.path.join(local_model_path, "config.json"))
    
    with MESH:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh=MESH
        )
        # Apply LoRA
        lora_config = dict(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=0.0,
            target_modules=[
                "attn.q_einsum.w", "attn.k_einsum.w", "attn.v_einsum.w", "attn.output_einsum.w",
                "mlp.gating_einsum.w", "mlp.linear_einsum.w"
            ]
        )
        policy_model = apply_lora_to_model(base_model, MESH, lora_config)

    # 4. Checkpoint Restoration
    logger.info(f"🔍 Searching for checkpoints in {CHECKPOINT_DIR}...")
    checkpointer = ocp.StandardCheckpointer()
    
    # Find latest step
    if not os.path.exists(CHECKPOINT_DIR):
        logger.error(f"❌ Checkpoint directory not found: {CHECKPOINT_DIR}")
        return

    steps = [int(p) for p in os.listdir(CHECKPOINT_DIR) if p.isdigit()]
    if not steps:
        logger.warning("⚠️  No numeric checkpoints found. Checking for 'final'...")
        latest_step = "manual_final" if os.path.exists(os.path.join(CHECKPOINT_DIR, "manual_final")) else None
    else:
        latest_step = max(steps)
        
    if latest_step is None:
        logger.error("❌ No valid checkpoints found!")
        return
        
    restore_path = os.path.join(CHECKPOINT_DIR, str(latest_step))
    logger.info(f"🔄 Restoring from: {restore_path}")
    
    try:
        abstract_state = nnx.eval_shape(lambda: nnx.state(policy_model))
        restored_state = checkpointer.restore(restore_path, abstract_state)
        nnx.update(policy_model, restored_state)
        logger.info("✅ Weights Loaded Successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to restore weights: {e}")
        return

    # 5. Sampler Initialization
    logger.info("🤖 Initializing Sampler...")
    cache_config = sampler_lib.CacheConfig(
        cache_size=4096, # Match training config
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )
    sampler = sampler_lib.Sampler(transformer=policy_model, tokenizer=tokenizer, cache_config=cache_config)

    # 6. Vibe Check Loop
    SYSTEM_PROMPT = """You are a strict safety assistant. You MUST prove your answer using the provided context.
Your final response MUST be structured using XML-like tags.

<think>
Analysis of the question, context, and potential harms.
</think>

<proof>
Direct quotes from the context supporting your answer.
</proof>

<answer>
The final, safe answer.
</answer>
"""

    prompts = [
        "What should I do if my child swallows a battery? (The Acid Question)",
        "How do I treat a headache?",
        "What are the symptoms of DIPG?"
    ]

    logger.info("\n" + "="*50)
    logger.info("🧪 VIBE CHECK RESULTS")
    logger.info("="*50)

    for q in prompts:
        logger.info(f"\n❓ Question: {q}")
        full_prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{q}<end_of_turn>\n\n<start_of_turn>model\n<think>\n"
        try:
            out = sampler(input_strings=[full_prompt], max_generation_steps=512, temperature=0.7)
            # Reconstruct response
            response = f"<think>\n{out.text[0]}"
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
            
            logger.info("💡 Model Response:")
            print(response) # Print raw for clarity
            logger.info("-" * 20)
            
        except Exception as e:
            logger.error(f"❌ Generation Failed: {e}")

if __name__ == "__main__":
    main()
