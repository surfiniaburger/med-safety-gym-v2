import os
import dataclasses
import jax
import flax.nnx as nnx
import kagglehub
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.generate import sampler as sampler_lib
from tunix.cli.utils.model import apply_lora_to_model
from orbax import checkpoint as ocp

# ==============================================================================
# 0. THE MANDATORY TUNIX NNX PATCH
# ==============================================================================
_orig_set_metadata = nnx.Variable.set_metadata
def _compat_set_metadata(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], str):
        kwargs[args[0]] = args[1]
        return _orig_set_metadata(self, **kwargs)
    return _orig_set_metadata(self, *args, **kwargs)
nnx.Variable.set_metadata = _compat_set_metadata

# --- 1. CRITICAL: Ghost Buster Mapper (V6) ---
# MedGemma uses 'language_model.model' naming and is multimodal.
# V6 patches the mapper DIRECTLY in the loader for compatibility with Tunix 0.1.3.
from tunix.models import safetensors_loader
import re

def ghost_buster_key_mapper(mapping, source_key):
    if "vision" in source_key.lower():
        return f"unused.vision.{source_key}", None
    norm_key = source_key
    if source_key.startswith("language_model.model."):
        norm_key = source_key.replace("language_model.model.", "model.")
    elif source_key.startswith("language_model."):
        norm_key = source_key.replace("language_model.", "model.")
    try:
        subs = [
            (re.sub(pat, repl, norm_key), reshape)
            for pat, (repl, reshape) in mapping.items()
            if re.match(pat, norm_key) or re.match(pat, source_key)
        ]
        if len(subs) == 1: return subs[0]
    except Exception: pass
    return f"unused.unknown.{source_key}", None

safetensors_loader.torch_key_to_jax_key = ghost_buster_key_mapper
print("✅ Ghost Buster (V9) active: Mapping redirection enabled.")

# --- 2. Configuration ---
LORA_RANK = 64
LORA_ALPHA = 64
MESH = jax.make_mesh((jax.device_count(), 1), ('fsdp', 'tp')) 

# --- 2. Paths Setup ---
print("📥 Loading Official MedGemma from Kaggle...")
# Using the official Google handle - NO DOWNLOAD/UPLOAD NEEDED!
KAGGLE_MODEL_HANDLE = "google/medgemma/jax/4b-it/1"
BASE_MODEL_PATH = kagglehub.model_download(KAGGLE_MODEL_HANDLE)

# --- 3. Load Tokenizer ---
print("📥 Loading Tokenizer...")
# Note: Ensure the tokenizer file name matches (tokenizer.model or tokenizer.json)
tokenizer_path = os.path.join(BASE_MODEL_PATH, "tokenizer.model")
if not os.path.exists(tokenizer_path):
    tokenizer_path = os.path.join(BASE_MODEL_PATH, "tokenizer.json")

tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)

# --- 4. Create Model Structure ---
print("🧠 Creating Model Structure for MedGemma 4B (Gemma 3)...")
try:
    # Gemma 3 4B is now supported in modern Tunix
    model_config = gemma_lib.ModelConfig.gemma3_4b()
except AttributeError:
    print("⚠️  tunix.models.gemma3.ModelConfig.gemma3_4b not found. Manual config required.")
    # Generic 4B config placeholder - adjust based on exact architecture if needed
    model_config = gemma_lib.ModelConfig(
        num_layers=34,
        num_embed=262208,
        embed_dim=2560,
        hidden_dim=10240,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )
    
# MedGemma-4B uses a slightly larger vocab (262208) than standard Gemma-3 (262144)
model_config = dataclasses.replace(model_config, num_embed=262208)

with MESH:
    model = params_safetensors_lib.create_model_from_safe_tensors(
        BASE_MODEL_PATH,
        model_config,
        mesh=MESH
    )
    
    # Apply LoRA if you intend to use your GRPO weights on top of MedGemma
    # If using MedGemma as is, you can skip LoRA or keep it for the adapter weights.
    lora_config = {"module_path": ".*(attn|mlp).*(einsum|proj).*", "rank": LORA_RANK, "alpha": LORA_ALPHA}
    model = apply_lora_to_model(model, MESH, lora_config)

# --- 5. Optional: Restore GRPO Weights (If applicable to 27B) ---
# NOTE: If your weights from Step 900 were for 1B, they will NOT fit 27B.
# We comment this out to ensure the script runs with base MedGemma first.
"""
print("🔄 Restoring GRPO Weights...")
TRAINED_MODEL_PATH = kagglehub.model_download("surfiniaburger/gemma-3-1b-tunix-grpo-v8/jax/dipg-safety-900steps-44r/1")
checkpointer = ocp.StandardCheckpointer()
abstract_state = nnx.eval_shape(lambda: nnx.state(model))
state_restored = checkpointer.restore(TRAINED_MODEL_PATH, abstract_state)
nnx.update(model, state_restored)
"""

# --- 6. Initialize Sampler ---
print("⚙️  Initializing Sampler...")
sampler = sampler_lib.Sampler(
    transformer=model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=4096, 
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim
    )
)

# --- 7. Define Inference Logic ---
SYSTEM_PROMPT = """You are a strict safety assistant. You MUST prove your answer using the provided context.
Your final response MUST be structured using XML-like tags.

1. <think>...</think>: Analyze the question and plan.
2. <proof>...</proof>: Copy direct quotes from the context.
3. <answer>...</answer>: Provide the final answer based on the proof."""

def run_query(context, question):
    prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\nCONTEXT: {context}\n\nQUESTION: {question}<end_of_turn>\n<start_of_turn>model\n"
    
    output = sampler(
        input_strings=[prompt],
        max_generation_steps=1024,
        temperature=0.0 # Greedy
    )
    text = output.text[0]
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0]
    return text

# --- 8. Live Test ---
test_context = "A phase II trial reported that the HDAC inhibitor panobinostat achieved an objective response rate of 18% in patients with H3K27M‑mutant DIPG."
test_question = "What was the response rate for panobinostat?"

print(f"\n🚀 Running MedGemma Inference on 4B model...\n")
response = run_query(test_context, test_question)
print(response)
