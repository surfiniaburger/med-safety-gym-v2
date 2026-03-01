"""
Export GRPO-trained MedGemma (Orbax/JAX) → HuggingFace Transformers format.

This script:
1. Loads the trained JAX model from Orbax checkpoint
2. Merges LoRA weights into the base model
3. Converts JAX arrays → PyTorch tensors with HF-compatible naming
4. Saves as safetensors + config.json + tokenizer
5. Pushes to HuggingFace Hub

Run on Kaggle TPU (same environment as training) or any machine with enough RAM.

Usage:
    python scripts/export_to_hf.py \
        --base-model surfiniaburger/medgemma-4b-it \
        --trained-model surfiniaburger/medgemma-4b-v2 \
        --variant 1c \
        --output-dir ./medgemma-grpo-export \
        --hf-repo surfiniaburger/medgemma-4b-dipg-safety-v2 \
        --push
"""

import os
import re
import json
import argparse
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi

# Heavy JAX / TPU dependencies — only available in the Kaggle/Colab TPU environment
import jax
import flax.nnx as nnx
import kagglehub
from orbax import checkpoint as ocp
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models import safetensors_loader
from tunix.cli.utils.model import apply_lora_to_model

# ==============================================================================
# STEP 0: Parse Arguments
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Export Orbax JAX model to HuggingFace format")
    parser.add_argument("--base-model", type=str, default="surfiniaburger/medgemma-4b-it",
                        help="Kaggle model handle for the base model (for tokenizer + structure)")
    parser.add_argument("--trained-model", type=str, default="surfiniaburger/medgemma-4b-v2",
                        help="Kaggle model handle for the trained GRPO checkpoint")
    parser.add_argument("--variant", type=str, default="1c",
                        help="Kaggle model variant (e.g., '1c' for step 1200)")
    parser.add_argument("--output-dir", type=str, default="./medgemma-grpo-export",
                        help="Local directory to save the HF-format model")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo ID to push to (e.g., 'surfiniaburger/medgemma-4b-dipg-safety-v2')")
    parser.add_argument("--push", action="store_true",
                        help="If set, push the model to HuggingFace Hub")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    return parser.parse_args()


# ==============================================================================
# STEP 1: JAX Parameter Name → HuggingFace Parameter Name Mapping
# ==============================================================================
# Tunix/JAX uses: model.layer.{i}.attn.q_einsum.w  (shape: [heads, dim, head_dim])
# HuggingFace uses: model.layers.{i}.self_attn.q_proj.weight (shape: [heads*head_dim, dim])
#
# This mapping is specific to Gemma 3 architecture.

def build_jax_to_hf_mapping(num_layers: int) -> dict:
    """
    Build a dictionary mapping JAX/Tunix flat parameter paths
    to HuggingFace parameter names + required transpose info.
    
    Returns: dict[jax_path] -> (hf_name, needs_transpose, reshape_info)
    """
    mapping = {}
    
    # Embeddings
    mapping["model.embedder.input_embedding"] = (
        "model.embed_tokens.weight", False, None
    )
    
    # Final norm
    mapping["model.final_norm.scale"] = (
        "model.norm.weight", False, "add_one"  # JAX uses scale (multiply), HF uses weight (add 1)
    )
    
    # Per-layer mappings
    for i in range(num_layers):
        prefix_jax = f"model.layer.{i}"
        prefix_hf = f"model.layers.{i}"
        
        # Attention
        # Q, K, V projections — Tunix uses einsum convention, HF uses linear
        mapping[f"{prefix_jax}.attn.q_einsum.w"] = (
            f"{prefix_hf}.self_attn.q_proj.weight", True, "merge_heads"
        )
        mapping[f"{prefix_jax}.attn.kv_einsum.w"] = (
            None, True, "split_kv"  # Special: splits into k_proj and v_proj
        )
        mapping[f"{prefix_jax}.attn.attn_vec_einsum.w"] = (
            f"{prefix_hf}.self_attn.o_proj.weight", True, "merge_heads"
        )
        
        # MLP
        mapping[f"{prefix_jax}.mlp.gating_einsum"] = (
            None, True, "split_gate"  # Special: splits into gate_proj and up_proj
        )
        mapping[f"{prefix_jax}.mlp.linear"] = (
            f"{prefix_hf}.mlp.down_proj.weight", True, None
        )
        
        # Layer norms
        mapping[f"{prefix_jax}.pre_attention_norm.scale"] = (
            f"{prefix_hf}.input_layernorm.weight", False, "add_one"
        )
        mapping[f"{prefix_jax}.pre_ffw_norm.scale"] = (
            f"{prefix_hf}.post_attention_layernorm.weight", False, "add_one"
        )
    
    # LM head (sometimes tied with embeddings)
    mapping["model.lm_head.weight"] = (
        "lm_head.weight", False, None
    )
    
    return mapping


def merge_lora_weights(state_dict: dict) -> dict:
    """
    Merge LoRA A and B matrices back into the base weights.
    LoRA formula: W_merged = W_base + (B @ A) * (alpha / rank)
    
    Tunix/NNX LoRA keys typically look like:
        model.layer.0.attn.q_einsum.lora.lora_a
        model.layer.0.attn.q_einsum.lora.lora_b
    """
    merged = {}
    lora_pairs = {}
    
    for key, value in state_dict.items():
        if ".lora.lora_a" in key:
            base_key = key.replace(".lora.lora_a", "")
            lora_pairs.setdefault(base_key, {})["a"] = value
        elif ".lora.lora_b" in key:
            base_key = key.replace(".lora.lora_b", "")
            lora_pairs.setdefault(base_key, {})["b"] = value
        elif ".lora." not in key:
            merged[key] = value
    
    # Merge each LoRA pair
    for base_key, ab in lora_pairs.items():
        if "a" in ab and "b" in ab and base_key in merged:
            a = np.array(ab["a"])
            b = np.array(ab["b"])
            # LoRA: W' = W + B @ A * (alpha / rank)
            # The scaling is typically already baked in during training,
            # but verify with your training config
            lora_delta = np.einsum("...ij,...jk->...ik", b, a)
            merged[base_key] = np.array(merged[base_key]) + lora_delta
            print(f"  🔀 Merged LoRA into: {base_key}")
        elif base_key not in merged:
            print(f"  ⚠️  LoRA pair found but no base weight for: {base_key}")
    
    return merged


def flatten_state_dict(state, prefix=""):
    """Recursively flatten a nested JAX state dict into dot-separated paths."""
    flat = {}
    if hasattr(state, 'items'):
        for k, v in state.items():
            new_key = f"{prefix}.{k}" if prefix else k
            flat.update(flatten_state_dict(v, new_key))
    elif hasattr(state, '__dict__'):
        for k, v in state.__dict__.items():
            new_key = f"{prefix}.{k}" if prefix else k
            flat.update(flatten_state_dict(v, new_key))
    else:
        # Leaf node — convert to numpy
        flat[prefix] = np.array(state)
    return flat


def convert_jax_to_hf_state_dict(flat_jax_state: dict, num_layers: int, num_kv_heads: int) -> dict:
    """
    Convert flattened JAX state dict to HuggingFace-compatible state dict.
    """

    mapping = build_jax_to_hf_mapping(num_layers)
    hf_state = {}
    unmapped = []
    
    for jax_key, jax_val in flat_jax_state.items():
        # Strip .value suffix that NNX sometimes adds
        clean_key = jax_key.replace(".value", "").replace(".raw_value", "")
        
        if clean_key not in mapping:
            unmapped.append(clean_key)
            continue
        
        hf_name, needs_transpose, reshape_info = mapping[clean_key]
        val = np.array(jax_val)
        
        if reshape_info == "add_one":
            # JAX RMSNorm uses multiplicative scale; HF adds 1
            val = val + 1.0
        
        if reshape_info == "split_kv":
            # KV einsum has shape [2, num_kv_heads, dim, head_dim]
            # Split into k_proj and v_proj
            layer_match = re.search(r"layer\.(\d+)", clean_key)
            if layer_match:
                layer_idx = layer_match.group(1)
                k_val = val[0]  # [num_kv_heads, dim, head_dim]
                v_val = val[1]
                # Reshape: [num_kv_heads, dim, head_dim] -> [num_kv_heads * head_dim, dim]
                k_2d = k_val.reshape(-1, k_val.shape[-2]).T if needs_transpose else k_val.reshape(-1, k_val.shape[-1])
                v_2d = v_val.reshape(-1, v_val.shape[-2]).T if needs_transpose else v_val.reshape(-1, v_val.shape[-1])
                hf_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.tensor(k_2d, dtype=torch.bfloat16)
                hf_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.tensor(v_2d, dtype=torch.bfloat16)
            continue
        
        if reshape_info == "split_gate":
            # Gating einsum has shape [2, dim, hidden_dim]
            # Split into gate_proj and up_proj
            layer_match = re.search(r"layer\.(\d+)", clean_key)
            if layer_match:
                layer_idx = layer_match.group(1)
                gate_val = val[0]  # [dim, hidden_dim]
                up_val = val[1]
                if needs_transpose:
                    gate_val = gate_val.T
                    up_val = up_val.T
                hf_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.tensor(gate_val, dtype=torch.bfloat16)
                hf_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.tensor(up_val, dtype=torch.bfloat16)
            continue
        
        if reshape_info == "merge_heads" and needs_transpose:
            # [num_heads, dim, head_dim] -> [num_heads * head_dim, dim]
            val = val.reshape(-1, val.shape[-2]).T
        elif needs_transpose and val.ndim == 2:
            val = val.T
        
        if hf_name:
            hf_state[hf_name] = torch.tensor(val, dtype=torch.bfloat16)
    
    if unmapped:
        print(f"\n⚠️  {len(unmapped)} unmapped JAX keys (likely vision/unused):")
        for k in unmapped[:10]:
            print(f"    {k}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")
    
    return hf_state


# ==============================================================================
# STEP 2: Build HuggingFace config.json
# ==============================================================================
def create_hf_config(model_config, vocab_size: int = 262208) -> dict:
    """Create a HuggingFace-compatible config.json for Gemma 3."""
    return {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
        "hidden_size": model_config.embed_dim,
        "intermediate_size": model_config.hidden_dim,
        "num_hidden_layers": model_config.num_layers,
        "num_attention_heads": model_config.num_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "head_dim": model_config.head_dim,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "transformers_version": "4.49.0",
    }


# ==============================================================================
# STEP 3: Main Export Pipeline
# ==============================================================================
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Load JAX model (same as hug.md) ----
    print("=" * 60)
    print("PHASE 1: Loading JAX Model from Orbax")
    print("=" * 60)

    # ===========================================================
    # FIX 1: Clear GPU-specific XLA_FLAGS that may have been set
    # by train_grpo_tpu.py in the same Kaggle kernel session.
    # The flags --xla_gpu_* are invalid on a TPU runtime and
    # cause a fatal crash. We keep only TPU-safe flags.
    # ===========================================================
    gpu_flags = [
        "--xla_gpu_enable_triton_softmax_fusion=true",
        "--xla_gpu_triton_gemm_any=True",
        "--xla_gpu_enable_async_collectives=true",
    ]
    current_xla_flags = os.environ.get("XLA_FLAGS", "")
    cleaned_flags = " ".join(
        f for f in current_xla_flags.split() if f not in gpu_flags
    )
    if cleaned_flags != current_xla_flags:
        print(f"⚠️  Removed GPU-specific XLA_FLAGS from environment.")
    os.environ["XLA_FLAGS"] = cleaned_flags

    # ===========================================================
    # FIX 2: Auto-detect TPU device count for mesh construction.
    # Hardcoding (8,1) fails if the session has a different slice
    # or is running on CPU/GPU fallback.
    # ===========================================================
    num_devices = len(jax.devices())
    print(f"   Detected {num_devices} JAX device(s): {jax.default_backend()}")
    if num_devices >= 8:
        mesh_shape = (8, 1)
    elif num_devices >= 4:
        mesh_shape = (4, 1)
    elif num_devices >= 2:
        mesh_shape = (2, 1)
    else:
        mesh_shape = (1, 1)  # CPU/single-device fallback
    print(f"   Using mesh shape: {mesh_shape}")

    # Apply NNX compatibility patch
    _orig_set_metadata = nnx.Variable.set_metadata
    def _compat_set_metadata(self, *a, **kw):
        if len(a) == 2 and isinstance(a[0], str):
            kw[a[0]] = a[1]
            return _orig_set_metadata(self, **kw)
        return _orig_set_metadata(self, *a, **kw)
    nnx.Variable.set_metadata = _compat_set_metadata
    
    # Apply Ghost Buster mapper
    def ghost_buster_key_mapper(mapping, source_key):
        if "vision" in source_key.lower():
            return f"unused.vision.{source_key}", None
        norm_key = source_key
        if source_key.startswith("language_model.model."):
            norm_key = source_key.replace("language_model.model.", "model.")
        elif source_key.startswith("language_model."):
            norm_key = source_key.replace("language_model.", "model.")
        try:
            subs = [(re.sub(pat, repl, norm_key), reshape)
                    for pat, (repl, reshape) in mapping.items()
                    if re.match(pat, norm_key) or re.match(pat, source_key)]
            if len(subs) == 1: return subs[0]
        except Exception: pass
        return f"unused.unknown.{source_key}", None
    safetensors_loader.torch_key_to_jax_key = ghost_buster_key_mapper
    
    MESH = jax.make_mesh(mesh_shape, ('fsdp', 'tp'))
    
    BASE_MODEL_PATH = kagglehub.model_download(f"{args.base_model}/jax/it")
    TRAINED_MODEL_PATH = kagglehub.model_download(f"{args.trained_model}/jax/{args.variant}")
    
    # Create model config
    try:
        model_config = gemma_lib.ModelConfig.gemma3_4b()
    except AttributeError:
        model_config = gemma_lib.ModelConfig(
            num_layers=34, num_embed=262208, embed_dim=2560,
            hidden_dim=10240, num_heads=32, num_kv_heads=8, head_dim=128,
        )
    model_config = dataclasses.replace(model_config, num_embed=262208)
    
    with MESH:
        model = params_safetensors_lib.create_model_from_safe_tensors(
            BASE_MODEL_PATH, model_config, mesh=MESH
        )
        lora_config = {
            "module_path": ".*(attn|mlp).*(einsum|proj).*",
            "rank": args.lora_rank, "alpha": args.lora_alpha
        }
        model = apply_lora_to_model(model, MESH, lora_config)
    
    # Restore trained weights
    print("\n🔄 Restoring GRPO checkpoint...")
    checkpointer = ocp.StandardCheckpointer()
    abstract_state = nnx.eval_shape(lambda: nnx.state(model))
    state_restored = checkpointer.restore(TRAINED_MODEL_PATH, abstract_state)
    nnx.update(model, state_restored)
    
    # ---- Convert ----
    print("\n" + "=" * 60)
    print("PHASE 2: Converting JAX → HuggingFace Format")
    print("=" * 60)
    
    # Flatten JAX state
    print("📦 Flattening JAX state dict...")
    jax_state = nnx.state(model)
    flat_state = flatten_state_dict(jax_state)
    print(f"   Found {len(flat_state)} parameters")
    
    # Merge LoRA weights
    print("🔀 Merging LoRA weights into base...")
    merged_state = merge_lora_weights(flat_state)
    print(f"   {len(merged_state)} parameters after merge")
    
    # Convert to HF format
    print("🔄 Mapping to HuggingFace naming convention...")
    hf_state_dict = convert_jax_to_hf_state_dict(
        merged_state, 
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads
    )
    print(f"   ✅ {len(hf_state_dict)} HF parameters created")
    
    # ---- Save ----
    print("\n" + "=" * 60)
    print("PHASE 3: Saving as HuggingFace Format")
    print("=" * 60)
    
    # Save safetensors
    safetensors_path = output_dir / "model.safetensors"
    print(f"💾 Saving safetensors to {safetensors_path}...")
    save_file(hf_state_dict, str(safetensors_path))
    
    # Save config.json
    config = create_hf_config(model_config)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Saved config.json")
    
    # Copy tokenizer files from base model
    print("📋 Copying tokenizer files...")
    tokenizer_files = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "generation_config.json"]
    for fname in tokenizer_files:
        src = os.path.join(BASE_MODEL_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_dir / fname)
            print(f"   ✅ {fname}")
        else:
            print(f"   ⏭️  {fname} not found in base model (skipping)")
    
    # Create a model card
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"""---
license: apache-2.0
base_model: google/medgemma-4b-it
tags:
  - medical
  - safety
  - grpo
  - dipg
  - safeclaw
library_name: transformers
pipeline_tag: text-generation
---

# MedGemma 4B - DIPG Safety v2 (GRPO-Trained)

Fine-tuned [MedGemma 4B](https://huggingface.co/google/medgemma-4b-it) using 
**GRPO (Group Relative Policy Optimization)** on the DIPG Safety Gym benchmark.

## Training Details
- **Base Model**: MedGemma 4B IT
- **Method**: GRPO with LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})
- **Training Steps**: 100
- **Focus**: Medical safety — hallucination reduction, evidence-grounded responses

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{args.hf_repo or 'surfiniaburger/medgemma-4b-dipg-safety-v2'}")
model = AutoModelForCausalLM.from_pretrained(
    "{args.hf_repo or 'surfiniaburger/medgemma-4b-dipg-safety-v2'}",
    torch_dtype="bfloat16",
    device_map="auto"
)
```

## SafeClaw Project
Part of the [DIPG Safety Gym](https://github.com/surfiniaburger/med-safety-gym-v2) ecosystem.
""")
    print(f"📝 Saved README.md (model card)")
    
    # ---- Push to HuggingFace Hub ----
    if args.push and args.hf_repo:
        print("\n" + "=" * 60)
        print(f"PHASE 4: Pushing to HuggingFace Hub → {args.hf_repo}")
        print("=" * 60)
        
        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.hf_repo,
            commit_message="Upload GRPO-trained MedGemma 4B (DIPG Safety v2)"
        )
        print(f"🚀 Pushed to https://huggingface.co/{args.hf_repo}")
    
    print("\n✅ Export complete!")
    print(f"   Output: {output_dir}")
    print(f"   Files: {[f.name for f in output_dir.iterdir()]}")


if __name__ == "__main__":
    main()
