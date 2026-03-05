"""SFT training script for SafeClaw Sentinel (Intent Classifier) expert."""
import asyncio
import art
from art.local import LocalBackend
from art.utils.sft import train_sft_from_file

async def main():
    # Use LocalBackend for local GPU training
    backend = LocalBackend()
    
    # Define the Sentinel model
    # We use Qwen2.5-7B-Instruct as a base for a good balance of speed and reasoning
    model = art.TrainableModel(
        name="sentinel-001",
        project="safeclaw-experts",
        base_model="OpenPipe/Qwen2.5-7B-Instruct",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.5}, # Be conservative with GPU memory
        ),
    )
    
    print(f"Registering Sentinel model: {model.name}")
    await model.register(backend)

    # Train from the JSONL file we extracted
    print("Starting SFT training for Sentinel...")
    await train_sft_from_file(
        model=model,
        file_path="data/sentinel_train.jsonl",
        epochs=10, # More epochs for small dataset
        batch_size=2,
        peak_lr=2e-4,
        verbose=True,
    )

    # --- Training complete ---
    step = await model.get_step()
    inference_name = model.get_inference_name()
    client = model.openai_client()

    print("\n" + "=" * 60)
    print("SENTINEL SFT TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {inference_name}")
    print(f"  Training step:  {step}")
    print(f"  Inference URL:  {client.base_url}")
    print("=" * 60)

    await backend.close()

if __name__ == "__main__":
    asyncio.run(main())
