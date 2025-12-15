
import os
import argparse
from datasets import load_dataset

def deploy_eval_dataset(token):
    # 1. Load the MASTER dataset (Split: TEST)
    # This ensures we are using the held-out data, NOT training data.
    SOURCE_REPO = "surfiniaburger/dipg-safety-instruction-1500"
    TARGET_REPO = "surfiniaburger/med-safety-gym-eval"
    
    print(f"📥 Loading test split from {SOURCE_REPO}...")
    # Authentication (optional if private, needed if gated)
    ds = load_dataset(SOURCE_REPO, split="test", token=token)
    
    print(f"✅ Loaded {len(ds)} samples.")
    
    # 2. Validation
    # Ensure this is actually the test set we expect (100 samples)
    if len(ds) != 100:
        print(f"⚠️ Warning: Expected 100 test samples, found {len(ds)}")
        # confirm = input("Continue anyway? (y/n): ")
        # if confirm.lower() != 'y': return

    # 3. Push to New Repo
    print(f"🚀 Pushing to {TARGET_REPO}...")
    ds.push_to_hub(TARGET_REPO, token=token, private=False) # Public for Gym usage
    
    print("✅ Deployment Complete!")
    print(f"   Evaluators can now use: dataset_path='{TARGET_REPO}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face Write Token", required=True)
    args = parser.parse_args()
    
    deploy_eval_dataset(args.token)
