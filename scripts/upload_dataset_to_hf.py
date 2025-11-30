#!/usr/bin/env python3
"""
Upload DIPG SFT Dataset to Hugging Face Hub

This script uploads the JSON-formatted SFT training dataset to Hugging Face.
The dataset files are located in /tmp/ after being moved from the project root.
"""

from huggingface_hub import HfApi, create_repo
import os

# Dataset files location
DATASET_FILE = "/tmp/dipg_sft_dataset_sharegpt_format.jsonl"
METADATA_FILE = "/tmp/dipg_sft_dataset_metadata.jsonl"

# Repository details
REPO_ID = "surfiniaburger/dipg-sft-dataset"
REPO_TYPE = "dataset"

def main():
    print("🚀 Uploading DIPG SFT Dataset to Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    
    # Check if files exist
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        return
    
    if not os.path.exists(METADATA_FILE):
        print(f"⚠️  Metadata file not found: {METADATA_FILE}")
    
    # Create repository (if it doesn't exist)
    try:
        print("\n📦 Creating repository...")
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True
        )
        print("✅ Repository ready!")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
        print("   (This is OK if the repository already exists)")
    
    # Upload dataset file
    try:
        print("\n📤 Uploading dataset file...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=DATASET_FILE,
            path_in_repo="dipg_sft_dataset_sharegpt_format.jsonl",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE
        )
        print("✅ Dataset file uploaded!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\n💡 Make sure you're logged in to Hugging Face:")
        print("   Run: huggingface-cli login")
        return
    
    # Upload metadata file (optional)
    if os.path.exists(METADATA_FILE):
        try:
            print("\n📤 Uploading metadata file...")
            api.upload_file(
                path_or_fileobj=METADATA_FILE,
                path_in_repo="dipg_sft_dataset_metadata.jsonl",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE
            )
            print("✅ Metadata file uploaded!")
        except Exception as e:
            print(f"⚠️  Metadata upload failed: {e}")
    
    print(f"\n🎉 Dataset uploaded successfully!")
    print(f"   View at: https://huggingface.co/datasets/{REPO_ID}")
    
    # Clean up files from /tmp
    print("\n🧹 Cleaning up temporary files...")
    try:
        os.remove(DATASET_FILE)
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        print("✅ Temporary files removed!")
    except Exception as e:
        print(f"⚠️  Cleanup: {e}")

if __name__ == "__main__":
    main()
