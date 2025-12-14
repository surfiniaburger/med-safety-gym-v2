import json
import random
import os
import argparse
from datasets import Dataset, DatasetDict

# Configuration
INPUT_FILE = "datasets/dipg_1500_final.jsonl"
TRAIN_SIZE = 1400
TEST_SIZE = 100
SEED = 42

def load_data(filepath):
    print(f"📖 Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(data)} items.")
    return data

def upload_to_hf(train_data, test_data, repo_id, token):
    print(f"🚀 Preparing to upload to Hugging Face Hub: {repo_id}")
    
    # Create HF Datasets
    hf_train = Dataset.from_list(train_data)
    hf_test = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        "train": hf_train,
        "test": hf_test
    })
    
    print(dataset_dict)
    
    # Push to Hub
    dataset_dict.push_to_hub(repo_id, token=token)
    print(f"✨ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Split and upload dataset to Hugging Face")
    parser.add_argument("--repo_id", required=True, help="Target HF repo ID (e.g., username/dataset-name)")
    parser.add_argument("--token", help="Hugging Face API Token")
    args = parser.parse_args()

    # Load and Shuffle
    data = load_data(INPUT_FILE)
    random.seed(SEED)
    random.shuffle(data)
    
    # Validate Size
    if len(data) < (TRAIN_SIZE + TEST_SIZE):
        print(f"⚠️  Not enough data! Have {len(data)}, need {TRAIN_SIZE + TEST_SIZE}.")
        return

    # Split
    train_data = data[:TRAIN_SIZE]
    test_data = data[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    
    print(f"✂️  Split Data:")
    print(f"   Train: {len(train_data)}")
    print(f"   Test:  {len(test_data)}")
    
    # Upload
    upload_to_hf(train_data, test_data, args.repo_id, args.token)

if __name__ == "__main__":
    main()
