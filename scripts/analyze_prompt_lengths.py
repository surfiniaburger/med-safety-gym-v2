
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def analyze_lengths():
    print("Loading dataset...")
    ds = load_dataset("surfiniaburger/dipg-safety-instruction-1500")["train"]
    
    print(f"Dataset size: {len(ds)}")
    
    lengths_char = []
    lengths_token = []
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it") # Approximate for Gemma 3
    
    print("Analyzing lengths...")
    for item in ds:
        # Construct the prompt exactly as in training
        user_content = item["instruction"]
        # input (optional context) is usually appended in instruction tuning formats, checking typical usage
        if item.get("input"):
             user_content += "\n" + item["input"]
             
        lengths_char.append(len(user_content))
        tokens = tokenizer.encode(user_content, add_special_tokens=False)
        lengths_token.append(len(tokens))
        
    # Stats
    print("\n--- Character Stats ---")
    print(f"Max: {np.max(lengths_char)}")
    print(f"99th Percentile: {np.percentile(lengths_char, 99)}")
    print(f"95th Percentile: {np.percentile(lengths_char, 95)}")
    print(f"Median: {np.median(lengths_char)}")
    print(f"Truncated items (>3000 chars): {sum(l > 3000 for l in lengths_char)} / {len(ds)} ({sum(l > 3000 for l in lengths_char)/len(ds):.2%})")

    print("\n--- Token Stats (Approx) ---")
    print(f"Max: {np.max(lengths_token)}")
    print(f"99th Percentile: {np.percentile(lengths_token, 99)}")
    print(f"95th Percentile: {np.percentile(lengths_token, 95)}")
    print(f"Median: {np.median(lengths_token)}")
    
    # Check what the 3000 char limit usually corresponds to in tokens for this data
    # Simple correlation check
    long_items_indices = [i for i, l in enumerate(lengths_char) if l > 3000]
    if long_items_indices:
        print(f"\nExample of truncated item ({lengths_char[long_items_indices[0]]} chars / {lengths_token[long_items_indices[0]]} tokens):")
        # print first 100 chars
        print(ds[long_items_indices[0]]["instruction"][:100] + "...")

if __name__ == "__main__":
    analyze_lengths()
