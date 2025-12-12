import sys
import json
import argparse
import difflib
from typing import List, Dict, Set

# Constants
REQUIRED_TAGS = ["<think>", "</think>", "<proof>", "</proof>", "<answer>", "</answer>"]

def load_dataset(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Check malformed JSON at line {i+1}", file=sys.stderr)
    return data

def get_user_content(item: Dict) -> str:
    """Helper to extract user content from messages."""
    messages = item.get("messages", [])
    return next((m["content"] for m in messages if m["role"] == "user"), "")

def validate_schema(item: Dict) -> bool:
    """Checks for required fields and non-empty content inside the nested message structure."""
    if "messages" not in item:
        return False
    
    # Extract user and assistant content
    user_content = get_user_content(item)
    assistant_content = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
    
    if not user_content or not assistant_content:
        return False

    # Check for basic tags in assistant content
    if not all(tag in assistant_content for tag in REQUIRED_TAGS):
        return False
        
    return True

def get_fingerprint(item: Dict) -> str:
    """Creates a fingerprint string for deduplication (based on question)."""
    # Simple extraction: just use the whole user content as fingerprint
    # Ideally we'd parse <question>...</question> but this is robust enough for exact duplicates
    return get_user_content(item).strip()

def is_similar(s1: str, s2: str, threshold: float = 0.9) -> bool:
    """Checks if two strings are highly similar."""
    return difflib.SequenceMatcher(None, s1, s2).ratio() > threshold

def clean_dataset(data: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
    cleaned_data = []
    seen_fingerprints: Set[str] = set()
    
    malformed_count = 0
    duplicate_count = 0
    
    print(f"🔍 analyzing {len(data)} items...")

    for i, item in enumerate(data):
        if i % 50 == 0:
            print(f"  ... processing item {i}/{len(data)}")

        # 1. Schema Validation
        if not validate_schema(item):
            malformed_count += 1
            continue
            
        # 2. Deduplication
        fingerprint = get_fingerprint(item)
        
        # Exact match check
        if fingerprint in seen_fingerprints:
            duplicate_count += 1
            continue
            
        # Fuzzy match check (expensive O(N^2) worst case, but manageable for 300 items)
        # We only check against accepted items to keep the accepted set clean
        is_fuzzy_duplicate = False
        for seen in seen_fingerprints:
            if is_similar(fingerprint, seen, threshold=similarity_threshold):
                is_fuzzy_duplicate = True
                break
        
        if is_fuzzy_duplicate:
            duplicate_count += 1
            continue

        seen_fingerprints.add(fingerprint)
        cleaned_data.append(item)

    print(f"❌ Removed {malformed_count} malformed items")
    print(f"❌ Removed {duplicate_count} duplicate/highly similar items")
    print(f"✅ Kept {len(cleaned_data)} high-quality items")
    
    return cleaned_data

def main():
    parser = argparse.ArgumentParser(description="Clean and filter dataset.")
    parser.add_argument("input_file", help="Path to input .jsonl file")
    parser.add_argument("output_file", help="Path to output .jsonl file")
    args = parser.parse_args()

    data = load_dataset(args.input_file)
    cleaned_data = clean_dataset(data)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"💾 Saved cleaned dataset to {args.output_file}")

if __name__ == "__main__":
    main()
