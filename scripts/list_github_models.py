#!/usr/bin/env python3
"""
List Available GitHub Models

This script fetches and displays the list of available models from the GitHub Models Catalog.
It requires a GitHub Personal Access Token (PAT) to be set in the GITHUB_TOKEN environment variable.

Usage:
    export GITHUB_TOKEN=your_token_here
    python scripts/list_github_models.py
"""

import os
import sys
import requests
import json
from typing import List, Dict

def get_github_models() -> List[Dict]:
    """Fetch models from GitHub Models Catalog."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ Error: GITHUB_TOKEN environment variable is not set.")
        print("Please generate a token at https://github.com/settings/tokens")
        print("and run: export GITHUB_TOKEN=your_token_here")
        sys.exit(1)

    # Based on GitHub Models documentation
    url = "https://models.github.ai/catalog/models"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching models: {e}")
        if hasattr(response, 'status_code') and response.status_code == 401:
            print("Check if your token is valid.")
        sys.exit(1)

def main():
    print("🔍 Fetching available models from GitHub Models...")
    models = get_github_models()
    
    print(f"\n✅ Found {len(models)} models:\n")
    
    # Sort by name
    models.sort(key=lambda x: x.get('name', ''))
    
    print(f"{'Model Name':<40} {'Model ID':<40} {'Publisher':<15}")
    print("-" * 95)
    
    for model in models:
        name = model.get('name', 'N/A')
        model_id = model.get('id', name)
            
        publisher = model.get('publisher', 'Unknown')
        
        print(f"{name[:38]:<40} {model_id[:38]:<40} {publisher[:15]}")

    print("\nTo use a model in the benchmark:")
    print("python scripts/generate_benchmark_report.py --provider github --model <Model ID>")

if __name__ == "__main__":
    main()
