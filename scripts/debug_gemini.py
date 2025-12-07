#!/usr/bin/env python3
"""Debug script to understand Gemini's response structure"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

prompt = """Generate a simple JSON with these keys: "context", "question", "think", "proof", "answer".
Make it about DIPG treatment."""

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[{"role": "user", "parts": [{"text": prompt}]}],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=512
        ),
        response_mime_type="application/json"
    )
)

print("=== Response Structure ===")
print(f"Number of candidates: {len(response.candidates)}")
print(f"Number of parts: {len(response.candidates[0].content.parts)}")

for i, part in enumerate(response.candidates[0].content.parts):
    print(f"\n--- Part {i} ---")
    print(f"Type: {type(part)}")
    print(f"Has 'thought' attr: {hasattr(part, 'thought')}")
    if hasattr(part, 'thought'):
        print(f"Is thought: {part.thought}")
    print(f"Has 'text' attr: {hasattr(part, 'text')}")
    if hasattr(part, 'text'):
        print(f"Text preview: {part.text[:200] if part.text else 'None'}...")

print("\n=== Usage Metadata ===")
if response.usage_metadata:
    print(f"Thinking tokens: {response.usage_metadata.thoughts_token_count}")
    print(f"Output tokens: {response.usage_metadata.candidates_token_count}")
