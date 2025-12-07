
import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_thinking():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        return

    client = genai.Client(api_key=api_key)
    
    # Test Prompt: A medical reasoning task to see how detailed the thought summary is.
    prompt = """
    A 7-year-old boy with DIPG presents with new onset facial weakness and gait instability. 
    MRI shows tumor progression. 
    He has an H3K27M mutation and ACVR1 mutation.
    What is the most evidence-based next line of therapy considering his molecular profile?
    """

    print(f"🚀 Sending request to gemini-2.5-pro with thinking enabled...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=1024 # Set a budget to ensure it thinks
                )
            )
        )

        print("\n--- Response Received ---\n")
        
        found_thought = False
        for part in response.candidates[0].content.parts:
            if part.thought:
                print(f"🧠 [THOUGHT SUMMARY]:\n{part.text}\n")
                found_thought = True
            else:
                print(f"📝 [ANSWER]:\n{part.text}\n")
        
        if not found_thought:
            print("⚠️ No thought summary found in response.")

        # Check token usage if available
        if response.usage_metadata:
            print(f"📊 Token Usage: {response.usage_metadata}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test with Gemini 2.5 Pro (Higher Rate Limit)
    print("Testing Gemini 2.5 Pro...")
    test_thinking()
