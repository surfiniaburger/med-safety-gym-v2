# test_client.py
import requests
import json

SERVER_URL = "http://127.0.0.1:8000/mcp"

def run_tests():
    print(f"--- Testing Server at {SERVER_URL} ---\n")

    # 1. Test GET (Check status)
    try:
        print("1. Sending GET request (Handshake)...")
        resp = requests.get(SERVER_URL)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}\n")
    except Exception as e:
        print(f"Server seems down. Error: {e}")
        return

    # 2. Test POST: List Tools
    print("2. Listing Tools...")
    payload_list = {"method": "tools/list"}
    resp = requests.post(SERVER_URL, json=payload_list)
    print(f"Response: {json.dumps(resp.json(), indent=2)}\n")

    # 3. Test POST: Call Tool
    # Note: Make sure Ollama is running or LiteLLM keys are set in .env!
    print("3. Calling Tool 'ask_opensource_model'...")
    
    prompt_text = "What is the capital of France?"
    
    payload_call = {
        "method": "tools/call",
        "arguments": {
            "prompt": prompt_text,
            # If using Ollama, make sure this model is pulled (`ollama pull llama3`)
            # Or use "gpt-3.5-turbo" if you have OPENAI_API_KEY in .env
            "model": "ollama/qwen3-coder:480b-cloud" 
        }
    }

    print(f"Sending Prompt: '{prompt_text}'")
    resp = requests.post(SERVER_URL, json=payload_call)
    
    if resp.status_code == 200:
        print(f"Result: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"Error ({resp.status_code}): {resp.text}")

if __name__ == "__main__":
    run_tests()