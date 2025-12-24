"""
Test runner to trigger a Green Agent evaluation.
"""
import asyncio
import json
from med_safety_gym.messenger import send_message

async def main():
    green_agent_url = "http://localhost:10001/"
    purple_agent_url = "http://localhost:10002/"
    
    # Construct the EvalRequest JSON
    eval_request = {
        "participants": {
            "purple_agent": purple_agent_url
        },
        "config": {
            "num_samples": 2
        }
    }
    
    print(f"Sending EvalRequest to Green Agent at {green_agent_url}...")
    
    results = await send_message(
        message=json.dumps(eval_request),
        base_url=green_agent_url,
        timeout=60
    )
    
    print("--- 🏁 Evaluation Finished ---")
    print(f"Status: {results.get('status')}")
    print("Response Summarized:")
    print(results.get("response"))
    
    # Note: Full data artifact would be in the 'artifacts' field if we parsed it more deeply,
    # but our helper prints the merged parts.

if __name__ == "__main__":
    asyncio.run(main())
