import asyncio
import json
import os
import httpx
from med_safety_gym.messenger import send_message

async def main():
    # Give the agents a moment to settle
    print("⏳ Waiting for agents to settle...")
    await asyncio.sleep(5)
    
    # Use 127.0.0.1 for host-to-container communication
    green_agent_url = "http://127.0.0.1:10001/"
    
    # The Green Agent (inside Docker) will use this URL to talk to the Purple Agent
    # It MUST match the service name in docker-compose.yml
    purple_agent_url_internal = "http://purple-agent:9009/"
    
    eval_request = {
        "participants": {
            "purple_agent": purple_agent_url_internal
        },
        "config": {
            "num_tasks": 2
        }
    }
    
    print(f"🚀 Sending EvalRequest to Green Agent at {green_agent_url}...")
    print(f"👉 Target Purple Agent (internal): {purple_agent_url_internal}")
    
    try:
        results = await send_message(
            message=json.dumps(eval_request),
            base_url=green_agent_url,
            timeout=120
        )
        
        print("\n--- 🏁 Evaluation Finished ---")
        print(f"Status: {results.get('status')}")
        
        print("\n--- 📊 Results Summary ---")
        print(results.get("response"))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())