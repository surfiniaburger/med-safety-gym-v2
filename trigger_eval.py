import asyncio
import json
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
from uuid import uuid4

async def main():
    url = "http://127.0.0.1:10001/"
    internal_purple_url = "http://purple-agent:9009/"
    
    eval_request = {
        "participants": {
            "purple_agent": internal_purple_url
        },
        "config": {
            "num_tasks": 1
        }
    }
    
    async with httpx.AsyncClient(timeout=120, http2=False) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        print(f"✅ Connected to: {agent_card.name}")
        
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(root=TextPart(kind="text", text=json.dumps(eval_request)))],
            message_id=uuid4().hex,
        )
        
        print("📤 Sending EvalRequest...")
        async for event in client.send_message(msg):
            print(f"📥 Event: {event}")

if __name__ == "__main__":
    asyncio.run(main())
