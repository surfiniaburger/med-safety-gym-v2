"""
A2A Test Client for DIPG Evaluation Agent

This client tests the A2A integration by connecting to the DIPG evaluation agent
and requesting evaluation tasks.

Based on Google's currency agent test client pattern.
"""

import os
import sys
import traceback
from typing import Any
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
)
import httpx

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:10000")


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a message."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


def print_json_response(response: Any, description: str) -> None:
    """Helper function to print the JSON representation of a response."""
    print(f"--- {description} ---")
    if hasattr(response, "root"):
        print(f"{response.root.model_dump_json(exclude_none=True, indent=2)}\n")
    else:
        print(f"{response.model_dump(mode='json', exclude_none=True, indent=2)}\n")


async def test_get_tasks(client: A2AClient) -> None:
    """Test getting evaluation tasks."""
    
    print("=" * 70)
    print("TEST 1: Get Evaluation Tasks")
    print("=" * 70)
    
    send_message_payload = create_send_message_payload(
        text="Get me 3 evaluation tasks from the DIPG dataset"
    )
    request = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    print("📤 Sending request...")
    response: SendMessageResponse = await client.send_message(request)
    print_json_response(response, "📥 Response")
    
    if not isinstance(response.root, SendMessageSuccessResponse):
        print("⚠️  Received non-success response")
        return

    if not isinstance(response.root.result, Task):
        print("⚠️  Received non-task response")
        return

    task_id: str = response.root.result.id
    print(f"✅ Task created: {task_id}")
    
    # Query the task to get final result
    print("\n📋 Querying task status...")
    get_request = GetTaskRequest(id=str(uuid4()), params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, "📥 Task Result")


async def test_evaluation_workflow(client: A2AClient) -> None:
    """Test the full evaluation workflow."""
    
    print("\n" + "=" * 70)
    print("TEST 2: Full Evaluation Workflow")
    print("=" * 70)
    
    # First, ask for tasks
    print("\n📤 Step 1: Requesting evaluation tasks...")
    first_payload = create_send_message_payload(
        text="I need 2 evaluation tasks to test my model"
    )
    request1 = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**first_payload)
    )
    first_response: SendMessageResponse = await client.send_message(request1)
    print_json_response(first_response, "📥 Tasks Response")
    
    context_id: str | None = None
    if isinstance(first_response.root, SendMessageSuccessResponse) and isinstance(
        first_response.root.result, Task
    ):
        task: Task = first_response.root.result
        context_id = task.context_id
        print(f"✅ Context ID captured: {context_id}")
        
        # Now ask about evaluation (multi-turn)
        if task.status.state == TaskState.input_required and context_id:
            print("\n📤 Step 2: Asking about evaluation...")
            second_payload = create_send_message_payload(
                "How do I evaluate responses with these tasks?",
                task.id,
                context_id
            )
            request2 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**second_payload)
            )
            second_response = await client.send_message(request2)
            print_json_response(second_response, "📥 Evaluation Instructions")


async def main() -> None:
    """Main function to run the tests."""
    print(f"🔗 Connecting to DIPG Evaluation Agent at {AGENT_URL}...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            # Create a resolver to fetch the agent card
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=AGENT_URL,
            )
            agent_card = await resolver.get_agent_card()
            print(f"✅ Agent card retrieved: {agent_card.name}")
            
            # Create a client to interact with the agent
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card,
            )
            print("✅ A2A client created\n")

            # Run tests
            await test_get_tasks(client)
            await test_evaluation_workflow(client)
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)

    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. FastMCP server is running on port 8081")
        print("2. A2A agent server is running on port 10000")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
