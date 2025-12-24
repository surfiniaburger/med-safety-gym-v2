"""
Simple OpenAI-compatible API server using LiteLLM
This wraps the MCP server to provide a standard /v1/chat/completions endpoint
"""
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import litellm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litellm_openai_server")

# Configure LiteLLM
DEFAULT_MODEL = "ollama/nemotron-3-nano:30b-cloud"  # Change to your model

async def chat_completions(request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        body = await request.json()
        
        model = body.get("model", DEFAULT_MODEL)
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 500)
        
        logger.info(f"Request: model={model}, messages={len(messages)}")
        
        # Call LiteLLM
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Return OpenAI-compatible response
        return JSONResponse({
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500
        )

async def health_check(request):
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "model": DEFAULT_MODEL})

app = Starlette(routes=[
    Route("/v1/chat/completions", endpoint=chat_completions, methods=["POST"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
])

if __name__ == "__main__":
    print(f"Starting LiteLLM OpenAI-compatible server on port 8082")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Endpoint: http://localhost:8082/v1/chat/completions")
    uvicorn.run(app, host="127.0.0.1", port=8082)
