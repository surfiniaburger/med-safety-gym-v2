# server/mcp_server.py
import asyncio
import json
import os
from typing import List, Optional, Dict, Any

from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

try:
    from .app import get_environment, EvaluationManager, EvaluationRequest, EvaluationResult
    from .evaluation_service import EvaluationItem, GroundTruth
except ImportError:
    # Fallback for running as a script directly
    from med_safety_gym.app import get_environment, EvaluationManager, EvaluationRequest, EvaluationResult
    from med_safety_gym.evaluation_service import EvaluationItem, GroundTruth

# Initialize MCP Server
server = Server("dipg-safety-gym-mcp")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="get_eval_tasks",
            description="Get evaluation tasks from the DIPG Safety Gym dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_samples": {
                        "type": "integer",
                        "description": "Maximum number of tasks to return (default: all tasks)",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Whether to shuffle tasks before sampling (default: True)",
                    },
                },
            },
        ),
        types.Tool(
            name="evaluate_batch",
            description="Evaluate a batch of model responses against the DIPG Safety Gym.",
            inputSchema={
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string"},
                                "ground_truth": {
                                    "type": "object",
                                    "properties": {
                                        "context": {"type": "string"},
                                        "question": {"type": "string"},
                                        "expected_answer": {"type": "object"}
                                    },
                                    "required": ["context", "question"]
                                }
                            },
                            "required": ["response", "ground_truth"]
                        },
                        "description": "List of evaluation items containing response and ground truth."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "custom_tags", "xml", "yaml", "auto"],
                        "default": "json",
                        "description": "Format of the model responses."
                    }
                },
                "required": ["evaluations"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution."""
    safe_args = arguments or {}
    
    if name == "get_eval_tasks":
        max_samples = safe_args.get("max_samples")
        shuffle = safe_args.get("shuffle", True)
        
        env = get_environment()
        tasks = env.get_eval_tasks(max_samples=max_samples, shuffle=shuffle)
        
        return [types.TextContent(type="text", text=json.dumps({
            "tasks": tasks,
            "total_tasks": len(tasks),
            "dataset_size": len(env.dataset)
        }, indent=2))]

    elif name == "evaluate_batch":
        evaluations_data = safe_args.get("evaluations")
        response_format = safe_args.get("format", "json")
        
        if not evaluations_data:
            raise ValueError("Missing 'evaluations' in arguments")

        # Convert raw dicts to Pydantic models
        evaluation_items = []
        for item_data in evaluations_data:
            # Convert ground_truth dict to GroundTruth model
            gt_data = item_data.get("ground_truth")
            ground_truth = GroundTruth(**gt_data) if gt_data else None
            
            # Create EvaluationItem
            evaluation_items.append(EvaluationItem(
                response=item_data["response"],
                ground_truth=ground_truth
            ))
        
        env = get_environment()
        eval_manager = EvaluationManager(env)
        
        # Use the stateless mode logic
        result = eval_manager.evaluate_with_ground_truth(
            evaluations=evaluation_items,
            response_format=response_format
        )
        
        # Result is an EvaluationResult object, we need to serialize it
        return [types.TextContent(type="text", text=result.model_dump_json(indent=2))]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def run():
    # Run the server using stdin/stdout
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="dipg-safety-gym-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
