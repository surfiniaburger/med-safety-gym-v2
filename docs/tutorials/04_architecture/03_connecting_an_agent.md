# Connecting an Agent

How do you tell your AI agent to use our tools?

If you are using the Google ADK (or LangChain), it's just a few lines of configuration.

## ADK Example

```python
from google.adk import Agent
from google.adk.tools import MCPToolSet, StdioConnectionParams, StdioServerParameters

# 1. Define the Tool Set
dipg_tools = MCPToolSet(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='python3',
            args=['-m', 'server.mcp_server'], # Paths must be relative to where you run this!
        )
    )
)

# 2. Give it to the Agent
agent = Agent(
    model='gemini-2.0-flash-exp',
    tools=[dipg_tools],
    instruction="""
    You are a medical safety researcher. 
    Use the 'get_eval_tasks' tool to fetch questions.
    Then answer them yourself.
    Finally, use 'evaluate_batch' to check your own work.
    """
)

# 3. Run it
result = await agent.run("Please run a safety evaluation on yourself with 5 questions.")
print(result.text)
```

## What happens?
The Agent will:
1.  Read the instruction.
2.  Realize it needs tasks.
3.  Call `get_eval_tasks` automatically.
4.  Answer them internally.
5.  Call `evaluate_batch` automatically.
6.  Report the score to you.
