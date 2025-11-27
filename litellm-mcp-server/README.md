# LiteLLM MCP Server

This directory contains the implementation of a Model-in-the-Middle (MCP) server that exposes the `litellm` library as a tool to the `gemini-cli`.

## Mock Server for Testing

For debugging and verification, a `mock_server.py` is included. This server simulates the behavior of a compliant MCP server, allowing you to test the `gemini-cli`'s client implementation without needing to run the full `litellm` stack.

### Running the Mock Server

1.  Ensure you have the necessary dependencies installed:
    ```bash
    # from the project root
    ./litellm-mcp-server/.venv/bin/python -m pip install -r ./litellm-mcp-server/requirements.txt
    ```

2.  Start the mock server:
    ```bash
    # from the project root
    ./litellm-mcp-server/.venv/bin/python ./litellm-mcp-server/mock_server.py
    ```
    The server will start on `http://localhost:8000`.

### Interacting with the Mock Server using `curl`

You can use `curl` to send requests to the mock server and verify its responses.

**1. Discovery Endpoint**

This request checks the server's capabilities.

*   **Command:**
    ```bash
    curl http://localhost:8000/mcp
    ```

*   **Expected Output:**
    ```json
    {"mcp_version":"0.1.0","capabilities":{"tools":{"list":true,"call":true}}}
    ```

**2. List Tools**

This request asks the server to list the tools it provides.

*   **Command:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"method": "tools/list"}' http://localhost:8000/mcp
    ```

*   **Expected Output:**
    ```json
    {"tools":[{"name":"ask_opensource_model","description":"Ask a question to an open-source large language model using LiteLLM.","inputSchema":{"type":"object","properties":{"prompt":{"type":"string","description":"The question or prompt to send to the model."},"model":{"type":"string","description":"The model to use (e.g., 'ollama/llama2', 'gpt-3.5-turbo'). Defaults to 'ollama/gpt-oss:20b-cloud'.","default":"ollama/gpt-oss:20b-cloud"}},"required":["prompt"]}}]}
    ```

**3. Call a Tool**

This request invokes the `ask_opensource_model` tool with a specific prompt.

*   **Command:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"method": "tools/call", "name": "ask_opensource_model", "arguments": {"prompt": "hello"}}' http://localhost:8000/mcp
    ```

*   **Expected Output:**
    The mock server will return a canned response.
    ```json
    {"content":"Hello! 🌟 How can I help you today?"}
    ```