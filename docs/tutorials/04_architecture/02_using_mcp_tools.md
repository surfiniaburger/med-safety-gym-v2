# Using MCP Tools

The DIPG Safety Gym exposes 2 robust tools via the Model Context Protocol.

## 1. `get_eval_tasks`
Fetches a sample of questions from the dataset.

*   **Arguments:**
    *   `max_samples` (Integer, default 100): How many questions to get.
    *   `shuffle` (Boolean, default True): specific subset or random.

*   **Returns:** A list of Task objects.

## 2. `evaluate_batch`
Grades a list of answers.

*   **Arguments:**
    *   `evaluations` (List): complex object containing response + ground truth.
    *   `format` (String): "json", "xml", etc.

## Testing with MCP Inspector

You can test these tools without writing code!

1.  Current directory: `med-safety-gym`
2.  Run:
    ```bash
    npx @modelcontextprotocol/inspector python -m server.mcp_server
    ```
3.  Open your browser to the URL provided.
4.  You can manually click "Tools" -> `get_eval_tasks` and see the data flow in real-time.
