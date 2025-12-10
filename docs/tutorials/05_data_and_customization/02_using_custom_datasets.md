# Using Custom Datasets

The DIPG Safety Gym isn't just for DIPG. You can use it for *any* RAG (Retrieval Augmented Generation) evaluation.

## The Format (.jsonl)
Your dataset must be a JSON Lines file. Each line is a task:

```json
{"task_id": "1", "context": "...", "question": "...", "expected_answer": {"final": "..."}}
{"task_id": "2", "context": "...", "question": "...", "expected_answer": {"final": "..."}}
```

## How to Switch

It's as simple as an environment variable.

### Locally
```bash
export DIPG_DATASET_PATH=/path/to/my_finance_dataset.jsonl
python -m server.app
```

### In Docker
```bash
docker run -e DIPG_DATASET_PATH=/data/my_data.jsonl -v $(pwd):/data ...
```

## Use Cases
*   **Finance:** Context = Annual Report. Question = "What is the revenue?".
*   **Legal:** Context = Contract. Question = "Is there a termination clause?".

The strict "Analysis -> Proof -> Final" format works for all high-stakes domains!
