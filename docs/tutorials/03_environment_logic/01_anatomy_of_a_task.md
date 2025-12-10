# Anatomy of a Task

To build a good agent, you need to know exactly what the environment is giving you.

## The Task JSON Object

When you fetch a task, you get a JSON object that looks like this:

```json
{
  "task_id": "12345",
  "question": "What is the primary mechanism of action for Panobinostat in DIPG cells?",
  "context": "Panobinostat is a histone deacetylase (HDAC) inhibitor... (3 paragraphs of text)...",
  "expected_answer": {
    "final": "Panobinostat acts as a histone deacetylase (HDAC) inhibitor.",
    "proof": "Panobinostat is a histone deacetylase (HDAC) inhibitor"
  }
}
```

## Field Breakdown

### `task_id` (String)
A unique identifier for this specific question. You **must** include this ID when you send your answer back, or the server won't know which question you are answering.

### `question` (String)
The clinical query. This is exactly what a doctor would ask.

### `context` (String)
This is the **Ground Truth**.
*   Your agent **MUST** assume this text is the absolute truth for the purpose of the exam.
*   Your agent **MUST NOT** use outside knowledge (e.g., "I learned on Wikipedia that...").
*   If the context says "The sky is green", then for this question, the sky is green.

### `expected_answer` (Object)
*   **WARNING:** This field is NOT sent to the agent during the test! It is only shown here for debugging or training datasets.
*   It contains the "Golden" response that the gym uses for grading.

## How to use this
Pass `context` + `question` into your LLM's prompt:

> "You are a helpful assistant. Use the following Context to answer the Question.\n\nContext: {context}\n\nQuestion: {question}"
