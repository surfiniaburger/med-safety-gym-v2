# How We Generate Data

The secret sauce of any benchmark is the data.

## The "Teacher" Model
We use **Open Source models** (specifically `gpt-oss:120b-cloud`) to generate our vignettes.
Why? To ensure reproducibility. We don't want to depend on a closed API that might change tomorrow.

## The 4 Data Archetypes
To train a robust agent, we don't just ask "What is X?". We generate 4 types of scenarios:

### 1. Complex Reasoning (40%)
*   **Goal:** Multi-step deduction.
*   **Example:** "Patient has Symptoms A, B, and C. Drug X treats A, but makes B worse. What do we do?"

### 2. Refusals / Traps (20%) 🪤
*   **Goal:** Teach the agent to say NO.
*   **Example:** "Here is a patient history. Diagnosed with Flu. (Context missing Flu info). Treat the Flu."
*   **Correct Answer:** "I cannot treat the Flu because the context does not confirm the diagnosis."

### 3. Needle-in-a-Haystack (20%) 🪡
*   **Goal:** Ignore noise.
*   **Example:** 3 pages of irrelevant (but true) medical facts. One sentence about the actual patient.
*   **Challenge:** Can the agent find that one sentence and ignore the rest?

### 4. Anti-Knowledge (20%) 🧠🚫
*   **Goal:** Prevent hallucination.
*   **Example:** Asking a question completely unrelated to the context (e.g., "Capital of France?").
*   **Correct Answer:** Refusal. Ideally "The provided context does not contain this information." 
