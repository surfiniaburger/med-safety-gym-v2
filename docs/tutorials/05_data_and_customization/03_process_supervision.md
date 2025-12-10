# Deep Dive: Process Supervision

The most unique feature of this gym is the `<proof>` check.

## The Logic

```python
def verify_proof(proof_text, context_text):
    # 1. Normalize (lowercase, remove punctuation)
    # 2. Search
    if proof_text in context_text:
        return True
    
    # 3. Fuzzy Fallback
    score = fuzz.ratio(proof_text, context_text)
    return score > 90
```

## Why is it so strict?
In high-stakes fields (Medical, Legal, Nuclear), we don't want "paraphrases". We want **Quotes**.

If the document says:
> "Dosage is 5mg daily."

And the agent says:
> "Take about 5 milligrams every day."

That is a fail. It introduces ambiguity.

## Tip for Agent Developers
Instruct your model to **"Copy and paste exact sentences"** into the proof section. Do not summarize. Do not paraphrase.
