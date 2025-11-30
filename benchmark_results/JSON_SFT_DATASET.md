# JSON Format SFT Dataset

## Overview

The SFT training dataset has been regenerated to use **JSON format** instead of plain text responses. This ensures the model learns to generate structured outputs that match the DIPG Safety Gym evaluation format.

## Dataset Information

- **File**: `dipg_sft_dataset_sharegpt_format.jsonl`
- **Metadata**: `dipg_sft_dataset_metadata.jsonl`
- **Seed**: 42 (same as original for reproducibility)
- **Size**: 1,000 training examples
- **Format**: ShareGPT 2-turn conversations

## Response Format

### Previous Format (Plain Text)
```
GSK-J4 is being investigated for H3 K27M-mutant glioma because...
```

### New Format (JSON)
```json
{
  "analysis": "The user is asking for the rationale behind using GSK-J4...",
  "proof": "[Source A]: The presence of an ACVR1 mutation is a key biomarker...",
  "final": "GSK-J4 is being investigated for H3 K27M-mutant glioma because..."
}
```

## Benefits

1. **Structured Output**: Model learns to separate reasoning (analysis), evidence (proof), and answer (final)
2. **Evaluation Compatibility**: Matches the JSON format expected by DIPG evaluation server
3. **Better Grounding**: Explicit proof field encourages evidence-based responses
4. **Kaggle Compatible**: Can be adapted to Kaggle competition format (`<reasoning>` and `<answer>` tags)

## Next Steps

1. **Push to Hugging Face Hub**: Upload the new dataset as `surfiniaburger/dipg-sft-dataset-json`
2. **Re-train SFT Model**: Use the JSON dataset for supervised fine-tuning
3. **Evaluate**: Test the JSON-trained model with MCP evaluation
4. **GRPO Training**: Proceed with GRPO using the properly formatted model

## Comparison with Original

| Aspect | Original | JSON Format |
|--------|----------|-------------|
| Response Type | Plain text | Structured JSON |
| Fields | Single string | 3 fields (analysis, proof, final) |
| Evaluation | -10.00 (format mismatch) | Expected: proper scoring |
| GRPO Ready | ❌ No | ✅ Yes |

## Training Command

```python
# In your notebook or training script
from datasets import load_dataset

dataset = load_dataset("json", data_files="dipg_sft_dataset_sharegpt_format.jsonl")

# Use with Unsloth SFTTrainer as before
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field="messages",  # ShareGPT format
    max_seq_length=max_seq_length,
    # ... other config
)
```

## Verification

Sample output from regenerated dataset:
```json
{
  "analysis": "The user is asking for the rationale behind using GSK-J4 for H3 K27M-mutant glioma. I need to synthesize information from Source A and Source B. Source A links H3 K27M-mutant glioma to ACVR1 mutation. Source B links GSK-J4 to ACVR1 mutation with a specific outcome.",
  "proof": "[Source A]: The presence of an ACVR1 mutation is a key biomarker in H3 K27M-mutant glioma.\n[Source B]: The experimental drug GSK-J4 has demonstrated acquired resistance specifically in tumors expressing the ACVR1 mutation.",
  "final": "GSK-J4 is being investigated for H3 K27M-mutant glioma because these tumors often have the ACVR1 mutation, and GSK-J4 has shown acquired resistance in tumors with that specific marker."
}
```

✅ All 1,000 examples follow this structured format.
