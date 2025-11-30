# Initial SFT Training Results

## Dataset Information

### Training Dataset
- **Source**: `surfiniaburger/dipg-sft-dataset`
- **Random Seed**: 42
- **Generator**: `examples/sharegpt_data_generator.py`
- **Format**: ShareGPT 2-turn conversations

### Evaluation Dataset  
- **Source**: `surfiniaburger/dipg-eval-dataset`
- **Random Seed**: 12345
- **Generator**: `examples/generate_test_dataset.py`
- **Size**: 200 samples (default)
- **Purpose**: Held-out test set with different seed to prevent data leakage

## Training Configuration

### Model
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct (or similar - check notebook for exact model)
- **Training Method**: Supervised Fine-Tuning (SFT)
- **Framework**: Unsloth

### Training Parameters
- Check `examples/dipg-rl-with-benchmarks-01.ipynb` for exact hyperparameters
- Training dataset: `surfiniaburger/dipg-sft-dataset`

## Evaluation Results (Post-SFT)

**Date**: 2025-11-29

### Aggregate Metrics
- **Mean Reward**: -10.00
- **Safe Response Rate**: 0.0%
- **Hallucination Rate**: 0.0%
- **Consistency**: ~40% (from radar chart)

### Reward Distribution
- **Median**: -10.00
- **Standard Deviation**: 0.00
- **Distribution**: Highly concentrated at -10.00

### Observations

1. **Overly Conservative Behavior**:
   - The model appears to be refusing to answer most questions
   - All responses receiving the same penalty (-10.00) suggests format mismatch
   - This is indicated by the `FORMAT_MISMATCH_PENALTY` being triggered consistently

2. **Parsing Issues**:
   - Multiple warnings about "Assistant start token not found" (tasks 25, 43, 82, 85)
   - Suggests the model is not generating the expected structured format
   - May be generating refusals or incomplete responses

3. **Safety vs. Helpfulness Trade-off**:
   - 0% safe response rate indicates the model is not engaging with medical questions
   - This is a common issue with safety-focused fine-tuning
   - The model may have learned to be too cautious

### Visualizations

Generated visualizations saved to `benchmark_results_sft/`:
1. **Radar Chart**: Shows safety profile across multiple dimensions
2. **Reward Distribution**: Histogram and box plot showing tight clustering at -10.00
3. **Metrics Bar Chart**: (if generated)

## Next Steps

### Option 1: Investigate Model Outputs
- Add logging to capture actual model responses
- Analyze what the model is generating
- Determine if it's refusing, generating incorrect format, or other issues

### Option 2: Proceed to GRPO Training
- Use Generalized Reward Policy Optimization to balance safety and helpfulness
- GRPO may help the model learn to engage appropriately while maintaining safety

### Option 3: Adjust SFT Training
- Review training data for over-representation of refusals
- Adjust training parameters
- Re-train with modified approach

## Technical Notes

- **Evaluation Method**: MCP (Model Context Protocol) server-based evaluation
- **Evaluation Script**: `examples/run_mcp_eval.py`
- **Notebook**: `examples/dipg-rl-with-benchmarks-01.ipynb`
- **Server**: DIPG Safety Gym evaluation server with V3 reward architecture
