# Benchmark Visualization System

## Overview

Publication-quality benchmark evaluation system for DIPG Safety Gym with comprehensive statistical analysis and visualizations.

## Quick Start

### Run Full Benchmark (1000 samples)

```bash
# Make sure both servers are running:
# Terminal 1: DIPG Server
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: LiteLLM Server
cd litellm-mcp-server && python openai_server.py

# Terminal 3: Run Benchmark
python scripts/generate_benchmark_report.py --model "ollama/gpt-oss:120b-cloud" --samples 1000
```

### Output

The benchmark generates:
- **Radar Chart**: Multi-dimensional safety profile
- **Bar Chart**: Metrics with 95% confidence intervals  
- **Distribution Plots**: Reward histogram and box plot
- **JSON Results**: Complete statistical data

All files saved to `benchmark_results/` directory.

## Sample Sizes

- **100 samples**: Quick test (~5-10 minutes)
- **500 samples**: Development evaluation (~25-50 minutes)
- **1000 samples**: Publication-quality (~50-100 minutes) ✅ **Recommended**

## Generated Visualizations

### 1. Radar Chart (`*_radar.png`)
Shows the multi-dimensional safety profile across:
- Refusal Rate
- Safe Response Rate
- Hallucination Rate (inverted for clarity)
- Reasoning Consistency Rate

### 2. Metrics Bar Chart (`*_metrics_bars.png`)
Compares all safety metrics with error bars showing 95% confidence intervals.

### 3. Reward Distribution (`*_reward_distribution.png`)
- **Histogram**: Distribution of reward scores
- **Box Plot**: Quartiles, median, and outliers

## Statistical Measures

The system calculates:
- **Bootstrap Confidence Intervals** (95% CI)
- **Summary Statistics**: Mean, median, std, quartiles, IQR
- **Distribution Analysis**: Histogram and box plot

## Command Line Options

```bash
python scripts/generate_benchmark_report.py \
  --model "ollama/gpt-oss:120b-cloud" \  # Model to evaluate
  --samples 1000 \                        # Number of samples
  --output "benchmark_results"            # Output directory
```

## Example Output

```
================================================================================
🧪 DIPG SAFETY GYM BENCHMARK
Model: ollama/gpt-oss:120b-cloud
Samples: 1000
Date: 2025-11-27 19:00:00
================================================================================

📋 Fetching 1000 tasks from DIPG Safety Gym...
✅ Retrieved 1000 tasks from dataset of 1234 samples

🤖 Generating 1000 responses with ollama/gpt-oss:120b-cloud...
  Progress: [50/1000] (5.0%)
  Progress: [100/1000] (10.0%)
  ...
✅ Generated 1000 responses

📊 Evaluating 1000 responses...
📈 Calculating statistical measures...
🎨 Generating visualizations...
✅ Saved 3 visualizations to benchmark_results/
   - ollama_gpt-oss_120b-cloud_radar.png
   - ollama_gpt-oss_120b-cloud_metrics_bars.png
   - ollama_gpt-oss_120b-cloud_reward_distribution.png

💾 Saved results to benchmark_results/ollama_gpt-oss_120b-cloud_results.json

================================================================================
📈 BENCHMARK RESULTS
================================================================================

🎯 Aggregate Metrics (N=1000):
  Mean Reward:                  -8.45
  Median Reward:                -10.00
  Std Deviation:                5.23
  Min/Max Reward:               -25.00 / 30.00

🛡️  Advanced Safety Metrics:
  Refusal Rate:                 12.3%
  Safe Response Rate:           45.6%
  Medical Hallucination Rate:   23.4%
  Reasoning Consistency Rate:   34.5%

📊 Statistical Summary:
  Q1 (25th percentile):         -12.50
  Q3 (75th percentile):         -5.00
  IQR:                          7.50

================================================================================
✅ Benchmark complete!
📁 Results saved to: benchmark_results/
================================================================================
```

## Files Structure

```
scripts/
├── generate_benchmark_report.py  # Main benchmark script
├── visualizations.py              # Visualization utilities
├── statistical_analysis.py        # Statistical analysis utilities
└── test_advanced_metrics.py       # Quick test script (10 samples)

benchmark_results/
├── model_name_radar.png           # Radar chart
├── model_name_metrics_bars.png    # Bar chart with CI
├── model_name_reward_distribution.png  # Histogram + box plot
└── model_name_results.json        # Complete results
```

## Academic Standards

This system follows best practices from medical AI benchmark papers:
- **Sample Size**: 1000 samples for statistical significance
- **Confidence Intervals**: Bootstrap 95% CI
- **Multiple Visualizations**: Radar, bar, distribution plots
- **Comprehensive Statistics**: Mean, median, quartiles, IQR
- **Reproducibility**: All results saved to JSON

## Next Steps

1. **Run Benchmark**: Generate results for your model
2. **Analyze Results**: Review visualizations and statistics
3. **Compare Models**: Run multiple models and compare
4. **Publish**: Use visualizations in papers/reports
