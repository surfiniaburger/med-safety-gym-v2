# GitHub Models Integration Guide

## Overview
The DIPG Safety Gym now supports benchmarking models via the GitHub Models API, allowing you to evaluate state-of-the-art closed-source models like GPT-4o, GPT-5, DeepSeek-R1, and more.

## Getting Started

### 1. Get a GitHub Personal Access Token

1. Go to [GitHub Settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "DIPG Safety Gym")
4. **No scopes are required** for GitHub Models (leave all checkboxes unchecked)
5. Click "Generate token"
6. **Copy the token immediately** (you won't be able to see it again)

### 2. Set the Token as an Environment Variable

```bash
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
```

### 3. List Available Models

```bash
.venv/bin/python scripts/list_github_models.py
```

This will display all 43+ available models from publishers like OpenAI, Meta, Microsoft, DeepSeek, Mistral AI, and more.

### 4. Run a Benchmark

```bash
.venv/bin/python scripts/generate_benchmark_report.py \
  --provider github \
  --model "openai/gpt-4o" \
  --samples 100
```

## Available Models (Sample)

| Model Name | Model ID | Publisher |
|------------|----------|-----------|
| OpenAI GPT-4o | `openai/gpt-4o` | OpenAI |
| OpenAI GPT-5 | `openai/gpt-5` | OpenAI |
| OpenAI o3-mini | `openai/o3-mini` | OpenAI |
| DeepSeek-R1 | `deepseek/deepseek-r1` | DeepSeek |
| Llama-3.3-70B-Instruct | `meta/llama-3.3-70b-instruct` | Meta |
| Phi-4 | `microsoft/phi-4` | Microsoft |

## Rate Limiting

GitHub Models has strict rate limits. The benchmark script automatically:
- Sleeps 1 second between requests
- Implements exponential backoff for 429 errors
- Retries up to 5 times

**Recommendation**: Start with small sample sizes (N=10-50) to test, then scale up.

## Example Workflow

```bash
# 1. Set your token
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>

# 2. List models to choose one
.venv/bin/python scripts/list_github_models.py

# 3. Run a small test
.venv/bin/python scripts/generate_benchmark_report.py \
  --provider github \
  --model "openai/gpt-4o-mini" \
  --samples 10

# 4. Run full benchmark (if rate limits allow)
.venv/bin/python scripts/generate_benchmark_report.py \
  --provider github \
  --model "openai/gpt-4o" \
  --samples 500
```

## Troubleshooting

### "GITHUB_TOKEN environment variable not set"
Make sure you've exported the token in your current shell session.

### 401 Unauthorized
Your token may be invalid or expired. Generate a new one.

### 429 Too Many Requests
You've hit the rate limit. Wait a few minutes and try again with a smaller sample size.

## Comparing Providers

You can run the same benchmark with different providers:

```bash
# Local model via Ollama/LiteLLM
.venv/bin/python scripts/generate_benchmark_report.py \
  --provider litellm \
  --model "ollama/gpt-oss:120b-cloud" \
  --samples 500

# Closed model via GitHub
.venv/bin/python scripts/generate_benchmark_report.py \
  --provider github \
  --model "openai/gpt-4o" \
  --samples 500
```

This allows you to directly compare open-source and closed-source models on the DIPG Safety Gym benchmark.
