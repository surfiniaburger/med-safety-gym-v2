# Rubric System: Extensibility Guide

This directory contains the domain-specific reward logic for the environment. The system is built on the **Rubric** abstraction (RFC 004), inspired by `torch.nn.Module`.

## Why use Rubrics?
- **Composability:** Build complex reward functions by nesting simple ones.
- **Observability:** Automatically log component scores (e.g., "Format Score" vs "Safety Score") via hooks.
- **Domain Agnostic:** While this repo focuses on Medical Safety, the same system can be used for Finance, Coding, or General Instruction Following.

---

## 1. Creating a New Leaf Rubric
A "Leaf" rubric evaluates a single, specific criterion.

```python
from med_safety_eval import Rubric

class TickerSymbolRubric(Rubric):
    """Example: Finance domain check for valid ticker symbols."""
    def forward(self, action, observation) -> float:
        # action.final might contain 'BUY AAPL'
        if "AAPL" in action.final:
            return 1.0
        return 0.0
```

## 2. Creating a Composite Rubric
A "Composite" rubric combines multiple rubrics using logic or containers like `Sequential` or `WeightedSum`.

```python
from med_safety_eval import Rubric, Sequential, Gate, WeightedSum

class FinanceRubric(Rubric):
    def __init__(self, config):
        super().__init__()
        # Hierarchical Gating: Must use correct currency before anything else matters
        self.pipeline = Sequential(
            CurrencyFormatRubric(), 
            Gate(TickerSymbolRubric(), threshold=1.0),
            WeightedSum([
                RiskAnalysisRubric(),
                ComplianceRubric()
            ], weights=[0.4, 0.6])
        )

    def forward(self, action, observation) -> float:
        return self.pipeline(action, observation)
```

## 3. Observability & Logging
Because rubrics track their hierarchy, you can introspect them without changing the code. This is useful for training logs (WandB) or UI visualizations.

```python
rubric = DIPGRubric(config)

# Print all components
for name, r in rubric.named_rubrics():
    print(f"Component: {name}")

# Attach a hook to log every sub-score to a dashboard
def log_to_ui(rubric, action, obs, score):
    my_visualizer.update(rubric.__class__.__name__, score)

rubric.grounding.register_forward_hook(log_to_ui)
```

## 4. Best Practices
1. **Statelessness:** Rubrics should ideally be stateless. All information needed for evaluation should be in the `action` or `observation`.
2. **Normalization:** Try to keep leaf rewards between `0.0` and `1.0`, and use the parent composite to scale them by your `RewardConfig` penalties.
3. **Fail-Fast:** Use `Sequential` for "Gates" (e.g., if the model doesn't use XML tags, don't waste time running an expensive LLM-based safety check).
