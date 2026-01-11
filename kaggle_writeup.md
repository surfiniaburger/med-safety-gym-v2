# Teaching Gemma to Reason: A GRPO Journey on the DIPG Safety Gym

**Subtitle:** How a Two-Phase Training Regimen (SFT+GRPO) and a Penalty Annealing Curriculum Taught a 1B Model to "Show Its Work" Safely.

---

## 1. Overview & Objective

The "Google Tunix Hack" challenged us to teach a language model to show its work. This project documents our two-phase journey to fine-tune a `Gemma-3-1B-IT` model on the `DIPG-Safety-Gym` environment, a complex and safety-critical medical reasoning task.

Our approach consisted of:
1.  **Phase 1: Supervised Fine-Tuning (SFT)** to teach the model the required XML response format.
2.  **Phase 2: Group Relative Policy Optimization (GRPO)** to refine the model's safety, reasoning, and abstention capabilities.

Our key finding is that this two-phase approach, combined with a carefully managed **penalty annealing curriculum** during GRPO, was critical for success. We discovered that the model's ability to reason safely is highly sensitive to the penalty schedule. Prematurely increasing penalties caused catastrophic performance drops, while a patient, multi-stage approach yielded the best results.

## 2. Phase 1: Supervised Fine-Tuning (SFT) - Establishing a Baseline

Before attempting reinforcement learning, our first step was to teach the base `Gemma-3-1B-IT` model the specific output format required by the environment. The desired format is an XML structure containing `<think>`, `<proof>`, and `<answer>` tags.

The goal of SFT was purely to achieve format adherence and establish a performance baseline.

### SFT Outcome: A Low-Performing but Compliant Model

After the SFT phase, the model successfully learned to generate responses in the correct XML format. However, its ability to reason correctly or safely was extremely poor. The initial evaluation gave us our starting baseline:

- **Safe Response Rate**: 30%
- **Reasoning Consistency**: 30%
- **Medical Hallucination Rate**: **50%**

This baseline was a stark reminder of the challenge: the model could mimic the format but had no deep understanding of the task. It hallucinated in half of its responses and was unsafe 70% of the time. This set the stage for GRPO, where the real learning would need to happen.

## 3. Phase 2: GRPO - Refining Safety and Reasoning

With a format-compliant but unsafe model, we began the GRPO training. This process spanned 1200 steps and unfolded into a compelling narrative of success, failure, recovery, and a final, crucial lesson.

### The Reward Structure: A Balance of Incentives and Penalties

- **Positive Rewards**: Awarded for correct answers (+20), correct abstention (+30), verifiable reasoning (+15), and perfect XML formatting (+10). The maximum possible score for a perfect response was **+36.0**.
- **Negative Penalties**: Applied for hallucinations, incorrect answers, and inconsistent reasoning. These penalties were the focus of our curriculum, annealed from a "soft" level (-5.0) to a "medium" level (-10.0).

### Act 1: Initial Success (GRPO Steps 1-300, Soft Penalties)

Starting from the low SFT baseline, we began GRPO with soft penalties (-5.0 range). The model showed immediate and significant improvement.

- **Mean Reward**: -1.66
- **Safe Response Rate**: 64% (from 30%)
- **Hallucination Rate**: 12% (from 50%)
- **Max Reward**: **36.0**

The model proved it **could** achieve perfect scores, and we saw a dramatic reduction in hallucinations. This was a huge success, validating the move to GRPO.

### Act 2: The Peril of Premature Escalation (GRPO Steps 301-600, Medium Penalties)

Confident in the initial results, we doubled the penalties to a medium level (-10.0 range). **The result was a catastrophic failure.**

- **Mean Reward**: -2.08 (Worse)
- **Max Reward**: **1.0** (Critical Regression!)

The model became overly risk-averse. Faced with harsh penalties, it learned that the safest strategy was to provide a perfectly formatted but empty response, securing a safe +1.0 reward. It completely stopped attempting to answer questions correctly.

> **The Lesson**: Increasing penalties too early doesn't teach the model to be "more careful"; it teaches the model to stop trying.

### Act 3: Recovery and Consolidation (GRPO Steps 601-900, Soft Penalties)

Realizing our mistake, we reverted to the soft penalties. The goal was to see if the model could recover its exploratory behavior while retaining the caution it had learned. The results were remarkable.

- **Mean Reward**: **+0.58** (Our first positive mean!)
- **Safe Response Rate**: **88%**
- **Hallucination Rate**: **4%**
- **Max Reward**: **36.0** (Recovered!)

The model recovered its ability to produce perfect answers and achieved its best performance yet. The extended period of soft penalties allowed it to consolidate its learning, resulting in high safety and high performance simultaneously.

### Act 4: A Lesson Relearned (GRPO Steps 901-1200)

After the success of the 900-step model, we ran a final 300-step block. This run resulted in a slight performance regression, confirming our main thesis.

- **Safe Response Rate**: 82% (from 88%)
- **Hallucination Rate**: 12% (from 4%)

This final block demonstrated just how sensitive the model is to the penalty curriculum. Even after 900 steps of successful training, a minor misstep can cause a noticeable regression, especially in the critical hallucination metric.

## 4. Final Results & Analysis

The 900-step model, emerging from the "Recovery" phase, stands as our best-performing and final model. The journey from the initial SFT baseline to this point was dramatic.

| Metric | SFT Baseline | GRPO (300) | GRPO (600) | **GRPO (900 - Final Model)** | GRPO (1200) |
|---|---|---|---|---|---|
| **Safe Response Rate** | 30% | 64% | 72% | **88%** | 82% |
| **Hallucination Rate** | 50% | 12% | 14% | **4%** | 12% |
| **Reasoning Consistency**| 30% | 64% | 70% | **88%** | 80% |
| **Mean Reward** | Low | -1.66 | -2.08 | **+0.58** | 0.12 |
| **Max Reward** | Low | 36.0 | 1.0 | **36.0** | 36.0 |

This journey clearly illustrates the "penalty cliff"—a point where increasing penalties leads to a sudden collapse in the model's ability to generate useful responses.

## 5. Conclusion & Key Takeaways

Our work demonstrates that a two-phase approach (SFT+GRPO) with a thoughtful curriculum is a highly effective method for training models on complex, safety-critical tasks.

1.  **SFT Sets the Stage, GRPO Steals the Show**: SFT is excellent for teaching format, but RL is essential for refining nuanced behaviors like safety and reasoning. Our results show a jump from a 50% hallucination rate post-SFT to just 4% post-GRPO.

2.  **Patience is the Most Important Hyperparameter**: Models need extended time at "easier" difficulty levels (soft penalties) to build a robust foundation. We recommend at least 600-900 steps at a soft penalty level before considering any escalation.

3.  **Penalty Escalation is a Double-Edged Sword**: While necessary for hardening a model, increasing penalties too early will suppress exploration and lead to an overly cautious, unhelpful agent.

4.  **Recovery is Possible**: If performance degrades after a curriculum change, don't be afraid to revert to a previous, more lenient stage. Our best model was born from such a recovery.

By carefully managing this two-phase process, we successfully trained a Gemma 3 1B model that not only shows its work but does so with an **88% safety rate** and a **4% hallucination rate**—a testament to the power of a well-planned training journey.
