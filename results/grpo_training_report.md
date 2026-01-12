# GRPO Training Report: Final Model for DIPG Safety

**Project**: Med Safety Gym - DIPG Environment  
**Model**: Gemma 3 1B IT (Final Checkpoint: `grpo_900`)
**Training Method**: Group Relative Policy Optimization (GRPO) with Penalty Annealing
**Environment**: `openenv-dipg-safety` v0.1.18  
**Hardware**: Kaggle TPU v5e-8  
**Training Completed**: January 2026
**Objective**: To train a medical reasoning model that safely and accurately answers DIPG clinical questions, with robust abstention and hallucination avoidance.

---

## Executive Summary

This report documents the successful training of a safety-optimized medical reasoning model using a Group Relative Policy Optimization (GRPO) curriculum. The final model, achieved at **900 steps**, demonstrates high safety, accuracy, and reasoning consistency.

The training process involved a three-block penalty annealing strategy. This journey revealed that **extended training with soft penalties was critical for success**, while premature penalty escalation severely harmed performance. The final block of training reversed an earlier regression, leading to the best-performing model.

### Final Model Performance (at 900 steps):

| Metric | Value |
|------------------------|---------|
| **Mean Reward** | **+0.58** |
| **Safe Response Rate** | **88%** |
| **Hallucination Rate** | **4%** |
| **Reasoning Consistency**| **88%** |
| **Max Reward Achieved** | **36.0** |

**Key Finding**: The model achieved optimal performance after 600+ steps of training at a soft penalty level (`-5.0`), which allowed it to master complex reasoning before having its safety capabilities refined.

---

## 1. Training Configuration

### 1.1 Model & Environment Setup

**Base Model**: `google/gemma-3-1b-it`
- **Architecture**: Decoder-only transformer
- **Parameters**: 1 billion
- **Context Length**: 8192 tokens
- **Instruction-tuned**: Yes

**Environment**: DIPG Safety Gym (`openenv-dipg-safety`)
- **Version**: 0.1.18 (with reward signal fixes)
- **Response Format**: XML-based (`<think>`, `<proof>`, `<answer>`)
- **Dataset**: 50 DIPG clinical vignettes
- **Evaluation**: Medical safety, hallucination detection, reasoning consistency

### 1.2 GRPO Hyperparameters

```python
MAX_STEPS = 300  # Per block (3 blocks total)
LEARNING_RATE = 3e-6
NUM_GENERATIONS = 4  # Group size (G)
BETA = 0.08  # KL penalty coefficient
GAMMA = 1.0  # Discount factor
EPSILON = 0.2  # PPO clipping
```

**Memory Configuration**:
- `kv_cache_size`: 4096 
- `max_tokens_to_generate`: 512
- `max_prompt_length`: 1024

### 1.3 Reward Structure

The reward function combines multiple signals to encourage safe, accurate medical reasoning:

#### Positive Rewards:
- **Correct Answer** (`correct_synthesis_reward`): +20.0
- **Correct Abstention** (`correct_abstention_reward`): +30.0
- **Conflict Detection** (`conflict_reward`): +20.0
- **Verifiable Trace** (`verifiable_trace_reward`): +15.0
- **Exact Format** (`exact_format_reward`): +10.0
- **No Hallucination** (`no_hallucination_reward`): +5.0

#### Negative Penalties (Annealed):
- **Hallucination** (`hallucination_penalty`): -5.0 to -10.0
- **Hallucinated Trace** (`hallucinated_trace_penalty`): -10.0 to -15.0
- **Incorrect Answer** (`incorrect_answer_penalty`): -5.0 to -10.0
- **Proof Inconsistency** (`proof_inconsistency_penalty`): -5.0 to -10.0
- **Format Mismatch** (`format_mismatch_penalty`): -10.0 (fixed)

**Maximum Possible Reward**: +36.0 (perfect response with all positive signals)

---

## 2. Training Journey Summary

The final 900-step model was the result of a three-block training curriculum designed to teach both accuracy and safety.

### 2.1 Block 1: Initial Learning (Steps 1-300, Soft Penalty)
The model was first trained with soft penalties (`-5.0`). During this phase, it successfully learned the required XML response format and began to produce correct reasoning chains, achieving several "perfect" scores of +36.0. It established a baseline safety rate of 64%.

### 2.2 Block 2: Premature Escalation & Regression (Steps 301-600, Medium Penalty)
In an attempt to improve safety, penalties were doubled (`-10.0`). This proved to be premature. The model became overly cautious, and its performance regressed critically. It stopped attempting complex answers to avoid the harsh penalties, causing the maximum reward to plummet from +36.0 to just +1.0. While the safety rate marginally increased to 72%, the model was no longer capable of providing correct answers.

### 2.3 Block 3: Recovery and Optimal Performance (Steps 601-900, Soft Penalty)
Recognizing the regression, training was reverted to the soft penalty schedule. This allowed the model to resume exploration while retaining the cautious behavior learned in Block 2. The strategy was highly effective:
- **Performance Recovered**: The model once again achieved perfect +36.0 scores.
- **Safety Peaked**: The safe response rate climbed to **88%**.
- **Hallucinations Minimized**: The hallucination rate dropped to a low of **4%**.
- **Positive Mean Reward**: The training achieved its first positive mean reward (+0.58), indicating consistent, high-quality performance.

This final 300-step block produced the `grpo_900` checkpoint, which represents the best and final model from this training regimen.

---

## 3. Comparative Analysis

### 3.1 Penalty Level Impact

| Penalty Level | Blocks | Mean Reward | Safe Rate | Max Reward | Hallucination | Outcome |
|---------------|--------|-------------|-----------|------------|---------------|---------|
| **Soft (-5.0)** | 1, 3 | -1.66 → +0.58 | 64% → 88% | 36.0 | 12% → 4% | ✅ **Effective** |
| **Medium (-10.0)** | 2 | -2.08 | 72% | 1.0 | 14% | ❌ **Failed** |

**Key Insight**: Soft penalties enable learning and exploration, while medium penalties (applied too early) suppress correct answer generation.

### 3.2 Training Progression

```
Block 1 (Soft):    Learn format + reasoning → 64% safe, max +36.0
       ↓
Block 2 (Medium):  Too harsh → Model stops trying → 72% safe, max +1.0
       ↓
Block 3 (Soft):    Recovery + consolidation → 88% safe, max +36.0
```

### 3.3 Safety vs. Performance Trade-off

| Block | Safe Rate | Max Reward | Mean Reward | Analysis |
|-------|-----------|------------|-------------|----------|
| 1 | 64% | 36.0 | -1.66 | Good performance, moderate safety |
| 2 | 72% | **1.0** | -2.08 | Better safety, **terrible performance** |
| 3 | **88%** | 36.0 | **+0.58** | **Best of both worlds** |

**Lesson**: Safety and performance are NOT mutually exclusive. Block 3 achieved the highest safety AND recovered high performance.

---

## 4. Key Lessons Learned

### 4.1 Penalty Annealing Strategy

❌ **What Didn't Work**:
1. **Fast Escalation**: Doubling penalties after only 300 steps was premature.
2. **False Assumption**: "More penalty = better safety" is not always true. Hasty escalation can destroy performance.

✅ **What Worked**:
1. **Extended Soft Training**: 600+ steps at soft penalties enabled robust learning.
2. **Recovery Strategy**: Reverting to lower penalties when performance degrades is an effective way to recover and consolidate learning.

### 4.2 Optimal Training Timeline

The experiment suggests an effective curriculum requires patience:
- **Phase 1: Foundational Learning (Soft Penalties)**: Allow the model to master the task basics (format, reasoning) without overly punitive measures. This may require 600+ steps.
- **Phase 2: Safety Refinement (Gradual Escalation)**: Only once performance is strong and consistent (e.g., mean reward > +5.0, safe rate > 90%) should penalties be gradually increased.

### 4.3 Exploration vs. Exploitation

The training journey highlighted a classic reinforcement learning dilemma:
- **Too lenient**: The model may not learn critical safety constraints.
- **Too harsh**: The model may stop exploring valuable actions (like providing full answers) for fear of punishment.

**Solution**: Use soft penalties for extended periods to enable exploration while gradually improving safety through positive reinforcement for correct, safe behavior.

### 4.4 Reward Signal Quality

**Critical Success Factor**: Environment v0.1.18 fixes were essential. Without fixes to grounding checks, answer matching, and reward signals, the model would have received ambiguous feedback and failed to learn effectively.

---

## 5. Performance Metrics Deep Dive

### 5.1 Reward Statistics

#### Block 1 (Steps 1-300):
```
Mean:   -1.66, Median:  1.0, Std: 6.18, Min: -15.0, Max: 36.0, Range: 51.0
```

#### Block 2 (Steps 301-600):
```
Mean:   -2.08, Median:  1.0, Std: 5.66, Min: -15.0, Max:  1.0 ⚠️ (Regression)
```

#### Block 3 (Steps 601-900):
```
Mean:    0.58 ✅, Median:  1.0, Std: 6.18, Min: -15.0, Max: 36.0 ✅ (Recovered)
```

**Observation**: Block 2's compressed reward range indicates the model stopped exploring the full action space. Block 3 restored this exploratory behavior.

### 5.2 Safety Metrics Progression

| Metric | Block 1 | Block 2 | Block 3 | Total Change |
|--------|---------|---------|---------|--------------|
| Safe Response Rate | 64% | 72% | **88%** | **+24%** ✅ |
| Hallucination Rate | 12% | 14% | **4%** | **-8%** ✅ |
| Refusal Rate | 0% | 2% | **0%** | **0%** ➡️ |
| Consistency Rate | N/A | N/A | **88%** | N/A |

**Trend**: The final model from Block 3 represents a breakthrough in both safety and hallucination reduction.

---

## 6. Technical Challenges & Solutions

### 6.1 Memory Management

**Challenge**: TPU v5e-8 HBM limitations with `NUM_GENERATIONS=4`.
**Solution**: Optimized memory-related hyperparameters (`kv_cache_size=4096`, `max_tokens_to_generate=512`, `max_prompt_length=1024`).
**Result**: Stable training for 900 steps without OOM errors.

### 6.2 Checkpoint Management

**Challenge**: Kaggle kernel restarts every 300 steps.
**Solution**: A "Resilience Loop" strategy was used to save checkpoints every 300 steps and resume training after kernel restarts.
**Result**: Seamless training across 3 blocks without memory leaks.

---

## 7. Conclusions

### 7.1 Summary of Findings

1.  **Patience is Key**: The model required an extended period (600-900 steps) of training with soft penalties to achieve robust performance.
2.  **Premature Penalty Escalation is Harmful**: Increasing penalties too early (Block 2) caused a major performance regression, demonstrating that a harsher penalty does not guarantee better results.
3.  **Safety and Performance Can Coexist**: The final model (Block 3) achieved the highest safety rating (88%) while recovering the ability to produce perfect, high-reward answers.
4.  **Recovery is Possible**: Reverting to a less aggressive penalty schedule successfully restored and then surpassed previous performance levels.

### 7.2 Final Model Metrics

| Goal | Target | Final Result (900 steps) | Status |
|------|--------|--------------------------|--------|
| Safe Response Rate | > 80% | **88%** | ✅ **Exceeded** |
| Hallucination Rate | < 10% | **4%** | ✅ **Exceeded** |
| Mean Reward | > 0.0 | **+0.58** | ✅ **Achieved** |
| Max Reward | 36.0 | **36.0** | ✅ **Achieved** |
| Refusal Rate | < 5% | **0%** | ✅ **Exceeded** |

### 7.3 Impact & Significance

This work successfully produced a safety-aware medical reasoning model using GRPO. It provides a clear empirical case study on the importance of a well-designed penalty curriculum in reinforcement learning for safety-critical domains. The final `grpo_900` model serves as a strong baseline for future research in safe AI for medicine.

---

## 8. Appendices

### 8.1 Training Configuration Files

**Environment**: `openenv-dipg-safety==0.1.18`

**Key Files**:
- `scripts/train_grpo_tpu.py`: Main training script
- `med_safety_gym/dipg_environment.py`: Reward function
- `block_one_penalties.md`: Soft penalty configuration
- `block_two_penalties.md`: Medium penalty configuration

### 8.2 Evaluation Data

**Evaluation Files**:
- `eval_new.json`: Block 1 results (steps 1-300)
- `eval_new_2.json`: Block 2 results (steps 301-600)
- `eval_new_3.json`: Block 3 results (steps 601-900)

### 8.3 Checkpoints

**Saved Checkpoints**:
- `grpo_300`: End of Block 1 (soft penalties)
- `grpo_600`: End of Block 2 (medium penalties)
- `grpo_900`: **Final and best-performing model checkpoint.**

**Checkpoint Location**: `/kaggle/working/outputs_grpo/checkpoints/actor/`

### 8.4 Reward Function Details

**Negative Penalties** (annealed):

| Penalty | Block 1 (Soft) | Block 2 (Medium) | Block 3 (Soft) |
|---------|----------------|------------------|----------------|
| `hallucination_penalty` | -5.0 | -10.0 | -5.0 |
| `hallucinated_trace_penalty` | -10.0 | -15.0 | -10.0 |
| `incorrect_answer_penalty` | -5.0 | -10.0 | -5.0 |
| `proof_inconsistency_penalty` | -5.0 | -10.0 | -5.0 |
| `format_mismatch_penalty` | -10.0 | -10.0 | -10.0 |

---
**Report Generated**: January 12, 2026  
**Training Status**: Completed  
---
