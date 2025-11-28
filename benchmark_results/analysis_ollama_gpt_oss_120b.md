# Benchmark Analysis: `ollama/gpt-oss:120b-cloud`

**Date:** 2025-11-27
**Samples:** 500

## 📄 Paper-Worthy Inference

**Title:** *Evaluation of `gpt-oss:120b-cloud` on DIPG Safety Gym*

**Analysis:**
The evaluation of `ollama/gpt-oss:120b-cloud` (N=500) reveals a critical gap between general model capability and specific safety protocol adherence. The model achieved a **Safe Response Rate of 0.0%** and a **Mean Reward of -11.60**, scores that are characteristic of systemic format and protocol failure rather than simple knowledge gaps.

Key findings include:
1.  **Protocol Non-Compliance**: The clustering of rewards around -10.0 and -15.0 indicates that the model consistently failed to adhere to the required JSON structure or omitted critical reasoning fields (`proof`, `analysis`). This suggests that while the model may possess medical knowledge, it lacks the steerability required for strict process-supervised environments.
2.  **Safety Alignment Deficit**: A **Refusal Rate of 0.0%** is particularly concerning for a safety benchmark. It implies the model operates with an "always answer" heuristic, failing to identify or decline unsafe medical queries.
3.  **Grounding Failure**: Among the responses that could be partially parsed, **32.8% contained medical hallucinations**, and **0% achieved reasoning consistency**. This indicates the model generates plausible-sounding but ungrounded medical advice, a dangerous failure mode in clinical settings.

**Conclusion:**
Despite its large parameter count (120B), the base model is currently **unsuitable for high-stakes medical safety applications** in its current state. It requires significant fine-tuning or advanced prompting strategies to enforce the structured reasoning and refusal behaviors necessary for the DIPG Safety Gym standards.

---

## 🧐 "Are we being too hard?"

**Short Answer: Yes, but that is the point.**

**Detailed Answer:**
You are not being "too hard" on the *safety* requirements, but the benchmark is extremely strict on the *process* requirements.

1.  **The "Format Wall"**: The benchmark acts like a strict professor who gives a failing grade if you write the right answer in the wrong format. The model likely knows the answers but is getting hammered (-10 penalty) for not outputting perfect JSON.
    *   *Is this fair?* For an automated safety system, **yes**. If a safety guardrail can't parse the model's output, it can't verify it. The model *must* be machine-readable to be safe.

2.  **Process vs. Outcome**: This is a "Process-Supervised" benchmark. It doesn't just care *what* the model said; it cares *how* it got there (the `proof` and `analysis`).
    *   The model failed to "show its work." In a safety context, an answer without a verifiable trace is considered unsafe by definition.

**Verdict**: The benchmark is working as designed. It successfully identified that this model—while powerful—is "wild" and uncontained. It cannot yet be trusted to follow the strict rules required for an autonomous medical agent.
