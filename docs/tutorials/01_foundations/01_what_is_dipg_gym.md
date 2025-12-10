# What is the DIPG Safety Gym?

## The Problem: High-Stakes Medical AI
Imagine an AI assistant helping a doctor treat a rare, fatal pediatric brain tumor called **Diffuse Intrinsic Pontine Glioma (DIPG)**. 
- If the AI makes a mistake, the cost is catastrophic.
- If the AI "hallucinates" a treatment that doesn't exist, it gives false hope.
- If the AI answers a question it relies on unverified knowledge, it risks patient safety.

Most "Safety Gyms" focus on "Chatbot Safety" (politeness, toxicity). **DIPG Safety Gym focuses on Epistemic Safety**—does the model *know* what it is talking about?

![Safety Concept Diagram](images/safety_concept.png)
*(The 3-Step Safety Protocol)*

## The Solution: "Process Supervision"
We don't just grade the final answer. We grade the **thought process**.

In this gym, an AI agent cannot just say "Take Aspirin". It must follow a strict 3-step protocol for every single response:

### 1. Analysis (`<analysis>`)
* **What it is:** The "scratchpad".
* **What happens:** The model reads the medical context and thinks about the problem. It identifies conflicts or missing information.
* **Analogy:** A doctor reading a patient's chart.

### 2. Proof (`<proof>`)
* **What it is:** The "citations".
* **What happens:** The model must **quote** the exact text from the provided context that supports its decision.
* **Rule:** If the text isn't in the context, it cannot be used.
* **Analogy:** Highlighting the relevant lines in a medical textbook.

### 3. Final Answer (`<final>`)
* **What it is:** The patient-facing response.
* **What happens:** A concise, synthesized answer (or a safe refusal) based *only* on the Proof.
* **Analogy:** The prescription given to the patient.

---

## Why "Gym"?
We call it a "Gym" because it is an environment for **training**. 
- It provides **Observations** (Medical Context + Question).
- It accepts **Actions** (The 3-step response).
- It gives **Rewards** (Scores for safety, accuracy, and grounding).

Your goal in this gym is to build an agent that gets a high score by being **Safe, Grounded, and Honest**.
