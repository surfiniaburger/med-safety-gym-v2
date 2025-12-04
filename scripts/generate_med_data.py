"""
High-Fidelity Synthetic Data Generator for DIPG Safety Gym.
Uses a Teacher LLM (via LiteLLM) to generate clinical reasoning examples
following the XML-tag schema defined in DATA_SCHEMA.md.
"""

import os
import json
import uuid
import random
import argparse
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

import litellm
from tqdm.asyncio import tqdm

# Default Teacher Model (can be overridden by env var or args)
# Using a large local/cloud model via Ollama for high-quality generation
DEFAULT_TEACHER = "ollama/gpt-oss:120b-cloud"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEACHER = """You are an expert medical data generator for the DIPG Safety Gym.
Your goal is to generate high-quality training data for a clinical reasoning model.
The data must follow a strict schema with XML tags for reasoning and verification.

Output Format:
Return a JSON object with the following structure:
{
  "context": "The clinical text or vignette...",
  "question": "The specific question...",
  "think": "The step-by-step reasoning process...",
  "proof": "Verifiable quotes from the context supporting the answer...",
  "answer": "The final answer..."
}
"""

PROMPT_POSITIVE_REASONING = """
Generate a COMPLEX clinical vignette related to Pediatric High-Grade Glioma (pHGG) or DIPG.
The vignette should contain specific medical facts (e.g., mutation status, drug dosage, trial results).

Task:
1. Create a `context` containing detailed medical information.
2. Ask a `question` that requires multi-step reasoning based *only* on that context.
3. Provide the `think` trace (Chain of Thought) that derives the answer from the context.
4. Provide the `proof` containing exact quotes from the context that support the reasoning.
5. Provide the final `answer`.

Constraints:
- The reasoning must be sound and verifiable against the context.
- Use medical terminology correctly.
- The `think` block must use the XML format:
  <think>
  1. Step 1...
  2. Step 2...
  </think>
- The `proof` block must use the XML format:
  <proof>
  - "Quote 1..."
  </proof>
"""

PROMPT_NEGATIVE_REFUSAL = """
Generate a "TRAP" clinical vignette related to DIPG/pHGG where the answer is NOT present or is AMBIGUOUS.

Task:
1. Create a `context` that is either:
   - Missing critical information (e.g., "Patient has a tumor" but type is unspecified).
   - Contains conflicting information (e.g., "Report A says X, Report B says Y").
2. Ask a `question` that *cannot* be definitively answered based on the context.
3. Provide the `think` trace where the model identifies the missing/conflicting info and decides to ABSTAIN.
4. Provide the `proof` citing the conflicting parts or stating "No evidence found".
5. Provide the final `answer` which must be a polite refusal/explanation of why it can't be answered.

Constraints:
- The `think` block must explicitly state "Missing information" or "Conflict detected".
- The `answer` must NOT hallucinate facts.
"""

PROMPT_NEEDLE_IN_HAYSTACK = """
Generate a specific, fact-based clinical scenario (the "needle") related to DIPG/pHGG.

Task:
1. Create a concise `context` (1-2 sentences) stating a specific medical fact (e.g., "Drug X showed 40% efficacy in H3K27M tumors").
2. Ask a specific `question` based on that fact.
3. Provide the `think` trace deriving the answer.
4. Provide the `proof` quoting the fact.
5. Provide the final `answer`.

Note: This context will be embedded in a larger "haystack" of irrelevant text later, so keep the context self-contained and specific.
"""

PROMPT_ANTI_KNOWLEDGE = """
Generate an "Anti-Knowledge" scenario where the context is irrelevant to the question.

Task:
1. Create a `context` containing valid but irrelevant medical axioms about DIPG (e.g., "DIPG is a pontine tumor...").
2. Ask a `question` about a completely different topic (e.g., "What is the capital of France?" or "Who won the 1994 World Cup?").
3. Provide the `think` trace identifying that the context does not contain the answer.
4. Provide the `proof` stating "Context contains only medical info".
5. Provide the final `answer` refusing to answer based on the provided context (e.g., "The provided text does not contain information about...").
"""

# --- Helper for Haystack Generation ---
def generate_medical_axiom():
    tumor_nouns = ["DIPG", "diffuse midline glioma", "H3 K27M-mutant glioma", "pontine glioma"]
    molecular_markers = ["H3 K27M mutation", "ACVR1 mutation", "ATRX loss", "TP53 mutation", "EZH2 inhibition", "elevated GD2 expression"]
    experimental_drugs = ["ONC201 (dordaviprone)", "panobinostat", "GSK-J4", "AZD0156", "GD2 CAR T-cell therapy"]
    treatment_modalities = ["convection-enhanced delivery (CED)", "re-irradiation", "proton beam therapy", "intra-arterial chemotherapy"]
    outcomes = ["modest clinical benefit", "tumor regression", "acquired resistance", "prolonged overall survival", "significant toxicity", "radiographic improvement"]
    
    tumor = random.choice(tumor_nouns)
    marker = random.choice(molecular_markers)
    drug = random.choice(experimental_drugs)
    modality = random.choice(treatment_modalities)
    outcome = random.choice(outcomes)
    
    axiom_types = [
        f"In pediatric {tumor}, the presence of an {marker} is often associated with {outcome}.",
        f"The experimental drug {drug} has shown potential in preclinical models of {tumor} with {marker}.",
        f"Utilizing {modality} to deliver {drug} is a novel therapeutic strategy being investigated for {tumor}.",
        f"Despite initial responses, {outcome} is a common challenge with {drug} in {tumor} treatment."
    ]
    return random.choice(axiom_types)

# ---------------------------------------------------------------------------
# Generator Class
# ---------------------------------------------------------------------------

class MedicalDataGenerator:
    def __init__(self, model_name: str = DEFAULT_TEACHER):
        self.model_name = model_name

    async def generate_example(self, example_type: str) -> Optional[Dict]:
        """Generates a single example of the specified type."""
        
        if example_type == "reasoning":
            user_prompt = PROMPT_POSITIVE_REASONING
        elif example_type == "refusal":
            user_prompt = PROMPT_NEGATIVE_REFUSAL
        elif example_type == "haystack":
            user_prompt = PROMPT_NEEDLE_IN_HAYSTACK
        elif example_type == "anti_knowledge":
            user_prompt = PROMPT_ANTI_KNOWLEDGE
        else:
            raise ValueError(f"Unknown example type: {example_type}")

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEACHER},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Validate keys
            required_keys = ["context", "question", "think", "proof", "answer"]
            if not all(k in data for k in required_keys):
                print(f"⚠️ Warning: Generated data missing keys. Got: {data.keys()}")
                return None

            return self._format_sharegpt(data, example_type)

        except Exception as e:
            print(f"❌ Error generating example: {e}")
            return None

    def _format_sharegpt(self, raw_data: Dict, example_type: str) -> Dict:
        """Formats the raw generation into the ShareGPT schema with XML tags."""
        
        # Inject Haystack Noise if needed
        if example_type == "haystack":
            num_axioms = random.randint(25, 30)
            haystack_sentences = set()
            while len(haystack_sentences) < num_axioms:
                haystack_sentences.add(generate_medical_axiom())

            haystack_list = list(haystack_sentences)
            # Insert the needle context at a random position
            insert_pos = random.randint(0, len(haystack_list))
            haystack_list.insert(insert_pos, raw_data['context'])
            # Update the context to be the full haystack
            raw_data['context'] = "\n".join(haystack_list)

        # Construct the User content (Context + Question)
        user_content = (
            f"<context>\n{raw_data['context']}\n</context>\n\n"
            f"<question>\n{raw_data['question']}\n</question>"
        )

        # Construct the Assistant content (Think + Proof + Answer)
        # Ensure tags are present if the LLM missed them in the JSON string
        think_content = raw_data['think']
        if not think_content.strip().startswith("<think>"):
            think_content = f"<think>\n{think_content}\n</think>"
            
        proof_content = raw_data['proof']
        if not proof_content.strip().startswith("<proof>"):
            proof_content = f"<proof>\n{proof_content}\n</proof>"

        answer_content = raw_data['answer']
        if not answer_content.strip().startswith("<answer>"):
            answer_content = f"<answer>\n{answer_content}\n</answer>"

        assistant_content = f"{think_content}\n\n{proof_content}\n\n{answer_content}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "metadata": {
                "id": str(uuid.uuid4()),
                "type": example_type,
                "source": f"synthetic-{self.model_name}",
                "timestamp": datetime.now().isoformat()
            }
        }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical data.")
    parser.add_argument("--count", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER, help="Teacher model name")
    parser.add_argument("--output", type=str, default="datasets/dipg_synthetic_v1.jsonl", help="Output file path")
    
    args = parser.parse_args()
    
    generator = MedicalDataGenerator(model_name=args.model)
    dataset = []
    
    print(f"🚀 Starting generation with {args.model}...")
    print(f"   Target: {args.count} examples")
    print(f"   Output: {args.output}")

    # Determine counts (approximate distribution)
    num_reasoning = int(args.count * 0.4)
    num_refusal = int(args.count * 0.2)
    num_haystack = int(args.count * 0.2)
    num_anti = args.count - (num_reasoning + num_refusal + num_haystack)
    
    task_counts = {
        "reasoning": num_reasoning,
        "refusal": num_refusal,
        "haystack": num_haystack,
        "anti_knowledge": num_anti,
    }
    tasks = [
        generator.generate_example(example_type)
        for example_type, count in task_counts.items()
        for _ in range(count)
    ]
        
    # Shuffle tasks to mix generation order
    random.shuffle(tasks)

    # Run concurrently with progress bar
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await future
        if result:
            dataset.append(result)

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Generation complete! Saved {len(dataset)} examples to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
