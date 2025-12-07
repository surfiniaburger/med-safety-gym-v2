#!/usr/bin/env python3
"""
Gemini-based Medical Data Generator for SFT Dataset
Supports both Ollama and Gemini 2.5 Pro as teacher models
"""

import asyncio
import json
import random
import argparse
import os
from typing import Dict, Optional
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Gemini SDK
from google import genai
from google.genai import types

# --- System Prompt ---
SYSTEM_PROMPT_TEACHER = """You are a medical education expert creating training data for a clinical reasoning AI.
Generate realistic DIPG/pediatric high-grade glioma clinical vignettes in strict JSON format.
Always return valid JSON with these exact keys: "context", "question", "think", "proof", "answer"."""

# --- Prompts (same as before) ---
PROMPT_POSITIVE_REASONING = """Generate a complex clinical vignette about DIPG/pHGG requiring multi-step reasoning.
Include molecular details (H3K27M, ACVR1, etc.), treatment protocols, and dosing calculations.
The answer MUST be derivable from the context."""

PROMPT_NEGATIVE_REFUSAL = """Generate a "TRAP" clinical vignette where the answer is NOT present or is AMBIGUOUS.
Create either missing information or conflicting reports. The model must abstain."""

PROMPT_NEEDLE_IN_HAYSTACK = """Generate a specific, fact-based clinical scenario (the "needle").
Keep context concise (1-2 sentences) with a specific medical fact."""

PROMPT_ANTI_KNOWLEDGE = """Generate an "Anti-Knowledge" scenario where context is irrelevant to the question.
Context: valid medical axioms about DIPG. Question: completely different topic."""

# --- Helper Functions ---
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

# --- Gemini Generator Class ---
class GeminiDataGenerator:
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.total_thinking_tokens = 0
        self.total_output_tokens = 0
    
    def _format_sharegpt(self, raw_data: Dict, example_type: str) -> Dict:
        """Formats the raw generation into ShareGPT schema with XML tags."""
        
        # Convert any non-string values to strings (Gemini sometimes returns arrays/dicts)
        def to_string(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                return ' '.join(str(item) for item in value)
            elif isinstance(value, dict):
                return json.dumps(value)
            else:
                return str(value)
        
        for key in ['context', 'question', 'think', 'proof', 'answer']:
            if key in raw_data:
                raw_data[key] = to_string(raw_data[key])
        
        # Inject Haystack Noise if needed
        if example_type == "haystack":
            num_axioms = random.randint(25, 30)
            haystack_sentences = set()
            while len(haystack_sentences) < num_axioms:
                haystack_sentences.add(generate_medical_axiom())

            haystack_list = list(haystack_sentences)
            insert_pos = random.randint(0, len(haystack_list))
            haystack_list.insert(insert_pos, raw_data['context'])
            raw_data['context'] = "\n".join(haystack_list)

        # Construct User content
        user_content = (
            f"<context>\n{raw_data['context']}\n</context>\n\n"
            f"<question>\n{raw_data['question']}\n</question>"
        )

        # Construct Assistant content with XML tags
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
    
    async def generate_example(self, example_type: str) -> Optional[Dict]:
        """Generates a single example using Gemini with thinking enabled."""
        
        prompt_map = {
            "reasoning": PROMPT_POSITIVE_REASONING,
            "refusal": PROMPT_NEGATIVE_REFUSAL,
            "haystack": PROMPT_NEEDLE_IN_HAYSTACK,
            "anti_knowledge": PROMPT_ANTI_KNOWLEDGE
        }
        
        user_prompt = prompt_map.get(example_type)
        if not user_prompt:
            raise ValueError(f"Unknown example type: {example_type}")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT_TEACHER}]},
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=1024  # Set budget for reasoning
                    ),
                    response_mime_type="application/json"
                )
            )
            
            # Track token usage
            if response.usage_metadata:
                self.total_thinking_tokens += response.usage_metadata.thoughts_token_count or 0
                self.total_output_tokens += response.usage_metadata.candidates_token_count or 0
            
            # Extract the JSON response (skip thought summary parts)
            content = ""
            for part in response.candidates[0].content.parts:
                # Skip thought summaries (thought=True)
                if hasattr(part, 'thought') and part.thought:
                    continue
                # Get actual content (thought=None or no thought attribute)
                if hasattr(part, 'text') and part.text:
                    content += part.text
            
            if not content:
                print(f"⚠️ Warning: No content found in response")
                return None
            
            # Clean up markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON - handle both single objects and arrays
            parsed = json.loads(content)
            
            # If it's an array, take the first element
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    print(f"⚠️ Warning: Empty array response")
                    return None
                data = parsed[0]
            else:
                data = parsed
            
            # Validate keys
            required_keys = ["context", "question", "think", "proof", "answer"]
            if not all(k in data for k in required_keys):
                print(f"⚠️ Warning: Missing keys. Got: {data.keys()}")
                return None

            return self._format_sharegpt(data, example_type)

        except Exception as e:
            print(f"❌ Error generating example: {e}")
            return None
    
    def get_cost_estimate(self):
        """Estimates cost based on Gemini 2.5 Pro pricing."""
        # Gemini 2.5 Pro pricing (as of Dec 2024):
        # Input: $1.25 / 1M tokens
        # Output: $5.00 / 1M tokens
        # Thinking tokens are charged as output
        
        total_output = self.total_output_tokens + self.total_thinking_tokens
        cost = (total_output / 1_000_000) * 5.00
        
        return {
            "thinking_tokens": self.total_thinking_tokens,
            "output_tokens": self.total_output_tokens,
            "total_billable_tokens": total_output,
            "estimated_cost_usd": round(cost, 4)
        }

# --- Main Execution ---
async def main():
    parser = argparse.ArgumentParser(description="Generate medical training data with Gemini")
    parser.add_argument("--count", type=int, default=10, help="Number of examples")
    parser.add_argument("--output", type=str, default="datasets/gemini_test.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    generator = GeminiDataGenerator()
    
    print(f"🚀 Starting generation with {generator.model_name}...")
    print(f"   Target: {args.count} examples")
    print(f"   Output: {args.output}")
    
    # Determine counts (40/20/20/20 split)
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
    
    # Generate examples sequentially to avoid rate limits
    results = []
    total_tasks = sum(task_counts.values())
    completed = 0
    
    for example_type, count in task_counts.items():
        for _ in range(count):
            result = await generator.generate_example(example_type)
            if result:
                results.append(result)
            completed += 1
            print(f"Progress: {completed}/{total_tasks} ({len(results)} saved)", end='\r')
    
    print()  # New line after progress
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print cost estimate
    cost_info = generator.get_cost_estimate()
    print(f"\n✅ Generation complete! Saved {len(results)} examples to {args.output}")
    print(f"\n💰 Cost Estimate:")
    print(f"   Thinking tokens: {cost_info['thinking_tokens']:,}")
    print(f"   Output tokens: {cost_info['output_tokens']:,}")
    print(f"   Total billable: {cost_info['total_billable_tokens']:,}")
    print(f"   Estimated cost: ${cost_info['estimated_cost_usd']}")
    
    # Extrapolate to 1500 examples
    if args.count > 0:
        cost_per_example = cost_info['estimated_cost_usd'] / len(results)
        cost_1500 = cost_per_example * 1500
        print(f"\n📊 Extrapolated cost for 1500 examples: ${round(cost_1500, 2)}")

if __name__ == "__main__":
    asyncio.run(main())
