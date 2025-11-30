# ==================================================================================
# Prepare the Dataset for GRPO with JSON Format
# ==================================================================================
print("--- Preparing dataset for GRPOTrainer using JSON format ---")

import json

def create_grpo_prompt_json(example):
    """
    Prepare GRPO dataset with JSON format instead of custom tags.
    
    For JSON format, the model should output:
    {
        "analysis": "reasoning about the medical context",
        "proof": "direct quotes from sources",
        "final": "conclusive answer"
    }
    """
    # Get the conversation messages
    messages = example['messages']

    # Manually construct the prompt string from the system and user messages
    prompt_parts = []
    for msg in messages[:-1]:  # Go through all messages EXCEPT the last assistant one
        if msg['role'] == 'system':
            prompt_parts.append(f"System: {msg['content']}")
        elif msg['role'] == 'user':
            prompt_parts.append(f"User: {msg['content']}")

    # Join the parts and add the generation prompt for the assistant
    prompt_text = "\n".join(prompt_parts) + "\nAssistant:"

    # The 'chosen' response should be in JSON format
    # Parse the original custom tags format and convert to JSON
    original_response = messages[-1]['content']
    
    # Try to extract the content from custom tags format
    # This is a simple parser - you might need to adjust based on your actual data
    import re
    
    analysis_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', original_response, re.DOTALL)
    proof_match = re.search(r'<\|channel\|>proof<\|message\|>(.*?)<\|end\|>', original_response, re.DOTALL)
    final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)<\|end\|>', original_response, re.DOTALL)
    
    # Build JSON response
    chosen_json = {
        "analysis": analysis_match.group(1).strip() if analysis_match else "",
        "proof": proof_match.group(1).strip() if proof_match else "",
        "final": final_match.group(1).strip() if final_match else ""
    }
    
    chosen_response = json.dumps(chosen_json, indent=2)

    # The 'rejected' response - a less helpful JSON response
    # In a real scenario, this would be a less-preferred output
    rejected_json = {
        "analysis": "This is a simple, less detailed analysis.",
        "proof": "",  # Missing evidence
        "final": "This is a rejected, less helpful answer."
    }
    
    rejected_response = json.dumps(rejected_json, indent=2)

    return {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

# Apply the transformation to your dataset
# Assuming 'dataset' is already loaded
# grpo_dataset = dataset.map(create_grpo_prompt_json, remove_columns=list(dataset['train'].features))

print("GRPO dataset preparation function created for JSON format.")
print("\n--- Example Usage ---")
print("grpo_dataset = dataset.map(create_grpo_prompt_json, remove_columns=list(dataset['train'].features))")

# Example of what the output looks like
example_chosen = {
    "analysis": "The provided texts indicate that ONC201 shows potential in preclinical models.",
    "proof": "Source A states: 'ONC201 has demonstrated radiographic improvement in H3 K27M-mutant glioma.'",
    "final": "ONC201 shows promise but requires further clinical validation."
}

example_rejected = {
    "analysis": "This is a simple, less detailed analysis.",
    "proof": "",
    "final": "This is a rejected, less helpful answer."
}

print("\n--- Sample Chosen Response (JSON) ---")
print(json.dumps(example_chosen, indent=2))

print("\n--- Sample Rejected Response (JSON) ---")
print(json.dumps(example_rejected, indent=2))
