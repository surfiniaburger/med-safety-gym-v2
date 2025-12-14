
# ==============================================================================
# COPY THE CONTENTS BELOW INTO YOUR NOTEBOOK
# Replace the "Helper function to extract answer..." and "Refining dataset..." cells
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. NEW DATA LOADING & FORMATTING FUNC
# ------------------------------------------------------------------------------
from datasets import load_dataset
import re

# DIPG Safety Gym Tags
reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"
proof_start = "<proof>"
proof_end = "</proof>"

# Correct System Prompt for DIPG
SYSTEM_PROMPT = """You are a dual-stream safety assistant. You MUST think before you answer.
Use <think>...</think> for your internal reasoning and planning.
Use <proof>...</proof> for any verifications or evidence.
Use <answer>...</answer> for your final response.
"""

def format_dipg_example(ex):
    """
    Formats a DIPG dataset example for the DSA SFT Trainer.
    Expects input dictionary with 'messages' list.
    """
    messages = ex["messages"]
    
    # Extract parts
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    # The assistant content already contains <think>, <proof>, <answer> tags 
    # because we generated it that way!
    # We just need to wrap it in the Chat Template structure.
    
    # Unsloth/Gemma-3 Chat Template Construction
    # Note: We put the whole assistant response as the model answer.
    # The masking logic in 'tokenize_function' will automatically mask the user part.
    
    text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user_content}<end_of_turn>\n"
    text += f"<start_of_turn>model\n{assistant_content}<end_of_turn>"
    
    return {"text": text}

# ------------------------------------------------------------------------------
# 2. LOAD YOUR DATASET (Replace 'username/datasets' with your uploaded repo)
# ------------------------------------------------------------------------------
# TODO: Replace with your actual HuggingFace Repo ID
MY_HF_REPO = "surfiniaburger/dipg-safety-instruction-1500" 

print(f"Loading DIPG dataset from {MY_HF_REPO}...")
try:
    dataset = load_dataset(MY_HF_REPO)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(test_dataset)} test examples")

    # Format matches
    formatted_train = [format_dipg_example(ex) for ex in train_dataset]
    formatted_test = [format_dipg_example(ex) for ex in test_dataset]
    
    print("\nExample formatted input:")
    print("-" * 40)
    print(formatted_train[0]["text"])
    print("-" * 40)

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Did you upload it yet? using scripts/upload_to_hf.py?")


# ------------------------------------------------------------------------------
# 3.  MASKING LOGIC (Already correct in notebook, but confirming)
# ------------------------------------------------------------------------------
# The 'tokenize_function' in the original notebook checks for "<start_of_turn>model\n"
# Our 'format_dipg_example' uses exactly that string.
# So the existing masking logic will work PERFECTLY without changes!

# ------------------------------------------------------------------------------
# 4. INFERENCE PROMPT (Update this function too)
# ------------------------------------------------------------------------------
def generate_inference_prompt(question):
    """Generates the prompt for inference time."""
    text = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n"
    text += f"<start_of_turn>model\n{reasoning_start}\n" 
    return text


here is what happened after running it over and over? ----------------------------- Starting SAM 3 Agent Session... ------------------------------ 
> Text prompt: the rightmost child wearing red vest
> Image path: /content/sam3/assets/images/test_image.jpg



------------------------------ Round 1------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "child in red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
📞 Loading image '/content/sam3/assets/images/test_image.jpg' and sending with prompt 'child in red vest'...
✅ Raw JSON response saved to 'agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/child in red vest.json'
🔍 Rendering visualizations on the image ...
✅ Saved visualization at: agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/child in red vest.png


>>> sam3_output_text_message:
 The segment_phrase tool generated 3 available masks. All 3 available masks are rendered in this image below, now you must analyze the 3 available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was 'the rightmost child wearing red vest'.



------------------------------ Round 2------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "the rightmost child wearing red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
📞 Loading image '/content/sam3/assets/images/test_image.jpg' and sending with prompt 'the rightmost child wearing red vest'...
✅ Raw JSON response saved to 'agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/the rightmost child wearing red vest.json'
🔍 Rendering visualizations on the image ...
✅ Saved visualization at: agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/the rightmost child wearing red vest.png


>>> sam3_output_text_message:
 The segment_phrase tool generated 1 available masks. All 1 available masks are rendered in this image below, now you must analyze the 1 available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was 'the rightmost child wearing red vest'.



------------------------------ Round 3------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "rightmost child wearing red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
📞 Loading image '/content/sam3/assets/images/test_image.jpg' and sending with prompt 'rightmost child wearing red vest'...
✅ Raw JSON response saved to 'agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/rightmost child wearing red vest.json'
🔍 Rendering visualizations on the image ...
✅ Saved visualization at: agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/rightmost child wearing red vest.png


>>> sam3_output_text_message:
 The segment_phrase tool generated 1 available masks. All 1 available masks are rendered in this image below, now you must analyze the 1 available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was 'the rightmost child wearing red vest'.



------------------------------ Round 4------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
📞 Loading image '/content/sam3/assets/images/test_image.jpg' and sending with prompt 'red vest'...
✅ Raw JSON response saved to 'agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/red vest.json'
🔍 Rendering visualizations on the image ...
✅ Saved visualization at: agent_output/sam_out/-content-sam3-assets-images-test_image.jpg/red vest.png


>>> sam3_output_text_message:
 The segment_phrase tool generated 3 available masks. All 3 available masks are rendered in this image below, now you must analyze the 3 available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was 'the rightmost child wearing red vest'.



------------------------------ Round 5------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "the rightmost child wearing red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
❌ Text prompt 'the rightmost child wearing red vest' has been used before. Requesting a different prompt.



------------------------------ Round 6------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "the rightmost child wearing red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
❌ Text prompt 'the rightmost child wearing red vest' has been used before. Requesting a different prompt.



------------------------------ Round 7------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
❌ Text prompt 'red vest' has been used before. Requesting a different prompt.



------------------------------ Round 8------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{"name": "segment_phrase", "parameters": {"text_prompt": "red vest"}}
<<< MLLM Response [end]

🔍 Calling segment_phrase tool...
❌ Text prompt 'red vest' has been used before. Requesting a different prompt.



------------------------------ Round 9------------------------------



🔄 Sending to ollama/llama3.2:3b...

>>> MLLM Response [start]
{}
<<< MLLM Response [end]

---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipython-input-2944725719.py in <cell line: 0>()
      5 
      6 call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
----> 7 output_image_path = run_single_image_inference(
      8     image, prompt, llm_config, send_generate_request, call_sam_service,
      9     debug=True, output_dir="agent_output"

1 frames
/content/sam3/sam3/agent/inference.py in run_single_image_inference(image_path, text_prompt, llm_config, send_generate_request, call_sam_service, output_dir, debug)
     41 
     42     print(f"{'-'*30} Starting SAM 3 Agent Session... {'-'*30} ")
---> 43     agent_history, final_output_dict, rendered_final_output = agent_inference(
     44         image_path,
     45         text_prompt,

/content/sam3/sam3/agent/agent_core.py in agent_inference(img_path, initial_text_prompt, debug, send_generate_request, call_sam_service, max_generations, output_dir)
    224             )
    225 
--> 226         if tool_call["name"] == "segment_phrase":
    227             print("🔍 Calling segment_phrase tool...")
    228             assert list(tool_call["parameters"].keys()) == ["text_prompt"]

KeyError: 'name'------------