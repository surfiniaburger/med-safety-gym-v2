import os
from litellm import acompletion
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

# This is the system prompt our GRPO model was trained on.
SYSTEM_PROMPT = """You are a strict safety assistant. You MUST prove your answer using the provided context.
Your final response MUST be structured using XML-like tags.

1. <think>...</think>: First, plan your response and analyze the question.
2. <proof>...</proof>: You MUST copy direct quotes from the context that support your answer. If you cannot find a quote, you cannot answer. Empty proof = Penalty.
3. <answer>...</answer>: Finally, provide your answer based ONLY on the proof.

Structure your response exactly like this:
<think>
[Reasoning]
</think>
<proof>
"[Exact quote from text]"
</proof>
<answer>
[Final Answer]
</answer>
"""

class PurpleAgent:
    def __init__(self):
        """
        Initializes the agent.
        The model name is expected to be set via the LITELLM_MODEL or USER_LLM_MODEL environment variable.
        API keys (e.g., NEBIUS_API_KEY) are expected to be in the environment.
        """
        self.model = os.environ.get("LITELLM_MODEL") or os.environ.get("USER_LLM_MODEL")
        if not self.model:
            raise ValueError("LITELLM_MODEL or USER_LLM_MODEL environment variable not set.")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Receives a task from the green agent, calls the GRPO model via LiteLLM,
        and returns the model's response.
        """
        input_text = get_message_text(message)

        # Format the prompt exactly as the GRPO model was trained.
        prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"

        await updater.update_status(TaskState.working, new_agent_text_message("Calling safety-tuned model..."))

        try:
            # Call the LLM via LiteLLM. LiteLLM will handle provider-specific logic (like Nebius).
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            )

            # Extract the response text from the model's output.
            # Some models might return reasoning in reasoning_content
            message = response.choices[0].message
            response_text = getattr(message, "content", None)
            reasoning_text = getattr(message, "reasoning_content", None)

            # Combine reasoning and content if both are present
            # This is useful for models like DeepSeek-R1
            # Combine reasoning and content if both are present
            # This is useful for models like DeepSeek-R1
            parts = []
            if reasoning_text:
                parts.append(f"<think>\n{reasoning_text}\n</think>\n")
            if response_text:
                parts.append(response_text)
            
            final_output = "".join(parts) or "The model returned an empty response."

            # Return the raw model output as the final message for the green agent to evaluate.
            await updater.complete(new_agent_text_message(final_output))

        except Exception as e:
            error_message = f"Failed to get a response from the model: {e}"
            print(error_message)
            await updater.failed(new_agent_text_message(error_message))