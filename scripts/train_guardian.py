"""RL training script for SafeClaw Guardian (Safety Reasoning) expert."""
import asyncio
import json
from openai import AsyncOpenAI
import art
from art.local import LocalBackend
from med_safety_gym.mcp_server import check_entity_parity

# --- Scenarios ---
def load_scenarios():
    with open("data/guardian_scenarios.jsonl") as f:
        return [json.loads(line) for line in f]

scenarios = load_scenarios()

# --- Rollout function ---
async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # We use the previous trajectory as context
    messages = scenario["trajectory"]
    intent = scenario["intent"]
    context_entities = scenario["entities"]
    
    # Reconstruct the prompt
    # In a real rollout, we'd want the model to generate the NEXT turn
    # but for this expert training, we want the model to generate the response 
    # to the LAST user message in the trajectory.
    
    # Filter for the actual prompt
    user_prompts = [m for m in messages if m["role"] == "user"]
    last_prompt = user_prompts[-1]["content"] if user_prompts else "hi"
    
    # Provide the context to the model
    # Note: We simulate the prompt as seen by SafeClawAgent
    system_prompt = (
        "You are SafeClaw, a strict but helpful medical AI assistant. "
        "Use the following verified context to answer the user.\n"
        "If the user asks a question not covered by the context, you MUST state that you do not know.\n\n"
        "Context: [Simulated Medical Knowledge + Previous Entities]\n"
    )
    
    model_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_prompt}
    ]

    response = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=model_msgs,
        temperature=0.7,
        max_tokens=256,
    )
    choice = response.choices[0]
    generated_text = choice.message.content

    # --- Compute reward ---
    # We use our own check_entity_parity logic
    # The 'context' for the safety gate is the set of verified entities
    context_str = ", ".join(context_entities)
    is_safe, _reason = await check_entity_parity(generated_text, context_str)
    
    # Reward structure:
    # 1.0 if safe
    # -1.0 if hallucination
    reward = 1.0 if is_safe else -1.0

    return art.Trajectory(
        messages_and_choices=[*model_msgs, choice],
        reward=reward,
        metrics={"safe": 1.0 if is_safe else 0.0},
    )

# --- Training loop ---
async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="guardian-001",
        project="safeclaw-experts",
        base_model="OpenPipe/Qwen2.5-7B-Instruct",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.5},
        ),
    )
    await model.register(backend)

    NUM_STEPS = 20 # Short run for demonstration
    ROLLOUTS_PER_GROUP = 8 # GRPO needs multiple rollouts to compute relative advantage
    GROUPS_PER_STEP = 1

    print(f"Starting Guardian RL training for {NUM_STEPS} steps...")

    for step in range(await model.get_step(), NUM_STEPS):
        # Sample scenarios
        batch_scenarios = [scenarios[step % len(scenarios)]]
        
        groups = [
            art.TrajectoryGroup(
                rollout(model, scenario)
                for _ in range(ROLLOUTS_PER_GROUP)
            )
            for scenario in batch_scenarios
        ]
        
        finished_groups = await art.gather_trajectory_groups(
            groups, pbar_desc=f"step {step}"
        )

        avg_reward = sum(
            t.reward for g in finished_groups for t in g.trajectories
        ) / max(1, sum(len(g.trajectories) for g in finished_groups))
        
        avg_safe = sum(
            t.metrics["safe"] for g in finished_groups for t in g.trajectories
        ) / max(1, sum(len(g.trajectories) for g in finished_groups))
        
        print(f"Step {step}: avg_reward={avg_reward:.3f}, safety_rate={avg_safe:.1%}")

        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )

    # --- Training complete ---
    step = await model.get_step()
    inference_name = model.get_inference_name()
    client = model.openai_client()

    print("\n" + "=" * 60)
    print("GUARDIAN RL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {inference_name}")
    print(f"  Final Safety:   {avg_safe:.1%}")
    print("=" * 60)

    await backend.close()

if __name__ == "__main__":
    asyncio.run(main())
