
import pytest
from med_safety_gym.dipg_environment import DIPGEnvironment
from med_safety_gym.models import DIPGAction
from typing import List
from med_safety_eval.schemas import NeuralSnapshot

class InMemorySink:
    def __init__(self):
        self.snapshots: List[NeuralSnapshot] = []

    def emit(self, snapshot: NeuralSnapshot) -> None:
        self.snapshots.append(snapshot)

@pytest.fixture
def mock_dataset(tmp_path):
    data_file = tmp_path / "train_loop.jsonl"
    import json
    with open(data_file, "w") as f:
        # Create enough data for 5 steps
        for i in range(10):
            item = {
                "messages": [
                    {"role": "user", "content": f"**CONTEXT:**\nC{i}\n\n**REQUEST:**\nQ{i}\n\n**REASONING STEPS:**\n..."},
                    {"role": "assistant", "content": f"A{i}"}
                ]
            }
            f.write(json.dumps(item) + "\n")
    return str(data_file)

def test_training_loop_recording(mock_dataset):
    """
    Simulates a training loop (Phase 5 goal) and asserts that
    the entire 'film' of the training is recorded by the sink.
    """
    sink = InMemorySink()
    session_id = "training_run_v4_experiment"
    
    env = DIPGEnvironment(
        dataset_path=mock_dataset,
        conflict_reward=0, abstain_reward=0, hallucination_penalty=0, missing_answer_penalty=0,
        hallucinated_trace_penalty=0, proof_inconsistency_penalty=0, incorrect_answer_penalty=0,
        conflict_penalty=0, abstain_penalty=0, missing_trace_penalty=0,
        correct_abstention_reward=0, verifiable_trace_reward=0, correct_synthesis_reward=0,
        exact_format_reward=0, format_mismatch_penalty=0, no_hallucination_reward=0,
        analysis_channel_start="<analysis>", proof_channel_start="<proof>", final_channel_start="<final>", channel_end="</end>",
        sinks=[sink],
        session_id=session_id
    )
    
    # Simulate 5 training steps
    for step in range(5):
        obs = env.reset()
        # Agent "learns" and acts
        action = DIPGAction(llm_response=f"<analysis>Thinking {step}</end><proof>Evidence {step}</end><final>Answer {step}</end>")
        env.step(action)
        
    # Verification
    assert len(sink.snapshots) == 5
    
    for i, snapshot in enumerate(sink.snapshots):
        assert snapshot.session_id == session_id
        assert snapshot.step == i + 1
        assert "root" in snapshot.scores
        # Verify metadata captured the action
        assert f"Answer {i}" in str(snapshot.metadata)

    print(f"\n✅ Successfully recorded {len(sink.snapshots)} frame training film.")
