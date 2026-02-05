
import pytest
from unittest.mock import MagicMock
from med_safety_gym.dipg_environment import DIPGEnvironment
from med_safety_gym.models import DIPGAction
# Import Config and Rubric from med_safety_eval to ensure correct types are mocked/used if needed
from med_safety_eval.models import RewardConfig

class MockSink:
    def __init__(self):
        self.snapshots = []
    def emit(self, snapshot):
        self.snapshots.append(snapshot)

@pytest.fixture
def mock_dataset(tmp_path):
    # Create a dummy dataset file
    data_file = tmp_path / "data.jsonl"
    import json
    with open(data_file, "w") as f:
        item = {
            "messages": [
                {"role": "user", "content": "**CONTEXT:**\nCtx\n\n**REQUEST:**\nQ\n\n**REASONING STEPS:**\n..."},
                {"role": "assistant", "content": "Ans"}
            ]
        }
        f.write(json.dumps(item) + "\n")
    return str(data_file)

def test_environment_emits_to_sink(mock_dataset):
    """Verify DIPGEnvironment emits to sink during step()."""
    sink = MockSink()
    
    env = DIPGEnvironment(
        dataset_path=mock_dataset,
        conflict_reward=0, abstain_reward=0, hallucination_penalty=0, missing_answer_penalty=0,
        hallucinated_trace_penalty=0, proof_inconsistency_penalty=0, incorrect_answer_penalty=0,
        conflict_penalty=0, abstain_penalty=0, missing_trace_penalty=0,
        correct_abstention_reward=0, verifiable_trace_reward=0, correct_synthesis_reward=0,
        exact_format_reward=0, format_mismatch_penalty=0, no_hallucination_reward=0,
        analysis_channel_start="<analysis>", proof_channel_start="<proof>", final_channel_start="<final>", channel_end="</end>",
        sinks=[sink],
        session_id="env_test"
    )
    
    env.reset()
    
    # Take a step
    action = DIPGAction(llm_response="<analysis>A</end><proof>P</end><final>F</end>")
    env.step(action)
    
    # Check sink
    assert len(sink.snapshots) == 1
    assert sink.snapshots[0].session_id == "env_test"
    # Step count should increment
    assert sink.snapshots[0].step == 1
