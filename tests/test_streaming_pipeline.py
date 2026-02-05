
import pytest
from med_safety_eval.manager import LocalEvaluationManager
from med_safety_eval.models import RewardConfig, EvaluationItem, GroundTruth
from med_safety_eval.schemas import NeuralSnapshot
from med_safety_eval.observer import DataSink
from typing import List

class MockSink:
    def __init__(self):
        self.snapshots: List[NeuralSnapshot] = []

    def emit(self, snapshot: NeuralSnapshot) -> None:
        self.snapshots.append(snapshot)

@pytest.fixture
def reward_config():
    return RewardConfig(
        hallucinated_trace_penalty=-10.0,
        correct_synthesis_reward=10.0
    )

@pytest.fixture
def eval_item():
    return EvaluationItem(
        response="<think>Plan</think><proof>evidence</proof><answer>result</answer>",
        ground_truth=GroundTruth(
            context="context containing evidence",
            question="q",
            expected_answer={"final": "result", "proof": "evidence"}
        )
    )

def test_manager_no_streaming(reward_config, eval_item):
    """Verify standard behavior without sinks."""
    manager = LocalEvaluationManager(reward_config)
    
    # Internal observer should be None
    assert getattr(manager, "_observer", None) is None
    
    result = manager.evaluate_batch([eval_item])
    assert result.total_responses == 1

def test_manager_with_streaming(reward_config, eval_item):
    """Verify streaming behavior with sinks."""
    sink = MockSink()
    manager = LocalEvaluationManager(reward_config, sinks=[sink], session_id="test_stream")
    
    # Internal observer should exist
    assert hasattr(manager, "_observer")
    
    result = manager.evaluate_batch([eval_item])
    
    # Verify emission
    assert len(sink.snapshots) == 1
    snapshot = sink.snapshots[0]
    
    assert snapshot.session_id == "test_stream"
    assert snapshot.step == 1
    assert "root" in snapshot.scores
    # Verify score is float
    assert isinstance(snapshot.scores["root"], float)

def test_streaming_multiple_items(reward_config, eval_item):
    """Verify streaming increments step count correctly."""
    sink = MockSink()
    manager = LocalEvaluationManager(reward_config, sinks=[sink], session_id="multi_step")
    
    items = [eval_item, eval_item, eval_item]
    manager.evaluate_batch(items)
    
    assert len(sink.snapshots) == 3
    assert sink.snapshots[0].step == 1
    assert sink.snapshots[1].step == 2
    assert sink.snapshots[2].step == 3
