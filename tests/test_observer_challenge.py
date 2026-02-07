import pytest
import threading
import time
from typing import Iterator, Tuple
from med_safety_eval.observer import RubricObserver
from med_safety_eval.rubric import Rubric
from med_safety_eval.schemas import NeuralSnapshot

class MockSink:
    def __init__(self):
        self.snapshots = []
    def emit(self, snapshot: NeuralSnapshot):
        self.snapshots.append(snapshot)

def test_observer_emits_challenge_on_pause():
    # Simple rubric that returns failure
    class FailRubric(Rubric):
        def forward(self, action, observation):
            # This is called by __call__
            return -1.0
        
        def update_config(self, config): pass
        
        def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, Rubric]]:
            # Mock grounding and root to trigger the hallucination guard
            yield "grounding", self
            yield "", self

    fail_rubric = FailRubric()
    sink = MockSink()
    # Setup observer with a pause on index 0
    observer = RubricObserver(root_rubric=fail_rubric, sinks=[sink], session_id="test_challenge", pause_on_indices=[0])
    
    # We must run this in a thread because it blocks on pause
    def run_eval():
        fail_rubric("action", "obs")
        
    t = threading.Thread(target=run_eval)
    t.start()
    
    # Wait for pause
    time.sleep(0.5)
    
    try:
        assert observer.is_paused is True
        assert len(sink.snapshots) > 0
        last_snapshot = sink.snapshots[-1]
        
        # This will fail because capture_snapshot doesn't populate challenge yet
        assert last_snapshot.is_paused is True
        assert last_snapshot.challenge is not None
        assert last_snapshot.challenge["type"] == "HALLUCINATION_GUARD"
    finally:
        # Resume to finish test
        observer.resume()
        t.join()
