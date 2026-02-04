import pytest
from med_safety_eval.rubric import Rubric, Sequential
from med_safety_eval.observer import RubricObserver, DataSink
from typing import Dict, Any

class MockSink(DataSink):
    def __init__(self):
        self.snapshots = []
    def emit(self, snapshot: Dict[str, Any]):
        self.snapshots.append(snapshot)

class Leaf(Rubric):
    def __init__(self, val):
        super().__init__()
        self.val = val
    def forward(self, action, obs):
        return self.val

def test_rubric_observer_captures_hierarchy():
    """Test 2.1 & 2.2: Observer captures full hierarchy scores."""
    r1 = Leaf(1.0)
    r2 = Leaf(0.5)
    root = Sequential(r1, r2)
    
    sink = MockSink()
    observer = RubricObserver(root, [sink])
    
    # Execute
    root("act", "obs")
    
    assert len(sink.snapshots) == 1
    snapshot = sink.snapshots[0]
    
    # Check scores
    # Sequential names children as step_0, step_1
    assert snapshot["scores"]["root"] == 0.5 # Sequential returns last score
    assert snapshot["scores"]["step_0"] == 1.0
    assert snapshot["scores"]["step_1"] == 0.5

def test_observer_multiple_emits():
    """Verify observer emits on every root execution."""
    root = Leaf(1.0)
    sink = MockSink()
    observer = RubricObserver(root, [sink])
    
    root("a", "o")
    root("a", "o")
    
    assert len(sink.snapshots) == 2
