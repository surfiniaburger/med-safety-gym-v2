import pytest
from typing import Dict, Any, List
from med_safety_eval.rubric import Rubric, Sequential
from med_safety_eval.observer import RubricObserver, DataSink

class MockSink:
    def __init__(self):
        self.snapshots = []

    def emit(self, snapshot: Dict[str, Any]) -> None:
        self.snapshots.append(snapshot)

class MockLeafRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0

def test_observer_snapshot_structure():
    """Test 2.1 & 2.2: Observer captures hierarchical snapshot."""
    # Setup Rubric Tree
    root = Sequential(MockLeafRubric())
    root.step_0 = MockLeafRubric() # Explicitly naming for clarity in test
    
    # Setup Observer
    sink = MockSink()
    observer = RubricObserver(root, [sink])
    
    # Execute
    root("act", "obs")
    
    # Verify
    assert len(sink.snapshots) == 1
    snapshot = sink.snapshots[0]
    
    assert snapshot.timestamp is not None
    assert snapshot.scores is not None
    scores = snapshot.scores
    
    # Check keys exist (root and children)
    assert "root" in scores
    assert "step_0" in scores
    
    # Check values
    assert scores["root"] == 1.0
    assert scores["step_0"] == 1.0

def test_observer_multiple_sinks():
    """Test 2.3: Observer broadcasts to multiple sinks."""
    root = MockLeafRubric()
    sink1 = MockSink()
    sink2 = MockSink()
    
    observer = RubricObserver(root, [sink1, sink2])
    root("act", "obs")
    
    assert len(sink1.snapshots) == 1
    assert len(sink2.snapshots) == 1
