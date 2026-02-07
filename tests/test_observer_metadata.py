"""
Test to verify metadata propagation in RubricObserver for Evolution Mode.
"""
from med_safety_eval.observer import RubricObserver, DatabaseSink, NeuralSnapshot
from med_safety_eval.rubric import Rubric

class MockRubric(Rubric):
    """Simple mock rubric for testing."""
    def __init__(self):
        super().__init__()
        self.last_score = 0.0
    
    def forward(self, action, observation):
        self.last_score = 42.0
        return self.last_score

class MockSink:
    """Mock sink to capture snapshots."""
    def __init__(self):
        self.snapshots = []
    
    def emit(self, snapshot: NeuralSnapshot):
        self.snapshots.append(snapshot)

def test_metadata_propagation():
    """Test that base_metadata is correctly merged into snapshots."""
    # Create mock rubric and sink
    rubric = MockRubric()
    mock_sink = MockSink()
    
    # Define base metadata (as used in GRPO training)
    base_metadata = {
        "run_type": "grpo",
        "task_id": "dipg_safety_v1",
        "model": "gemma-3-2b",
        "timestamp": 1707340800
    }
    
    # Create observer with metadata
    observer = RubricObserver(
        root_rubric=rubric,
        sinks=[mock_sink],
        session_id="test_session",
        base_metadata=base_metadata
    )
    
    # Trigger a score event (this would normally happen via hooks)
    rubric.last_score = 50.0
    snapshot = observer.capture_snapshot(action="test_action", observation="test_obs")
    
    # Verify snapshot has merged metadata
    assert snapshot.metadata["run_type"] == "grpo"
    assert snapshot.metadata["task_id"] == "dipg_safety_v1"
    assert snapshot.metadata["model"] == "gemma-3-2b"
    assert snapshot.metadata["timestamp"] == 1707340800
    assert snapshot.metadata["action"] == "test_action"
    assert snapshot.metadata["observation"] == "test_obs"
    
    print("✓ Metadata propagation test passed!")
    print(f"  Snapshot metadata: {snapshot.metadata}")

def test_observer_without_metadata():
    """Test observer works without base_metadata (backwards compat)."""
    rubric = MockRubric()
    mock_sink = MockSink()
    
    # Create observer WITHOUT metadata
    observer = RubricObserver(
        root_rubric=rubric,
        sinks=[mock_sink],
        session_id="test_session"
    )
    
    rubric.last_score = 30.0
    snapshot = observer.capture_snapshot(action="test", observation="test")
    
    # Should still work, just without the base metadata fields
    assert "action" in snapshot.metadata
    assert "observation" in snapshot.metadata
    assert "run_type" not in snapshot.metadata  # Not provided
    
    print("✓ Backwards compatibility test passed!")

if __name__ == "__main__":
    test_metadata_propagation()
    test_observer_without_metadata()
    print("\n✅ All metadata tests passed!")
