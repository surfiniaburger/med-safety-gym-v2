import time
import threading
import pytest
from unittest.mock import MagicMock
from med_safety_eval.observer import RubricObserver
from med_safety_eval.rubric import Rubric

class MockRubric(Rubric):
    def forward(self, action, observation):
        return 10.0

def test_observer_partial_stop_blocking():
    """Verify that RubricObserver correctly pauses the execution loop on specified indices."""
    root = MockRubric()
    sink = MagicMock()
    # Trigger pause on step 0
    observer = RubricObserver(root, sinks=[sink], pause_on_indices=[0])
    
    # We run the rubric call in a separate thread so we can check if it blocks
    def run_eval():
        root(action="test", observation="test")
        
    eval_thread = threading.Thread(target=run_eval)
    eval_thread.start()
    
    # Give it a moment to reach the pause
    time.sleep(0.1)
    
    assert observer.is_paused is True, "Observer should be in paused state"
    assert eval_thread.is_alive() is True, "Thread should be blocked waiting for resume"
    
    # Resume the observer
    observer.resume()
    
    # Wait for completion
    eval_thread.join(timeout=1.0)
    
    assert observer.is_paused is False
    assert not eval_thread.is_alive(), "Thread should have finished after resume"
    assert sink.emit.called, "Sink should have received the snapshot"

def test_remote_command_resume():
    """Verify that manual resume() unblocks the loop (simulating remote command)."""
    root = MockRubric()
    observer = RubricObserver(root, sinks=[], pause_on_indices=[0])
    
    observer.is_paused = True
    observer._resume_event.clear()
    
    # Simulate the polling thread calling resume()
    observer.resume()
    
    assert observer._resume_event.is_set(), "Resume event should be set"
