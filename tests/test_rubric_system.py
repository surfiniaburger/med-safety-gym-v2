import pytest
from typing import Any
from med_safety_eval.rubric import Rubric, Sequential, Gate, WeightedSum

class LeafRubric(Rubric):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
    
    def forward(self, action: Any, observation: Any) -> float:
        return self.value

class CompositeRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.child1 = LeafRubric(0.5)
        self.child2 = LeafRubric(0.3)
    
    def forward(self, action: Any, observation: Any) -> float:
        return self.child1(action, observation) + self.child2(action, observation)

def test_rubric_registration():
    """Test that child rubrics are automatically registered."""
    root = CompositeRubric()
    
    # Check children
    children = dict(root.named_children())
    assert "child1" in children
    assert "child2" in children
    assert isinstance(children["child1"], LeafRubric)

def test_named_rubrics():
    """Test that named_rubrics yields all nested rubrics."""
    root = CompositeRubric()
    named = dict(root.named_rubrics())
    
    assert "" in named  # Root
    assert "child1" in named
    assert "child2" in named
    assert named["child1"].value == 0.5

def test_forward_hooks():
    """Test that forward hooks are called with correct arguments."""
    root = LeafRubric(0.8)
    hook_called = False
    hook_data = {}

    def hook(rubric, action, obs, result):
        nonlocal hook_called, hook_data
        hook_called = True
        hook_data = {"rubric": rubric, "result": result}

    root.register_forward_hook(hook)
    
    result = root("action", "obs")
    
    assert result == 0.8
    assert hook_called
    assert hook_data["rubric"] == root
    assert hook_data["result"] == 0.8

def test_last_score():
    """Test that last_score is updated after call."""
    root = LeafRubric(0.7)
    assert root.last_score == 0.0
    root("a", "o")
    assert root.last_score == 0.7

def test_nested_composite():
    """Test deeper hierarchy."""
    class DeepComposite(Rubric):
        def __init__(self):
            super().__init__()
            self.sub = CompositeRubric()
        
        def forward(self, action, obs):
            return self.sub(action, obs)

    root = DeepComposite()
    named = dict(root.named_rubrics())
    
    assert "sub" in named
    assert "sub.child1" in named
    assert "sub.child2" in named
    assert isinstance(named["sub.child1"], LeafRubric)

def test_sequential_container():
    """Test Sequential container (fail-fast)."""
    # Should return 0.0 if any child returns 0.0
    seq = Sequential(
        LeafRubric(1.0),
        LeafRubric(0.0),
        LeafRubric(1.0)
    )
    assert seq("a", "o") == 0.0
    
    # Should return the last value if all are non-zero
    seq2 = Sequential(
        LeafRubric(0.5),
        LeafRubric(0.8)
    )
    assert seq2("a", "o") == 0.8

def test_gate_container():
    """Test Gate container."""
    gate = Gate(LeafRubric(0.8), threshold=0.5)
    assert gate("a", "o") == 0.8
    
    gate_fail = Gate(LeafRubric(0.4), threshold=0.5)
    assert gate_fail("a", "o") == 0.0

def test_weighted_sum_container():
    """Test WeightedSum container."""
    ws = WeightedSum(
        [LeafRubric(1.0), LeafRubric(0.5)],
        weights=[0.7, 0.3]
    )
    # 1.0 * 0.7 + 0.5 * 0.3 = 0.7 + 0.15 = 0.85
    assert ws("a", "o") == pytest.approx(0.85)
