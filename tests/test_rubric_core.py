import pytest
from med_safety_eval.rubric import Rubric, Sequential, Gate

class MockLeafRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if action == "correct" else 0.0

def test_register_forward_pre_hook():
    """Test 1.1: Pre-Forward Hooks capture input before execution."""
    rubric = MockLeafRubric()
    captured_input = []

    def pre_hook(r, action, observation):
        captured_input.append((action, observation))

    rubric.register_forward_pre_hook(pre_hook)
    
    score = rubric("correct", "some_obs")
    
    assert score == 1.0
    assert len(captured_input) == 1
    assert captured_input[0] == ("correct", "some_obs")

def test_get_rubric_nested_path():
    """Test 1.2: Nested Path Access for granular introspection."""
    class CompositeRubric(Rubric):
        def __init__(self):
            super().__init__()
            self.inner = MockLeafRubric()
            self.nested = Sequential(MockLeafRubric())

        def forward(self, action, observation) -> float:
            return self.inner(action, observation)

    root = CompositeRubric()
    
    # Test direct child
    inner = root.get_rubric("inner")
    assert isinstance(inner, MockLeafRubric)
    
    # Test nested child
    nested_leaf = root.get_rubric("nested.step_0")
    assert isinstance(nested_leaf, MockLeafRubric)
    
    # Test non-existent path
    with pytest.raises(KeyError):
        root.get_rubric("non.existent")

def test_pre_hook_execution_order():
    """Verify pre-hooks run before forward and post-hooks run after."""
    execution_log = []

    class OrderRubric(Rubric):
        def forward(self, action, observation) -> float:
            execution_log.append("forward")
            return 1.0

    rubric = OrderRubric()
    rubric.register_forward_pre_hook(lambda *args: execution_log.append("pre"))
    rubric.register_forward_hook(lambda *args: execution_log.append("post"))

    rubric("act", "obs")
    assert execution_log == ["pre", "forward", "post"]
