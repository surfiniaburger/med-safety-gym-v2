from typing import Dict, List, Any, Callable, Optional, Iterator, Tuple
import abc

class Rubric(abc.ABC):
    """
    Base class for all reward components. 
    Inspired by torch.nn.Module for composability and observability.
    """
    def __init__(self):
        self._children: Dict[str, 'Rubric'] = {}
        self._forward_hooks: List[Callable] = []
        self.last_score: float = 0.0

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Rubric):
            self._children[name] = value
        super().__setattr__(name, value)

    @abc.abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        """
        Compute reward. Environment authors should implement this.
        
        Args:
            action: The action taken by the agent.
            observation: The observation resulting from the action.
            
        Returns:
            A float representing the reward/score.
        """
        pass

    def __call__(self, action: Any, observation: Any) -> float:
        """
        Execute the rubric and all registered hooks.
        """
        score = self.forward(action, observation)
        self.last_score = score
        
        # Post-hooks for observability
        for hook in self._forward_hooks:
            hook(self, action, observation, score)
            
        return score

    def children(self) -> Iterator['Rubric']:
        """Returns an iterator over immediate child rubrics."""
        for child in self._children.values():
            yield child

    def named_children(self) -> Iterator[Tuple[str, 'Rubric']]:
        """Returns an iterator over immediate child rubrics, yielding both the name and the rubric."""
        for name, child in self._children.items():
            yield name, child

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, 'Rubric']]:
        """
        Returns an iterator over all rubrics in the hierarchy, yielding both the name and the rubric.
        """
        yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_rubrics(child_prefix)

    def register_forward_hook(self, hook: Callable):
        """
        Registers a forward hook on the rubric.
        The hook will be called every time after forward() has computed a result.
        
        Signature: hook(rubric, action, observation, result) -> None
        """
        self._forward_hooks.append(hook)

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the state of the rubric and its children."""
        state = {}
        for name, child in self._children.items():
            state[name] = child.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state of the rubric and its children from a dictionary."""
        for name, child in self._children.items():
            if name in state_dict:
                child.load_state_dict(state_dict[name])

class Sequential(Rubric):
    """
    Runs child rubrics in order. 
    If any returns 0.0, stops immediately and returns 0.0 (fail-fast).
    Otherwise returns the score of the last rubric.
    """
    def __init__(self, *rubrics: Rubric):
        super().__init__()
        for i, r in enumerate(rubrics):
            setattr(self, f"step_{i}", r)

    def forward(self, action: Any, observation: Any) -> float:
        score = 0.0
        for child in self.children():
            score = child(action, observation)
            if score <= 0.0:
                return 0.0
        return score

class Gate(Rubric):
    """
    Wraps a rubric with a threshold. 
    Returns the child's score if it meets the threshold, otherwise returns 0.0.
    """
    def __init__(self, rubric: Rubric, threshold: float = 1.0):
        super().__init__()
        self.rubric = rubric
        self.threshold = threshold

    def forward(self, action: Any, observation: Any) -> float:
        score = self.rubric(action, observation)
        return score if score >= self.threshold else 0.0

class WeightedSum(Rubric):
    """
    Computes a weighted combination of child rubrics.
    """
    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError("Number of rubrics must match number of weights")
        
        self.weights = weights
        for i, r in enumerate(rubrics):
            setattr(self, f"rubric_{i}", r)

    def forward(self, action: Any, observation: Any) -> float:
        total = 0.0
        for i, child in enumerate(self.children()):
            total += child(action, observation) * self.weights[i]
        return total

class RubricList(Rubric):
    """
    A container for a list of rubrics. 
    Does not define aggregation - use within a parent rubric.
    """
    def __init__(self, rubrics: List[Rubric]):
        super().__init__()
        for i, r in enumerate(rubrics):
            setattr(self, f"item_{i}", r)

    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError("RubricList does not implement forward. Use it as a container.")

class RubricDict(Rubric):
    """
    A container for a dictionary of rubrics.
    """
    def __init__(self, rubrics: Dict[str, Rubric]):
        super().__init__()
        for name, r in rubrics.items():
            setattr(self, name, r)

    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError("RubricDict does not implement forward. Use it as a container.")
