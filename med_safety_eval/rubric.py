from typing import Dict, List, Any, Callable, Optional, Union, Iterable, Iterator, Tuple, TypeVar
import inspect
import asyncio
import logging

T = TypeVar("T", bound="Rubric")

import abc
import re

class Rubric(abc.ABC):
    """
    Base class for all reward components. 
    Inspired by torch.nn.Module for composability and observability.
    """
    def __init__(self):
        self._forward_hooks: List[Callable] = []
        self._forward_pre_hooks: List[Callable] = []
        self._children: Dict[str, 'Rubric'] = {}
        self._name: Optional[str] = None
        self.last_score: float = 0.0

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Rubric) and name != "_children":
            self._children[name] = value
        super().__setattr__(name, value)

    def register_forward_hook(self, hook: Callable) -> None:
        """
        Registers a forward hook on the rubric.
        Signature: hook(rubric, action, observation, result) -> None
        """
        self._forward_hooks.append(hook)

    def register_forward_pre_hook(self, hook: Callable) -> None:
        """
        Registers a forward pre-hook on the rubric.
        Signature: hook(rubric, action, observation) -> None
        """
        self._forward_pre_hooks.append(hook)

    def __call__(self, action: Any, observation: Any) -> float:
        """Execute the rubric and all registered hooks."""
        # Run pre-hooks
        for hook in self._forward_pre_hooks:
            hook(self, action, observation)

        score = self.forward(action, observation)
        self.last_score = score

        # Run post-hooks
        for hook in self._forward_hooks:
            hook(self, action, observation, score)
        return score

    @abc.abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        """Compute reward. Environment authors should implement this."""
        pass

    def get_rubric(self, path: str) -> 'Rubric':
        """
        Retrieves a nested rubric by its dot-separated path.
        Example: rubric.get_rubric("grounding.fuzzy_match")
        """
        if not path:
            return self
        
        parts = path.split(".", 1)
        head = parts[0]
        tail = parts[1] if len(parts) > 1 else None

        if head in self._children:
            child = self._children[head]
        else:
            # Fallback for attributes not in _children but are Rubrics
            child = getattr(self, head, None)
            if not isinstance(child, Rubric):
                raise KeyError(f"Rubric path '{head}' not found in {self.__class__.__name__}")

        if tail:
            return child.get_rubric(tail)
        return child

    def children(self) -> Iterator['Rubric']:
        """Returns an iterator over immediate child rubrics."""
        yield from self._children.values()

    def named_children(self) -> Iterator[Tuple[str, 'Rubric']]:
        """Returns an iterator over immediate child rubrics, yielding both name and rubric."""
        yield from self._children.items()

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, 'Rubric']]:
        """Returns an iterator over all rubrics in the hierarchy."""
        yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_rubrics(child_prefix)

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the state of the rubric and its children."""
        return {name: child.state_dict() for name, child in self._children.items()}

    def capture_scores(self) -> Dict[str, Any]:
        """
        Captures the current scores of all rubrics in the hierarchy.
        Returns a dictionary mapping dot-separated paths to scores.
        """
        return {path or "root": getattr(rubric, "last_score", 0.0) for path, rubric in self.named_rubrics()}

    def capture_snapshot(self, action: Any = None, observation: Any = None) -> "NeuralSnapshot":
        """
        Captures a NeuralSnapshot of the current state.
        Default implementation using capture_scores().
        """
        # Avoid circular import
        from med_safety_eval.schemas import NeuralSnapshot
        
        scores = self.capture_scores()
        # Ensure strict floats
        safe_scores = {k: float(v) if v is not None else 0.0 for k, v in scores.items()}
        
        return NeuralSnapshot(
            session_id="local_eval",
            step=0,
            scores=safe_scores,
            metadata={
                "action": str(action) if action is not None else "",
                "observation": str(observation) if observation is not None else ""
            }
        )

    def update_config(self, config: Dict[str, Any]):
        """
        Updates the rubric's configuration (e.g., reward values).
        Environment authors can override this for specific logic.
        Default: Recursively update children and try to set attributes.
        """
        for k, v in config.items():
            if k in self._children:
                self._children[k].update_config(v)
            elif hasattr(self, k):
                # Only update if it's a basic type (not a child rubric or method)
                current_val = getattr(self, k)
                if not isinstance(current_val, (Rubric, Callable)):
                    setattr(self, k, v)
                    logger.debug(f"Updated {self.__class__.__name__}.{k} to {v}")

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
    """A container for a list of rubrics."""
    def __init__(self, rubrics: List[Rubric]):
        super().__init__()
        for i, r in enumerate(rubrics):
            setattr(self, f"item_{i}", r)

    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError("RubricList does not implement forward. Use it as a container.")

class RubricDict(Rubric):
    """A container for a dictionary of rubrics."""
    def __init__(self, rubrics: Dict[str, Rubric]):
        super().__init__()
        for name, r in rubrics.items():
            setattr(self, name, r)

    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError("RubricDict does not implement forward. Use it as a container.")

class LLMJudge(Rubric):
    """
    A rubric that uses an LLM to evaluate the action.
    """
    def __init__(
        self,
        prompt_template: str,
        inference_fn: Callable[[str], str],
        score_parser: Optional[Callable[[str], float]] = None
    ):
        super().__init__()
        self.prompt_template = prompt_template
        self.inference_fn = inference_fn
        self.score_parser = score_parser or self._default_score_parser

    def forward(self, action: Any, observation: Any) -> float:
        action_str = getattr(action, 'content', str(action))
        
        if hasattr(observation, 'context'):
             obs_context = observation.context
             obs_question = getattr(observation, 'question', "")
             obs_str = f"Context: {obs_context}\nQuestion: {obs_question}"
        else:
             obs_str = str(observation)

        prompt = self.prompt_template.format(action=action_str, observation=obs_str)
        response = self.inference_fn(prompt)
        return self.score_parser(response)

    def _default_score_parser(self, response: str) -> float:
        match = re.search(r"Score:\s*(1\.0*|0?\.\d+|1|0)", response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        if "PASS" in response: return 1.0
        if "FAIL" in response: return 0.0
        return 0.0

