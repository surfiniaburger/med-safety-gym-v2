from abc import abstractmethod
from typing import List, Tuple, Any, Optional
from med_safety_eval.rubric import Rubric

class TrajectoryRubric(Rubric):
    """Abstract base for rubrics that score based on full trajectories.

    Subclasses implement:
    - score_trajectory(): Compute final score from trajectory
    - compute_step_rewards(): Define credit assignment strategy

    The __call__ method accumulates steps and returns rewards according
    to the subclass's implementation.

    IMPORTANT: Trajectories are stored in CPU memory to avoid GPU pressure.
    """

    def __init__(self, intermediate_reward: float = 0.0):
        super().__init__()
        self.intermediate_reward = intermediate_reward
        self._trajectory: List[Tuple[Any, Any]] = []

    def forward(self, action: Any, observation: Any) -> float:
        """Accumulate step and return reward.

        Returns intermediate_reward until done, then computes trajectory score.
        """
        self._trajectory.append((action, observation))

        # Check if observation indicates the episode is done
        is_done = False
        if hasattr(observation, 'done'):
            is_done = observation.done
        elif isinstance(observation, dict) and 'done' in observation:
            is_done = observation['done']
        
        if is_done:
            return self.score_trajectory(self._trajectory)
        else:
            return self.intermediate_reward

    @abstractmethod
    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Score the complete trajectory. Return 0.0-1.0.

        Called when observation.done=True.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step_rewards(self) -> List[float]:
        """Compute per-step rewards from the accumulated trajectory.

        Returns: List of rewards, one per step.
        Define your credit assignment strategy here.
        """
        raise NotImplementedError

    def reset(self):
        """Clear accumulated trajectory. Call on env.reset()."""
        self._trajectory = []
        for child in self.children():
            if hasattr(child, 'reset'):
                child.reset()

    @property
    def trajectory(self) -> List[Tuple[Any, Any]]:
        """Current trajectory (read-only copy)."""
        return list(self._trajectory)

class ExponentialDiscountingTrajectoryRubric(TrajectoryRubric):
    """TrajectoryRubric with exponential discounting for credit assignment.

    Per-step reward: r_t = gamma^(T-1-t) * R_final

    With gamma=0.99, later steps get higher reward (they're "closer" to the outcome).
    With gamma=1.0, all steps get equal reward.
    """

    def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
        super().__init__(intermediate_reward=intermediate_reward)
        self.gamma = gamma

    def compute_step_rewards(self) -> List[float]:
        """Apply exponential discounting from final reward."""
        if not self._trajectory:
            return []

        final_score = self.score_trajectory(self._trajectory)
        T = len(self._trajectory)
        return [final_score * (self.gamma ** (T - 1 - t)) for t in range(T)]
