"""Base agent interface for the Sound Flower environment."""

from abc import ABC, abstractmethod
from environment import Observation


class BaseAgent(ABC):
    """Minimal base class for agents."""

    def __init__(self) -> None:
        """Initialize the base agent."""
        return

    @abstractmethod
    def decide(self, observation: Observation, reward: float | None):
        """
        Decide action given current observation and reward from the previous action.

        Args:
            observation: Current observation (state after the last applied action).
            reward: Reward received for the previous action; None on first call.

        Returns:
            action: Torques to apply at each joint.
        """
        raise NotImplementedError
