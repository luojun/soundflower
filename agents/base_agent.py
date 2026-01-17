"""Base agent interface for the Sound Flower environment."""

from abc import ABC, abstractmethod
from environment import Observation


class BaseAgent(ABC):
    """Minimal base class for agents."""

    def __init__(self) -> None:
        """Initialize the base agent."""
        return

    @abstractmethod
    def select_action(self, observation: Observation):
        """
        Select action based on observation.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        raise NotImplementedError

    def observe(self, reward: float, observation: Observation) -> None:
        """
        Optional learning hook for agents that update from reward/observation.

        Args:
            reward: Reward received for the previous action.
            observation: Current observation after transition.
        """
        return
