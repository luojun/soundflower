"""Agent that only minimizes distance to sound source without orientation control."""

import numpy as np
from environment import Observation
from .base_agent import BaseAgent


class ApproachingAgent(BaseAgent):
    """
    Agent that only minimizes distance to the sound source.

    This agent moves the end effector closer to the sound source without
    explicit orientation control. This is the behavior of the original HeuristicAgent.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0):
        """
        Initialize approaching agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
        """
        super().__init__(kp=kp, kd=kd, max_torque=max_torque)
        self.target_angle = 0.0

    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action to minimize distance to sound source.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        # Compute target angle for approaching (from end effector to source)
        target_angle = self._compute_target_angle_for_approaching(observation)
        self.target_angle = target_angle

        # Compute desired joint angles for approaching
        desired_angles = self._compute_desired_joint_angles(
            target_angle, observation.arm_angles
        )

        # Compute PD torques
        torques = self._compute_pd_torques(
            desired_angles,
            observation.arm_angles,
            observation.arm_angular_velocities
        )

        return torques

