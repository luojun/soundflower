"""Agent that both points toward and minimizes distance to sound source."""

import numpy as np
from environment import Observation
from .base_agent import BaseAgent


class TrackingAgent(BaseAgent):
    """
    Agent that both points toward and minimizes distance to the sound source.

    This agent combines orientation control (pointing) and distance minimization,
    achieving optimal sound reception through both proper orientation and proximity.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0,
                 pointing_weight: float = 0.5):
        """
        Initialize tracking agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
            pointing_weight: Weight for pointing objective (0-1).
                           Distance minimization weight is (1 - pointing_weight)
        """
        super().__init__(kp=kp, kd=kd, max_torque=max_torque)
        self.pointing_weight = pointing_weight
        self.target_angle = 0.0

    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action combining pointing and distance minimization.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        # Compute target angles for both objectives
        pointing_angle = self._compute_target_angle_for_pointing(observation)
        approaching_angle = self._compute_target_angle_for_approaching(observation)

        # Combine angles based on weights
        # Normalize angles to [-pi, pi] before combining
        pointing_angle = np.arctan2(np.sin(pointing_angle), np.cos(pointing_angle))
        approaching_angle = np.arctan2(np.sin(approaching_angle), np.cos(approaching_angle))

        # Weighted combination of angles
        # For angles, we need to handle the wrap-around at ±π
        # Use vector addition approach: convert to unit vectors, combine, then convert back
        pointing_vec = np.array([np.cos(pointing_angle), np.sin(pointing_angle)])
        approaching_vec = np.array([np.cos(approaching_angle), np.sin(approaching_angle)])

        combined_vec = (self.pointing_weight * pointing_vec +
                       (1 - self.pointing_weight) * approaching_vec)

        # Normalize and convert back to angle
        combined_vec_norm = combined_vec / (np.linalg.norm(combined_vec) + 1e-6)
        target_angle = np.arctan2(combined_vec_norm[1], combined_vec_norm[0])
        self.target_angle = target_angle

        # Compute desired joint angles
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

