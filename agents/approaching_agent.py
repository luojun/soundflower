"""Agent that only minimizes distance to sound source without orientation control."""

import numpy as np
from environment import Observation
from .base_agent import BaseAgent


class ApproachingAgent(BaseAgent):
    """
    Agent that only minimizes distance to the sound source.

    This agent moves the end effector closer to the sound source without
    explicit orientation control. Orientation may change naturally but is not
    subject to IK optimization pressure.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0,
                 link_lengths: np.ndarray = None):
        """
        Initialize approaching agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clipping)
            link_lengths: Array of link lengths for IK computation
        """
        super().__init__(kp=kp, kd=kd, max_torque=max_torque, link_lengths=link_lengths)
        self.target_angle = 0.0

    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action to minimize distance to sound source.

        Minimizes distance while ignoring target orientation. Orientation may change
        naturally but is not subject to IK optimization pressure.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        if not observation.sound_source_positions:
            return np.zeros(len(observation.arm_angles))

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        source_pos = observation.sound_source_positions[nearest_idx]

        # Optimize directly toward source position (position-only, orientation_weight=0.0 to ignore orientation)
        # The IK solver naturally handles unreachable targets by converging to the closest reachable point.
        # The physics engine enforces min_distance_to_source constraint automatically.
        desired_angles = self._solve_inverse_kinematics(
            current_angles=observation.arm_angles,
            link_lengths=self.link_lengths,
            target_pos=source_pos,
            target_orientation=None,
            position_weight=1.0,
            orientation_weight=0.0
        )

        # Compute PD torques
        torques = self._compute_pd_torques(
            desired_angles,
            observation.arm_angles,
            observation.arm_angular_velocities
        )

        return torques

