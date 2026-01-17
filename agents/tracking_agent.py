"""Agent that both points toward and minimizes distance to sound source."""

import numpy as np
from environment import Observation
from .base_agent import BaseAgent
from .pd_ik_mixin import PDControlMixin


class TrackingAgent(PDControlMixin, BaseAgent):
    """
    Agent that both points toward and minimizes distance to the sound source.

    This agent combines orientation control (pointing) and distance minimization,
    achieving optimal sound reception through both proper orientation and proximity.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.5, pointing_weight: float = 0.5):
        """
        Initialize tracking agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            pointing_weight: Weight for pointing objective (0-1).
                           Distance minimization weight is (1 - pointing_weight)
        """
        super().__init__(kp=kp, kd=kd)
        self.pointing_weight = pointing_weight
        self.target_angle = 0.0

    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action combining pointing and distance minimization.

        Optimizes both orientation and distance simultaneously using weighted combination.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        if self._warn_if_missing_sources(observation):
            return np.zeros(len(observation.arm_angles))

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        source_pos = observation.sound_source_positions[nearest_idx]

        # Compute target orientation: direction from end effector to source
        direction_to_source = source_pos - end_effector_pos
        distance_to_source = np.linalg.norm(direction_to_source)
        if distance_to_source > 1e-6:
            direction_to_source_normalized = direction_to_source / distance_to_source
            target_orientation = np.arctan2(direction_to_source_normalized[1], direction_to_source_normalized[0])
        else:
            # Fallback: if source is at end effector, use base-to-source direction
            if np.linalg.norm(source_pos) > 1e-6:
                target_orientation = np.arctan2(source_pos[1], source_pos[0])
            else:
                target_orientation = observation.microphone_orientation

        link_lengths = self._get_link_lengths(observation)
        if link_lengths is None:
            return np.zeros(len(observation.arm_angles))

        # Optimize directly toward source position with both position and orientation objectives
        # The IK solver naturally handles unreachable targets by converging to the closest reachable point.
        # The physics engine enforces min_distance_to_source constraint automatically.
        # Use pointing_weight to balance position vs orientation emphasis
        desired_angles = self._solve_inverse_kinematics(
            current_angles=observation.arm_angles,
            link_lengths=link_lengths,
            target_pos=source_pos,
            target_orientation=target_orientation,
            position_weight=1.0,
            orientation_weight=self.pointing_weight
        )

        # Compute PD torques
        torques = self._compute_pd_torques(
            desired_angles,
            observation.arm_angles,
            observation.arm_angular_velocities
        )

        return torques

