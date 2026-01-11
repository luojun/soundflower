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
                 pointing_weight: float = 0.5, link_lengths: np.ndarray = None,
                 min_distance_to_source: float = 0.2):
        """
        Initialize tracking agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
            pointing_weight: Weight for pointing objective (0-1).
                           Distance minimization weight is (1 - pointing_weight)
            link_lengths: Array of link lengths for IK computation
            min_distance_to_source: Minimum distance constraint (meters)
        """
        super().__init__(kp=kp, kd=kd, max_torque=max_torque, link_lengths=link_lengths)
        self.pointing_weight = pointing_weight
        self.min_distance_to_source = min_distance_to_source
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
        if not observation.sound_source_positions:
            return np.zeros(len(observation.arm_angles))

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        source_pos = observation.sound_source_positions[nearest_idx]

        # Compute target position and orientation for both objectives
        direction_to_source = source_pos - end_effector_pos
        distance_to_source = np.linalg.norm(direction_to_source)

        # Target orientation: direction from end effector to source
        if distance_to_source > 1e-6:
            direction_to_source_normalized = direction_to_source / distance_to_source
            target_orientation = np.arctan2(direction_to_source_normalized[1], direction_to_source_normalized[0])
        else:
            # Fallback: if source is at end effector, use base-to-source direction
            if np.linalg.norm(source_pos) > 1e-6:
                target_orientation = np.arctan2(source_pos[1], source_pos[0])
            else:
                target_orientation = observation.microphone_orientation

        # Target position: weighted combination of pointing and approaching targets
        current_distance = np.linalg.norm(end_effector_pos)
        if current_distance > 1e-6:
            direction_to_source_normalized = source_pos / np.linalg.norm(source_pos)
            pointing_target = direction_to_source_normalized * current_distance
        else:
            pointing_target = source_pos

        # Approaching target: move toward source (respecting minimum distance)
        if distance_to_source > self.min_distance_to_source:
            step_size = min(0.1, distance_to_source - self.min_distance_to_source)
            approaching_target = end_effector_pos + (direction_to_source / distance_to_source) * step_size
        else:
            approaching_target = end_effector_pos

        # Weighted combination of target positions
        target_pos = (self.pointing_weight * pointing_target +
                     (1 - self.pointing_weight) * approaching_target)

        # Solve IK with both position and orientation objectives
        # Use pointing_weight to balance position vs orientation emphasis
        desired_angles = self._solve_inverse_kinematics(
            current_angles=observation.arm_angles,
            link_lengths=self.link_lengths,
            target_pos=target_pos,
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

