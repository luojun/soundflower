"""Base agent class with shared components for sound source tracking."""

import numpy as np
from abc import ABC, abstractmethod
from environment import Observation


class BaseAgent(ABC):
    """Base class for agents with shared PD controller and utility functions."""

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0):
        """
        Initialize base agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
        """
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque

    @abstractmethod
    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action based on observation.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        pass

    def _compute_target_angle_for_pointing(self, observation: Observation) -> float:
        """
        Compute target angle for end effector to point at nearest sound source.

        This computes the angle from the center (base) to the sound source,
        which is used for orientation control.

        Args:
            observation: Current observation

        Returns:
            target_angle: Target angle in radians (from center to sound source)
        """
        if not observation.sound_source_positions:
            return 0.0

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        nearest_source = observation.sound_source_positions[nearest_idx]

        # Compute angle from center (origin) to sound source
        target_angle = np.arctan2(nearest_source[1], nearest_source[0])

        return target_angle

    def _compute_target_angle_for_approaching(self, observation: Observation) -> float:
        """
        Compute target angle for end effector to move toward nearest sound source.

        This computes the angle from the current end effector position to the sound source,
        which is used for distance minimization.

        Args:
            observation: Current observation

        Returns:
            target_angle: Target angle in radians (from end effector to sound source)
        """
        if not observation.sound_source_positions:
            return 0.0

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        nearest_source = observation.sound_source_positions[nearest_idx]

        # Compute angle from end effector to sound source
        direction = nearest_source - end_effector_pos
        target_angle = np.arctan2(direction[1], direction[0])

        return target_angle

    def _compute_desired_joint_angles(self, target_angle: float,
                                     current_angles: np.ndarray) -> np.ndarray:
        """
        Compute desired joint angles using inverse kinematics (simplified).

        For a multi-link arm, we use a simple strategy:
        - First joint points toward target
        - Subsequent joints try to extend the arm

        Args:
            target_angle: Target angle for end effector
            current_angles: Current joint angles

        Returns:
            desired_angles: Desired joint angles
        """
        num_links = len(current_angles)
        desired_angles = np.zeros(num_links)

        if num_links == 1:
            # Single link: just point at target
            desired_angles[0] = target_angle
        elif num_links == 2:
            # Two links: first joint points roughly toward target
            # Second joint extends the arm
            desired_angles[0] = target_angle * 0.7  # First joint does most of the pointing
            desired_angles[1] = target_angle * 0.3  # Second joint fine-tunes
        else:
            # Three links: distribute angle across joints
            desired_angles[0] = target_angle * 0.5
            desired_angles[1] = target_angle * 0.3
            desired_angles[2] = target_angle * 0.2

        return desired_angles

    def _compute_pd_torques(self, desired_angles: np.ndarray,
                           current_angles: np.ndarray,
                           current_angular_velocities: np.ndarray) -> np.ndarray:
        """
        Compute torques using PD controller.

        Args:
            desired_angles: Desired joint angles
            current_angles: Current joint angles
            current_angular_velocities: Current joint angular velocities

        Returns:
            torques: Torques to apply at each joint
        """
        # Compute angle errors
        angle_errors = desired_angles - current_angles

        # Proportional term
        proportional_torques = self.kp * angle_errors

        # Derivative term (damping)
        derivative_torques = -self.kd * current_angular_velocities

        # Total torques
        torques = proportional_torques + derivative_torques

        # Clamp to valid range
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        return torques

