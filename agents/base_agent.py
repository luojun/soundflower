"""Base agent class with shared components for sound source tracking."""

import numpy as np
import warnings
from typing import Optional
from abc import ABC, abstractmethod
from environment import Observation


class BaseAgent(ABC):
    """Base class for agents with shared PD controller and utility functions."""

    def __init__(self, kp: float = 5.0, kd: float = 0.5):
        """
        Initialize base agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
        """
        self.kp = kp
        self.kd = kd
        self._warned_missing_sources = False
        self._warned_missing_link_lengths = False

    def _warn_if_missing_sources(self, observation: Observation) -> bool:
        """Warn once if sound source positions are unavailable (Sensorimotor mode)."""
        if observation.sound_source_positions:
            return False
        if not self._warned_missing_sources:
            warnings.warn(
                f"{self.__class__.__name__} received no sound source positions; "
                "behaving blindly with zero torques. This is expected in Sensorimotor mode.",
                RuntimeWarning,
                stacklevel=2
            )
            self._warned_missing_sources = True
        return True

    def _get_link_lengths(self, observation: Observation) -> Optional[np.ndarray]:
        """Get link lengths from observation, warning once if unavailable."""
        link_lengths = observation.link_lengths
        if link_lengths is not None:
            return link_lengths
        if not self._warned_missing_link_lengths:
            warnings.warn(
                f"{self.__class__.__name__} received no link_lengths; "
                "cannot run IK without Full observations.",
                RuntimeWarning,
                stacklevel=2
            )
            self._warned_missing_link_lengths = True
        return None

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

    def _forward_kinematics(self, angles: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics: end effector position from joint angles.

        Args:
            angles: Joint angles (rad)
            link_lengths: Lengths of arm links

        Returns:
            end_effector_pos: (x, y) position of end effector
        """
        joint_positions = np.zeros((len(link_lengths) + 1, 2))
        cumulative_angle = 0.0

        for i in range(len(link_lengths)):
            cumulative_angle += angles[i]
            joint_positions[i + 1] = joint_positions[i] + link_lengths[i] * np.array([
                np.cos(cumulative_angle),
                np.sin(cumulative_angle)
            ])

        return joint_positions[-1]

    def _solve_inverse_kinematics(self, current_angles: np.ndarray,
                                  link_lengths: np.ndarray,
                                  target_pos: np.ndarray = None,
                                  target_orientation: float = None,
                                  position_weight: float = 1.0,
                                  orientation_weight: float = 1.0,
                                  max_iterations: int = 20,
                                  tolerance: float = 1e-3,
                                  damping: float = 1e-6) -> np.ndarray:
        """
        Solve inverse kinematics to reach target position and/or orientation using Jacobian-based method.

        Args:
            current_angles: Current joint angles
            link_lengths: Lengths of arm links
            target_pos: Optional target (x, y) position for end effector (None to ignore position)
            target_orientation: Optional target orientation angle for last link (None to ignore orientation)
            position_weight: Weight for position objective (0.0 to ignore position)
            orientation_weight: Weight for orientation objective (0.0 to ignore orientation)
            max_iterations: Maximum iterations for IK solver
            tolerance: Convergence tolerance
            damping: Damping factor for regularization

        Returns:
            desired_angles: Joint angles to reach target position and/or orientation
        """
        angles = current_angles.copy()
        # Normalize input angles to [-π, π] for consistency
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        optimize_position = target_pos is not None and position_weight > 0.0
        optimize_orientation = target_orientation is not None and orientation_weight > 0.0

        if optimize_orientation:
            target_orientation = np.arctan2(np.sin(target_orientation), np.cos(target_orientation))

        for iteration in range(max_iterations):
            # Forward kinematics
            current_pos = self._forward_kinematics(angles, link_lengths)

            # Position error
            pos_error = np.zeros(2)
            pos_error_magnitude = 0.0
            if optimize_position:
                pos_error = target_pos - current_pos
                pos_error_magnitude = np.linalg.norm(pos_error)

            # Orientation error
            orientation_error = 0.0
            if optimize_orientation:
                current_cumulative_angle = np.sum(angles)
                orientation_error = np.arctan2(
                    np.sin(target_orientation - current_cumulative_angle),
                    np.cos(target_orientation - current_cumulative_angle)
                )

            # Check convergence
            pos_converged = not optimize_position or pos_error_magnitude < tolerance
            orient_converged = not optimize_orientation or abs(orientation_error) < tolerance
            if pos_converged and orient_converged:
                break

            # Compute position Jacobian numerically
            pos_jacobian = np.zeros((2, len(angles)))
            if optimize_position:
                epsilon = 1e-5
                for i in range(len(angles)):
                    perturbed_angles = angles.copy()
                    perturbed_angles[i] += epsilon
                    perturbed_pos = self._forward_kinematics(perturbed_angles, link_lengths)
                    pos_jacobian[:, i] = (perturbed_pos - current_pos) / epsilon

            # Compute orientation Jacobian
            # Since microphone_orientation = sum(angles), the Jacobian is all ones
            orientation_jacobian = np.zeros((1, len(angles)))
            if optimize_orientation:
                orientation_jacobian = np.ones((1, len(angles)))

            # Combine into augmented Jacobian and error
            if optimize_position and optimize_orientation:
                # Both objectives: stack position and orientation
                weighted_pos_jacobian = position_weight * pos_jacobian
                weighted_orientation_jacobian = orientation_weight * orientation_jacobian
                augmented_jacobian = np.vstack([weighted_pos_jacobian, weighted_orientation_jacobian])

                weighted_pos_error = position_weight * pos_error
                weighted_orientation_error = orientation_weight * orientation_error
                augmented_error = np.hstack([weighted_pos_error, weighted_orientation_error])
            elif optimize_position:
                # Position only
                augmented_jacobian = position_weight * pos_jacobian
                augmented_error = position_weight * pos_error
            elif optimize_orientation:
                # Orientation only
                augmented_jacobian = orientation_weight * orientation_jacobian
                augmented_error = np.array([orientation_weight * orientation_error])
            else:
                # Neither (shouldn't happen, but handle gracefully)
                break

            # Solve: delta_angles = J^+ * error (damped least squares)
            try:
                jjt = augmented_jacobian @ augmented_jacobian.T
                jjt_damped = jjt + np.eye(jjt.shape[0]) * damping
                jjt_inv = np.linalg.inv(jjt_damped)
                delta_angles = augmented_jacobian.T @ jjt_inv @ augmented_error
                angles += 0.5 * delta_angles  # Damped update
            except np.linalg.LinAlgError:
                # Fallback: gradient descent
                for i in range(len(angles)):
                    if np.linalg.norm(augmented_jacobian[:, i]) > 1e-6:
                        gradient = np.dot(augmented_error, augmented_jacobian[:, i]) / np.linalg.norm(augmented_jacobian[:, i])**2
                        angles[i] += 0.1 * gradient

            # Don't clip during iterations - allow angles to evolve naturally
            # Normalization will happen at the end to avoid discontinuities

        # Normalize angles to [-π, π] at the end
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        return angles

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
        # Wrap errors to [-π, π] to handle angle wrap-around
        angle_errors = np.arctan2(np.sin(angle_errors), np.cos(angle_errors))

        # Proportional term
        proportional_torques = self.kp * angle_errors

        # Derivative term (damping)
        derivative_torques = -self.kd * current_angular_velocities

        # Total torques
        torques = proportional_torques + derivative_torques

        # Do not clip torques here - let physics boundary handle it
        # This is important for future RL learning where agents need to observe actual control outputs

        return torques

