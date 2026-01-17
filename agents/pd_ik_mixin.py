"""PD/IK helper mixin for control-theoretic agents."""

import warnings
from typing import Optional

import numpy as np

from environment import Observation


class PDControlMixin:
    """Mixin providing PD control and simple IK helpers."""

    def __init__(self, kp: float = 5.0, kd: float = 0.5):
        super().__init__()
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

    def _compute_target_angle_for_pointing(self, observation: Observation) -> float:
        """
        Compute target angle for end effector to point at nearest sound source.

        This computes the angle from the center (base) to the sound source.
        """
        if not observation.sound_source_positions:
            return 0.0

        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        nearest_source = observation.sound_source_positions[nearest_idx]

        return np.arctan2(nearest_source[1], nearest_source[0])

    def _compute_target_angle_for_approaching(self, observation: Observation) -> float:
        """
        Compute target angle for end effector to move toward nearest sound source.

        This computes the angle from the current end effector position to the sound source.
        """
        if not observation.sound_source_positions:
            return 0.0

        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        nearest_source = observation.sound_source_positions[nearest_idx]

        direction = nearest_source - end_effector_pos
        return np.arctan2(direction[1], direction[0])

    def _compute_desired_joint_angles(self, target_angle: float,
                                     current_angles: np.ndarray) -> np.ndarray:
        """
        Compute desired joint angles using inverse kinematics (simplified).

        For a multi-link arm, we use a simple strategy:
        - First joint points toward target
        - Subsequent joints try to extend the arm
        """
        num_links = len(current_angles)
        desired_angles = np.zeros(num_links)

        if num_links == 1:
            desired_angles[0] = target_angle
        elif num_links == 2:
            desired_angles[0] = target_angle * 0.7
            desired_angles[1] = target_angle * 0.3
        else:
            desired_angles[0] = target_angle * 0.5
            desired_angles[1] = target_angle * 0.3
            desired_angles[2] = target_angle * 0.2

        return desired_angles

    def _forward_kinematics(self, angles: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
        """Compute forward kinematics: end effector position from joint angles."""
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
        """Solve inverse kinematics using a Jacobian-based method."""
        angles = current_angles.copy()
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        optimize_position = target_pos is not None and position_weight > 0.0
        optimize_orientation = target_orientation is not None and orientation_weight > 0.0

        if optimize_orientation:
            target_orientation = np.arctan2(np.sin(target_orientation), np.cos(target_orientation))

        for _ in range(max_iterations):
            current_pos = self._forward_kinematics(angles, link_lengths)

            pos_error = np.zeros(2)
            pos_error_magnitude = 0.0
            if optimize_position:
                pos_error = target_pos - current_pos
                pos_error_magnitude = np.linalg.norm(pos_error)

            orientation_error = 0.0
            if optimize_orientation:
                current_cumulative_angle = np.sum(angles)
                orientation_error = np.arctan2(
                    np.sin(target_orientation - current_cumulative_angle),
                    np.cos(target_orientation - current_cumulative_angle)
                )

            pos_converged = not optimize_position or pos_error_magnitude < tolerance
            orient_converged = not optimize_orientation or abs(orientation_error) < tolerance
            if pos_converged and orient_converged:
                break

            pos_jacobian = np.zeros((2, len(angles)))
            if optimize_position:
                epsilon = 1e-5
                for i in range(len(angles)):
                    perturbed_angles = angles.copy()
                    perturbed_angles[i] += epsilon
                    perturbed_pos = self._forward_kinematics(perturbed_angles, link_lengths)
                    pos_jacobian[:, i] = (perturbed_pos - current_pos) / epsilon

            orientation_jacobian = np.zeros((1, len(angles)))
            if optimize_orientation:
                orientation_jacobian = np.ones((1, len(angles)))

            if optimize_position and optimize_orientation:
                weighted_pos_jacobian = position_weight * pos_jacobian
                weighted_orientation_jacobian = orientation_weight * orientation_jacobian
                augmented_jacobian = np.vstack([weighted_pos_jacobian, weighted_orientation_jacobian])

                weighted_pos_error = position_weight * pos_error
                weighted_orientation_error = orientation_weight * orientation_error
                augmented_error = np.hstack([weighted_pos_error, weighted_orientation_error])
            elif optimize_position:
                augmented_jacobian = position_weight * pos_jacobian
                augmented_error = position_weight * pos_error
            elif optimize_orientation:
                augmented_jacobian = orientation_weight * orientation_jacobian
                augmented_error = np.array([orientation_weight * orientation_error])
            else:
                break

            try:
                jjt = augmented_jacobian @ augmented_jacobian.T
                jjt_damped = jjt + np.eye(jjt.shape[0]) * damping
                jjt_inv = np.linalg.inv(jjt_damped)
                delta_angles = augmented_jacobian.T @ jjt_inv @ augmented_error
                angles += 0.5 * delta_angles
            except np.linalg.LinAlgError:
                for i in range(len(angles)):
                    if np.linalg.norm(augmented_jacobian[:, i]) > 1e-6:
                        gradient = np.dot(augmented_error, augmented_jacobian[:, i]) / np.linalg.norm(augmented_jacobian[:, i])**2
                        angles[i] += 0.1 * gradient

        return np.arctan2(np.sin(angles), np.cos(angles))

    def _compute_pd_torques(self, desired_angles: np.ndarray,
                           current_angles: np.ndarray,
                           current_angular_velocities: np.ndarray) -> np.ndarray:
        """Compute torques using a PD controller."""
        angle_errors = desired_angles - current_angles
        angle_errors = np.arctan2(np.sin(angle_errors), np.cos(angle_errors))

        proportional_torques = self.kp * angle_errors
        derivative_torques = -self.kd * current_angular_velocities

        return proportional_torques + derivative_torques
