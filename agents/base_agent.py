"""Base agent class with shared components for sound source tracking."""

import numpy as np
from abc import ABC, abstractmethod
from environment import Observation


class BaseAgent(ABC):
    """Base class for agents with shared PD controller and utility functions."""

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0,
                 link_lengths: np.ndarray = None):
        """
        Initialize base agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
            link_lengths: Array of link lengths for IK computation (required for position-based IK)
        """
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque
        self.link_lengths = link_lengths if link_lengths is not None else np.array([0.5, 0.5])

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

    def _solve_inverse_kinematics(self, target_pos: np.ndarray,
                                  current_angles: np.ndarray,
                                  link_lengths: np.ndarray,
                                  max_iterations: int = 20,
                                  tolerance: float = 1e-3,
                                  damping: float = 1e-6) -> np.ndarray:
        """
        Solve inverse kinematics to reach target position using Jacobian-based method.

        Args:
            target_pos: Target (x, y) position for end effector
            current_angles: Current joint angles
            link_lengths: Lengths of arm links
            max_iterations: Maximum iterations for IK solver
            tolerance: Convergence tolerance
            damping: Damping factor for regularization

        Returns:
            desired_angles: Joint angles to reach target position
        """
        angles = current_angles.copy()

        for iteration in range(max_iterations):
            # Forward kinematics
            current_pos = self._forward_kinematics(angles, link_lengths)
            error = target_pos - current_pos
            error_magnitude = np.linalg.norm(error)

            if error_magnitude < tolerance:
                break

            # Compute Jacobian numerically
            jacobian = np.zeros((2, len(angles)))
            epsilon = 1e-5

            for i in range(len(angles)):
                perturbed_angles = angles.copy()
                perturbed_angles[i] += epsilon
                perturbed_pos = self._forward_kinematics(perturbed_angles, link_lengths)
                jacobian[:, i] = (perturbed_pos - current_pos) / epsilon

            # Solve: delta_angles = J^+ * error (damped least squares)
            # Use pseudo-inverse with damping: delta_angles = J^T * (J * J^T + λI)^(-1) * error
            try:
                jjt = jacobian @ jacobian.T
                jjt_damped = jjt + np.eye(2) * damping
                jjt_inv = np.linalg.inv(jjt_damped)
                delta_angles = jacobian.T @ jjt_inv @ error
                angles += 0.5 * delta_angles  # Damped update
            except np.linalg.LinAlgError:
                # Fallback: gradient descent
                for i in range(len(angles)):
                    if np.linalg.norm(jacobian[:, i]) > 1e-6:
                        gradient = np.dot(error, jacobian[:, i]) / np.linalg.norm(jacobian[:, i])**2
                        angles[i] += 0.1 * gradient

            # Clamp angles to reasonable range
            angles = np.clip(angles, -np.pi, np.pi)

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

        # Proportional term
        proportional_torques = self.kp * angle_errors

        # Derivative term (damping)
        derivative_torques = -self.kd * current_angular_velocities

        # Total torques
        torques = proportional_torques + derivative_torques

        # Clamp to valid range
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        return torques

