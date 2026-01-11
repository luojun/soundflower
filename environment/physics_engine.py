"""Asynchronous physics engine for Sound Flower."""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from .physics import ArmPhysics, ArmState, SoundPropagation


@dataclass
class PhysicsState:
    """Complete physics state of the system."""
    arm_state: ArmState
    sound_source_angles: list


class PhysicsEngine:
    """Asynchronous physics engine that runs independently."""

    def __init__(self, config):
        """
        Initialize physics engine.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize physics components
        self.arm_physics = ArmPhysics(
            link_lengths=config.link_lengths,
            link_masses=config.link_masses,
            joint_frictions=config.joint_frictions,
            dt=config.dt
        )

        self.sound_propagation = SoundPropagation(
            attenuation_coeff=config.sound_attenuation_coeff
        )

        # Initialize state
        self.state = PhysicsState(
            arm_state=ArmState(
                angles=np.zeros(config.num_links),
                angular_velocities=np.zeros(config.num_links)
            ),
            sound_source_angles=[]
        )

        self._initialize_sound_sources()

        # Ensure initial arm state respects minimum distance constraint
        self._enforce_minimum_distance_constraint()

        # Current torques (set by control loop)
        self.current_torques = np.zeros(config.num_links)

        # Running state
        self.running = False


    def _initialize_sound_sources(self):
        """Initialize sound sources."""
        self.state.sound_source_angles = []
        for i in range(self.config.num_sound_sources):
            angle = self.config.sound_source_initial_angle + (2 * np.pi * i / self.config.num_sound_sources)
            self.state.sound_source_angles.append(angle)

    def set_torques(self, torques: np.ndarray):
        """
        Set torques to be applied (called by control loop).

        Args:
            torques: Torques for each joint
        """
        self.current_torques = np.clip(torques, -self.config.max_torque, self.config.max_torque)

    def get_state(self) -> PhysicsState:
        """
        Get current physics state (thread-safe).

        Returns:
            Current physics state
        """
        return self.state

    def _get_sound_source_positions(self) -> np.ndarray:
        """Get current sound source positions."""
        if not self.state.sound_source_angles:
            return np.array([]).reshape(0, 2)

        angles_array = np.array(self.state.sound_source_angles)
        positions = np.column_stack([
            self.config.circle_radius * np.cos(angles_array),
            self.config.circle_radius * np.sin(angles_array)
        ])
        return positions

    def _enforce_minimum_distance_constraint(self):
        """
        Enforce physical minimum distance constraint between end effector and sound sources.

        If the end effector is too close to any sound source, adjust the arm state
        to maintain the minimum distance by moving the end effector away.
        """
        # Get current end effector position
        _, end_effector_pos = self.arm_physics.forward_kinematics(
            self.state.arm_state.angles
        )

        # Get sound source positions
        sound_source_positions = self._get_sound_source_positions()

        if len(sound_source_positions) == 0:
            return

        min_distance = self.config.min_distance_to_source

        # Check distance to each sound source
        for source_pos in sound_source_positions:
            direction = end_effector_pos - source_pos
            distance = np.linalg.norm(direction)

            if distance < min_distance:
                # End effector is too close - move it away
                # Normalize direction and push to minimum distance
                if distance > 1e-6:
                    direction_normalized = direction / distance
                else:
                    # If exactly at source, choose arbitrary direction
                    direction_normalized = np.array([1.0, 0.0])

                # Target position at minimum distance from source
                target_pos = source_pos + direction_normalized * min_distance

                # Use inverse kinematics to adjust arm angles
                # Simple approach: adjust the last joint angle to point away
                # More sophisticated: solve inverse kinematics
                self._adjust_arm_to_position(target_pos)
                break  # Only adjust for first violation

    def _adjust_arm_to_position(self, target_pos: np.ndarray):
        """
        Adjust arm angles to move end effector to target position.

        Uses a simple gradient-based approach to minimize distance to target.
        """
        current_angles = self.state.arm_state.angles.copy()
        max_iterations = 50  # Increased iterations for better convergence
        step_size = 0.05  # Reduced step size for stability

        best_angles = current_angles.copy()
        best_error = float('inf')

        for iteration in range(max_iterations):
            _, current_pos = self.arm_physics.forward_kinematics(current_angles)
            error = target_pos - current_pos
            error_magnitude = np.linalg.norm(error)

            if error_magnitude < best_error:
                best_error = error_magnitude
                best_angles = current_angles.copy()

            if error_magnitude < 1e-4:  # Tighter tolerance
                break

            # Compute Jacobian numerically
            jacobian = np.zeros((2, len(current_angles)))
            epsilon = 1e-5

            for i in range(len(current_angles)):
                perturbed_angles = current_angles.copy()
                perturbed_angles[i] += epsilon
                _, perturbed_pos = self.arm_physics.forward_kinematics(perturbed_angles)
                jacobian[:, i] = (perturbed_pos - current_pos) / epsilon

            # Solve for angle update: error = J * delta_angles
            # Use pseudo-inverse: delta_angles = J^T * (J * J^T)^(-1) * error
            try:
                jjt = jacobian @ jacobian.T
                jjt_inv = np.linalg.inv(jjt + np.eye(2) * 1e-6)  # Regularization
                delta_angles = jacobian.T @ jjt_inv @ error
                current_angles += step_size * delta_angles
            except np.linalg.LinAlgError:
                # Fallback to simple gradient if inversion fails
                for i in range(len(current_angles)):
                    if np.linalg.norm(jacobian[:, i]) > 1e-6:
                        gradient_component = np.dot(error, jacobian[:, i]) / np.linalg.norm(jacobian[:, i])**2
                        current_angles[i] += step_size * gradient_component

            # Clamp angles to reasonable range
            current_angles = np.clip(current_angles, -np.pi, np.pi)

        # Use best angles found
        self.state.arm_state.angles = best_angles
        # Reset velocities when constraint is applied (represents collision/repulsion)
        self.state.arm_state.angular_velocities = np.zeros_like(self.state.arm_state.angular_velocities)

    def step(self):
        # Update sound sources
        if self.config.sound_source_angular_velocity != 0.0:
            for i in range(len(self.state.sound_source_angles)):
                self.state.sound_source_angles[i] += (
                    self.config.sound_source_angular_velocity * self.config.dt
                )

        # Step physics
        self.state.arm_state = self.arm_physics.step(
            self.state.arm_state,
            self.current_torques
        )

        # Enforce minimum distance constraint (physical limit)
        self._enforce_minimum_distance_constraint()
