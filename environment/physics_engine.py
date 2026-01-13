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

        # Note: Initial arm state constraint enforcement happens after first step
        # when repulsion forces can be applied properly

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

    def _compute_position_jacobian(self, angles: np.ndarray) -> np.ndarray:
        """
        Compute position Jacobian numerically for end effector.

        Args:
            angles: Current joint angles

        Returns:
            jacobian: (2, num_joints) Jacobian matrix
        """
        _, current_pos = self.arm_physics.forward_kinematics(angles)
        jacobian = np.zeros((2, len(angles)))
        epsilon = 1e-5

        for i in range(len(angles)):
            perturbed_angles = angles.copy()
            perturbed_angles[i] += epsilon
            _, perturbed_pos = self.arm_physics.forward_kinematics(perturbed_angles)
            jacobian[:, i] = (perturbed_pos - current_pos) / epsilon

        return jacobian

    def _enforce_minimum_distance_constraint(self):
        """
        Enforce physical minimum distance constraint using soft spring-damper repulsion force.

        When the end effector is too close to a sound source, apply a repulsion force
        proportional to the violation (spring term) and velocity toward the source (damping term).
        The force is converted to joint torques using Jacobian transpose and added to current_torques.
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
        k_repulsion = self.config.repulsion_coefficient
        c_damping = self.config.repulsion_damping

        # Compute position Jacobian for converting forces to torques
        position_jacobian = self._compute_position_jacobian(self.state.arm_state.angles)

        # Check distance to each sound source and apply repulsion force if needed
        for source_pos in sound_source_positions:
            direction = end_effector_pos - source_pos
            distance = np.linalg.norm(direction)

            if distance < min_distance:
                # Compute direction from source to end effector (normalized)
                if distance > 1e-6:
                    direction_to_source_normalized = -direction / distance
                else:
                    # If exactly at source, choose arbitrary direction
                    direction_to_source_normalized = np.array([1.0, 0.0])

                # Compute end effector velocity from joint velocities using position Jacobian
                # v_ee = J * q_dot
                end_effector_velocity = position_jacobian @ self.state.arm_state.angular_velocities

                # Project velocity onto direction toward source
                # Positive velocity_toward_source means moving toward source (bad)
                velocity_toward_source = np.dot(end_effector_velocity, direction_to_source_normalized)

                # Scale repulsion force to be proportional to max_torque
                # This prevents repulsion from overwhelming agent control
                # Use a scaling factor so repulsion is a fraction of max_torque
                max_repulsion_force = 0.3 * self.config.max_torque  # 30% of max_torque

                # Normalize violation by min_distance to get relative violation [0, 1]
                violation = min_distance - distance
                relative_violation = violation / min_distance

                # Spring term: proportional to violation, scaled to max_repulsion_force
                spring_force = k_repulsion * relative_violation * max_repulsion_force

                # Damping term: oppose velocity toward source
                # Scale damping to be proportional to max_repulsion_force as well
                damping_force = c_damping * velocity_toward_source * (max_repulsion_force / k_repulsion)

                repulsion_force_magnitude = spring_force - damping_force

                # Ensure force is repulsive (positive magnitude) but not too large
                repulsion_force_magnitude = np.clip(repulsion_force_magnitude, 0.0, max_repulsion_force)

                # Convert force to joint torques using Jacobian transpose
                # τ = J^T * F
                repulsion_force_vector = repulsion_force_magnitude * direction_to_source_normalized
                repulsion_torques = position_jacobian.T @ repulsion_force_vector

                # Clip repulsion torques to prevent excessive values
                # Limit to a fraction of max_torque to ensure agent control dominates
                max_repulsion_torque = 0.3 * self.config.max_torque
                repulsion_torques = np.clip(repulsion_torques, -max_repulsion_torque, max_repulsion_torque)

                # Add repulsion torques to current torques
                self.current_torques += repulsion_torques

                # Clip total torques to ensure they don't exceed limits
                self.current_torques = np.clip(self.current_torques, -self.config.max_torque, self.config.max_torque)

                break  # Only apply repulsion for first violation

    def step(self):
        # Update sound sources
        if self.config.sound_source_angular_velocity != 0.0:
            for i in range(len(self.state.sound_source_angles)):
                self.state.sound_source_angles[i] += (
                    self.config.sound_source_angular_velocity * self.config.dt
                )

        # Enforce minimum distance constraint using soft repulsion force
        # This adds repulsion torques to current_torques if constraint is violated
        self._enforce_minimum_distance_constraint()

        # Step physics with torques (including any repulsion torques)
        self.state.arm_state = self.arm_physics.step(
            self.state.arm_state,
            self.current_torques
        )
