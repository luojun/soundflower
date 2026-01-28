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

        # Variability state (runtime values that drift over time)
        self.current_orbit_radius = (config.orbit_radius_min + config.orbit_radius_max) / 2.0
        self.current_orbital_speed = (config.orbital_speed_min + config.orbital_speed_max) / 2.0
        # Drift direction: +1 or -1, changes when hitting bounds
        self.orbit_radius_drift_dir = 1.0
        self.orbital_speed_drift_dir = 1.0

        # Running state
        self.running = False

    def set_num_active_sources(self, num: int):
        """Set number of active sound sources (1-3)."""
        self.config.num_active_sources = max(1, min(3, int(num)))

    def set_orbit_radius_range(self, min_radius: float, max_radius: float):
        """Set orbit radius variability range."""
        self.config.orbit_radius_min = max(0.1, min_radius)
        self.config.orbit_radius_max = max(self.config.orbit_radius_min, max_radius)
        # Clamp current radius to new range
        self.current_orbit_radius = np.clip(
            self.current_orbit_radius,
            self.config.orbit_radius_min,
            self.config.orbit_radius_max
        )

    def set_orbital_speed_range(self, min_speed: float, max_speed: float):
        """Set orbital speed variability range."""
        self.config.orbital_speed_min = min_speed
        self.config.orbital_speed_max = max(max_speed, min_speed)
        # Clamp current speed to new range
        self.current_orbital_speed = np.clip(
            self.current_orbital_speed,
            self.config.orbital_speed_min,
            self.config.orbital_speed_max
        )

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
        if not np.all(np.isfinite(torques)):
            raise ValueError(f"Non-finite torques applied: {torques}")
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
        # Only use active sources (first num_active_sources)
        num_active = min(self.config.num_active_sources, len(self.state.sound_source_angles))
        if num_active == 0:
            return np.array([]).reshape(0, 2)

        angles_array = np.array(self.state.sound_source_angles[:num_active])
        positions = np.column_stack([
            self.current_orbit_radius * np.cos(angles_array),
            self.current_orbit_radius * np.sin(angles_array)
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

                # Scale repulsion force using a lever-arm estimate so units are consistent.
                # This keeps repulsion modest relative to actuator limits without mixing force/torque.
                total_link_length = max(1e-6, float(np.sum(self.config.link_lengths)))
                max_repulsion_force = 0.3 * (self.config.max_torque / total_link_length)

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
        # Update variability parameters with slow drift
        # Orbit radius drift
        radius_drift = self.config.orbit_radius_drift_rate * self.config.dt * self.orbit_radius_drift_dir
        self.current_orbit_radius += radius_drift
        # Bounce off bounds
        if self.current_orbit_radius >= self.config.orbit_radius_max:
            self.current_orbit_radius = self.config.orbit_radius_max
            self.orbit_radius_drift_dir = -1.0
        elif self.current_orbit_radius <= self.config.orbit_radius_min:
            self.current_orbit_radius = self.config.orbit_radius_min
            self.orbit_radius_drift_dir = 1.0

        # Orbital speed drift
        speed_drift = self.config.orbital_speed_drift_rate * self.config.dt * self.orbital_speed_drift_dir
        self.current_orbital_speed += speed_drift
        # Bounce off bounds
        if self.current_orbital_speed >= self.config.orbital_speed_max:
            self.current_orbital_speed = self.config.orbital_speed_max
            self.orbital_speed_drift_dir = -1.0
        elif self.current_orbital_speed <= self.config.orbital_speed_min:
            self.current_orbital_speed = self.config.orbital_speed_min
            self.orbital_speed_drift_dir = 1.0

        # Update sound sources with current orbital speed
        if abs(self.current_orbital_speed) > 1e-6:
            num_active = min(self.config.num_active_sources, len(self.state.sound_source_angles))
            for i in range(num_active):
                self.state.sound_source_angles[i] += (
                    self.current_orbital_speed * self.config.dt
                )

        # Enforce minimum distance constraint using soft repulsion force
        # This adds repulsion torques to current_torques if constraint is violated
        self._enforce_minimum_distance_constraint()

        # Step physics with torques (including any repulsion torques)
        self.state.arm_state = self.arm_physics.step(
            self.state.arm_state,
            self.current_torques
        )
        if (not np.all(np.isfinite(self.state.arm_state.angles)) or
                not np.all(np.isfinite(self.state.arm_state.angular_velocities))):
            raise RuntimeError(
                "Non-finite arm state after physics step "
                f"(angles={self.state.arm_state.angles}, "
                f"angular_velocities={self.state.arm_state.angular_velocities})"
            )