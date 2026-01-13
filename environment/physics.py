"""Physics simulation for the robotic arm and sound propagation."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ArmState:
    """State of the robotic arm."""
    angles: np.ndarray  # Joint angles (rad)
    angular_velocities: np.ndarray  # Joint angular velocities (rad/s)


class ArmPhysics:
    """Physics simulation for a multi-link robotic arm."""

    def __init__(self, link_lengths: List[float], link_masses: List[float],
                 joint_frictions: List[float], dt: float = 0.01):
        """
        Initialize arm physics.

        Args:
            link_lengths: Length of each link
            link_masses: Mass of each link (concentrated at midpoint)
            joint_frictions: Friction coefficient at each joint
            dt: Time step for simulation
        """
        self.link_lengths = np.array(link_lengths)
        self.link_masses = np.array(link_masses)
        self.joint_frictions = np.array(joint_frictions)
        self.dt = dt
        self.num_links = len(link_lengths)

        # Calculate moment of inertia for each link (simplified: point mass at midpoint)
        # I = m * r^2 where r is distance from joint to mass center
        self.moments_of_inertia = self.link_masses * (self.link_lengths / 2) ** 2

    def forward_kinematics(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics: joint positions and end effector position.

        Args:
            angles: Joint angles (rad)

        Returns:
            joint_positions: Array of (x, y) positions for each joint
            end_effector_pos: (x, y) position of end effector
        """
        joint_positions = np.zeros((self.num_links + 1, 2))
        cumulative_angle = 0.0

        for i in range(self.num_links):
            cumulative_angle += angles[i]
            joint_positions[i + 1] = joint_positions[i] + self.link_lengths[i] * np.array([
                np.cos(cumulative_angle),
                np.sin(cumulative_angle)
            ])

        return joint_positions, joint_positions[-1]

    def compute_dynamics(self, state: ArmState, torques: np.ndarray) -> np.ndarray:
        """
        Compute angular accelerations given current state and applied torques.

        Args:
            state: Current arm state
            torques: Torques applied at each joint

        Returns:
            angular_accelerations: Angular accelerations at each joint
        """
        # Simplified dynamics: each joint is independent with friction
        # τ = I * α + friction
        # α = (τ - friction) / I

        # Friction opposes motion
        friction_torques = -self.joint_frictions * np.sign(state.angular_velocities) * np.abs(state.angular_velocities)

        # Net torque at each joint
        net_torques = torques + friction_torques

        # Angular acceleration
        angular_accelerations = net_torques / (self.moments_of_inertia + 1e-6)  # Small epsilon to avoid division by zero

        return angular_accelerations

    def step(self, state: ArmState, torques: np.ndarray) -> ArmState:
        """
        Simulate one time step.

        Args:
            state: Current arm state
            torques: Torques applied at each joint (already clipped at physics boundary)

        Returns:
            new_state: Updated arm state
        """

        # Compute accelerations
        angular_accelerations = self.compute_dynamics(state, torques)

        # Update velocities (Euler integration)
        new_angular_velocities = state.angular_velocities + angular_accelerations * self.dt

        # Update angles
        new_angles = state.angles + new_angular_velocities * self.dt

        return ArmState(angles=new_angles, angular_velocities=new_angular_velocities)


class SoundPropagation:
    """Sound propagation simulation using inverse square law."""

    def __init__(self, attenuation_coeff: float = 1.0):
        """
        Initialize sound propagation.

        Args:
            attenuation_coeff: Coefficient for inverse square law
        """
        self.attenuation_coeff = attenuation_coeff

    def compute_sound_intensity(self, source_pos: np.ndarray, source_strength: float,
                                receiver_pos: np.ndarray) -> float:
        """
        Compute sound intensity at receiver position using inverse square law.

        Args:
            source_pos: (x, y) position of sound source
            source_strength: Base strength of sound source
            receiver_pos: (x, y) position of receiver (microphone)

        Returns:
            sound_intensity: Sound intensity at receiver (power per unit area)
        """
        # Distance between source and receiver
        distance = np.linalg.norm(receiver_pos - source_pos)

        # Avoid division by zero
        if distance < 1e-6:
            distance = 1e-6

        # Inverse square law: intensity = P / (4πr²) where P is power
        sound_intensity = self.attenuation_coeff * source_strength / (4 * np.pi * distance ** 2)

        return sound_intensity

    def compute_sound_intensity_multiple_sources(self, source_positions: List[np.ndarray],
                                                 source_strengths: List[float],
                                                 receiver_pos: np.ndarray) -> float:
        """
        Compute total sound intensity from multiple sources (additive).

        Args:
            source_positions: List of (x, y) positions of sound sources
            source_strengths: List of base strengths of sound sources
            receiver_pos: (x, y) position of receiver

        Returns:
            total_sound_intensity: Total sound intensity at receiver (power per unit area)
        """
        total_intensity = 0.0
        for pos, strength in zip(source_positions, source_strengths):
            total_intensity += self.compute_sound_intensity(pos, strength, receiver_pos)
        return total_intensity

