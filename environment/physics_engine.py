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
