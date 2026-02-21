"""Configuration parameters for the Sound Flower environment."""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SoundFlowerConfig:
    """Configuration for the Sound Flower environment."""

    # Arm configuration
    num_links: int = 2  # Number of arm links (1-3 DOF)
    link_lengths: List[float] = field(default_factory=lambda: [0.5, 0.5])  # Length of each link
    link_masses: List[float] = field(default_factory=lambda: [1.0, 1.0])  # Mass of each link
    joint_frictions: List[float] = field(default_factory=lambda: [0.1, 0.1])  # Friction coefficient at each joint

    # Circle/environment configuration
    circle_radius: float = 1.0  # Radius of the circle (meters)

    # Microphone configuration
    num_microphones: int = 1  # Number of microphones at arm tip
    microphone_area: float = 0.0001  # Effective area of microphone (m², default 1 cm²)
    microphone_gain: float = 1.0  # Gain of microphones

    # Sound source configuration
    num_sound_sources: int = 1  # Number of sound sources
    sound_source_strength: float = 1.0  # Base strength of sound sources
    sound_attenuation_coeff: float = 1.0  # Coefficient for inverse square law (1/r^2)
    min_distance_to_source: float = 0.2  # Minimum physical distance between microphone and sound source (meters)

    # Force
    max_torque: float = 2.0  # Maximum torque that can be applied at joints
    repulsion_coefficient: float = 10.0  # Spring coefficient for soft minimum distance constraint (reduced from 50.0 for stability)
    repulsion_damping: float = 5.0  # Damping coefficient for soft minimum distance constraint (reduced from 10.0 for stability)

    # Simulation frequencies
    dt: float = 0.01  # Time step for simulation
    control_frequency: float = 10.0  # Agent control frequency in Hz (0.1-100)
    logging_frequency: float = 1.0 # Logging freqency in Hz (0.01-100)
    animation_frequency: float = 10.0  # Animation frequncy in Hz (0.0-100)
    plotting_frequency: float = 0.1  # Plotting frequency in Hz (0.0-100)

    # Sound source behavior
    sound_source_angular_velocity: float = 0.0  # Angular velocity for moving sound source (rad/s)
    sound_source_initial_angle: float = 0.0  # Initial angle of sound source (rad)

    # Variability configuration (First batch)
    num_active_sources: int = 1  # Number of active sound sources (1-3), runtime adjustable
    orbit_radius_min: float = 0.8  # Minimum orbit radius (meters)
    orbit_radius_max: float = 1.2  # Maximum orbit radius (meters)
    orbit_radius_drift_rate: float = 0.01  # Rate of slow drift for orbit radius (per second)
    orbital_speed_min: float = -0.5  # Minimum orbital speed (rad/s, negative = counterclockwise)
    orbital_speed_max: float = 0.5  # Maximum orbital speed (rad/s)
    orbital_speed_drift_rate: float = 0.02  # Rate of slow drift for orbital speed (per second)

    # Reward normalization
    reward_normalization_factor: float = None  # Auto-computed if None

    # Performance: window for "average harvest over last N seconds" (seconds)
    performance_window_seconds: float = 10.0

    # Observation mode ("sensorimotor" or "full")
    observation_mode: str = "full"

    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure link_lengths and link_masses match num_links
        if len(self.link_lengths) != self.num_links:
            self.link_lengths = self.link_lengths[:self.num_links] + [0.5] * (self.num_links - len(self.link_lengths))
        if len(self.link_masses) != self.num_links:
            self.link_masses = self.link_masses[:self.num_links] + [1.0] * (self.num_links - len(self.link_masses))
        if len(self.joint_frictions) != self.num_links:
            self.joint_frictions = self.joint_frictions[:self.num_links] + [0.1] * (self.num_links - len(self.joint_frictions))

        # Ensure all values are positive
        self.link_lengths = [max(0.01, l) for l in self.link_lengths]
        self.link_masses = [max(0.01, m) for m in self.link_masses]
        self.joint_frictions = [max(0.0, f) for f in self.joint_frictions]

        # Compute reward normalization factor if not set
        if self.reward_normalization_factor is None:
            # Maximum intensity at minimum distance
            max_intensity = self.sound_source_strength / (4 * np.pi * self.min_distance_to_source ** 2)
            # Maximum energy per step (perfect orientation, minimum distance)
            max_energy_per_step = max_intensity * self.microphone_area * self.dt * 1.0
            # Maximum possible delta (moving from 0 to max in one step)
            self.reward_normalization_factor = max_energy_per_step


def create_default_config(sound_source_angular_velocity: float = 0.2) -> SoundFlowerConfig:
    """
    Create default configuration for experiments.

    Args:
        sound_source_angular_velocity: Angular velocity of sound source (rad/s)

    Returns:
        Default SoundFlowerConfig
    """
    return SoundFlowerConfig(
        num_links=2,
        link_lengths=[0.6, 0.4],
        link_masses=[6.0, 6.0],
        joint_frictions=[1.0, 1.0],
        circle_radius=1.0,
        num_microphones=1,
        microphone_gain=1.0,
        num_sound_sources=3,  # Initialize 3 sources to support variability (1-3 active)
        sound_source_strength=2.0,
        sound_attenuation_coeff=1.0,
        min_distance_to_source=0.2,
        dt=0.01,
        max_torque=2.0,
        sound_source_angular_velocity=sound_source_angular_velocity,
        sound_source_initial_angle=np.pi / 4,  # Start at 45 degrees
        num_active_sources=1,  # Start with 1 active source
        orbit_radius_min=0.8,
        orbit_radius_max=1.2,
        orbital_speed_min=-0.5,
        orbital_speed_max=0.5
    )

