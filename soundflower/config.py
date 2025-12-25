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
    circle_radius: float = 1.0  # Radius of the circle
    
    # Microphone configuration
    num_microphones: int = 1  # Number of microphones at arm tip
    microphone_gain: float = 1.0  # Gain of microphones
    
    # Sound source configuration
    num_sound_sources: int = 1  # Number of sound sources
    sound_source_strength: float = 1.0  # Base strength of sound sources
    sound_attenuation_coeff: float = 1.0  # Coefficient for inverse square law (1/r^2)
    
    # Physics simulation
    dt: float = 0.01  # Time step for simulation
    max_torque: float = 10.0  # Maximum torque that can be applied at joints
    
    # Simulation frequencies
    control_frequency: float = 50.0  # Agent control frequency in Hz (10-100)
    visualization_fps: float = 60.0  # Visualization frame rate (10-100)
    
    # Headless mode
    headless: bool = False  # Run without visualization (faster than real-time)
    real_time_factor: float = 1.0  # Speed multiplier (1.0 = real-time, >1.0 = faster)
    
    # Sound source behavior
    sound_source_angular_velocity: float = 0.0  # Angular velocity for moving sound source (rad/s)
    sound_source_initial_angle: float = 0.0  # Initial angle of sound source (rad)
    
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

