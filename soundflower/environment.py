"""Asynchronous RL environment for Sound Flower."""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .physics import ArmPhysics, ArmState, SoundPropagation
from .config import SoundFlowerConfig


@dataclass
class Observation:
    """Observation from the environment."""
    arm_angles: np.ndarray  # Current joint angles
    arm_angular_velocities: np.ndarray  # Current joint angular velocities
    end_effector_pos: np.ndarray  # End effector position (x, y)
    sound_energy: float  # Current sound energy at microphone
    sound_energy_delta: float  # Change in sound energy (for reward)
    sound_source_positions: List[np.ndarray]  # Positions of sound sources


class SoundFlowerEnvironment:
    """Asynchronous RL environment for sound source tracking with robotic arm."""
    
    def __init__(self, config: Optional[SoundFlowerConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or SoundFlowerConfig()
        
        # Initialize physics
        self.arm_physics = ArmPhysics(
            link_lengths=self.config.link_lengths,
            link_masses=self.config.link_masses,
            joint_frictions=self.config.joint_frictions,
            dt=self.config.dt
        )
        
        self.sound_propagation = SoundPropagation(
            attenuation_coeff=self.config.sound_attenuation_coeff
        )
        
        # Initialize arm state
        self.arm_state = ArmState(
            angles=np.zeros(self.config.num_links),
            angular_velocities=np.zeros(self.config.num_links)
        )
        
        # Initialize sound sources
        self.sound_source_angles = []
        self.sound_source_strengths = []
        self._initialize_sound_sources()
        
        # Track previous sound energy for delta computation
        self.previous_sound_energy = 0.0
        
        # Environment state
        self.running = False
        self.step_count = 0
        
        # Event loop for async operations
        self._loop = None
    
    def _initialize_sound_sources(self):
        """Initialize sound sources along the circle."""
        self.sound_source_angles = []
        self.sound_source_strengths = []
        
        for i in range(self.config.num_sound_sources):
            # Distribute sources evenly around circle
            angle = self.config.sound_source_initial_angle + (2 * np.pi * i / self.config.num_sound_sources)
            self.sound_source_angles.append(angle)
            self.sound_source_strengths.append(self.config.sound_source_strength)
    
    def _get_sound_source_positions(self) -> List[np.ndarray]:
        """Get current positions of sound sources."""
        positions = []
        for angle in self.sound_source_angles:
            pos = np.array([
                self.config.circle_radius * np.cos(angle),
                self.config.circle_radius * np.sin(angle)
            ])
            positions.append(pos)
        return positions
    
    def _update_sound_sources(self):
        """Update sound source positions (if they're moving)."""
        if self.config.sound_source_angular_velocity != 0.0:
            for i in range(len(self.sound_source_angles)):
                self.sound_source_angles[i] += self.config.sound_source_angular_velocity * self.config.dt
    
    def _compute_sound_energy(self) -> float:
        """Compute current sound energy at microphone."""
        _, end_effector_pos = self.arm_physics.forward_kinematics(self.arm_state.angles)
        source_positions = self._get_sound_source_positions()
        
        total_energy = self.sound_propagation.compute_sound_energy_multiple_sources(
            source_positions=source_positions,
            source_strengths=self.sound_source_strengths,
            receiver_pos=end_effector_pos
        )
        
        # Apply microphone gain
        return total_energy * self.config.microphone_gain
    
    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
        """
        # Reset arm state
        self.arm_state = ArmState(
            angles=np.zeros(self.config.num_links),
            angular_velocities=np.zeros(self.config.num_links)
        )
        
        # Reset sound sources
        self._initialize_sound_sources()
        
        # Reset tracking variables
        self.previous_sound_energy = self._compute_sound_energy()
        self.step_count = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> Observation:
        """Get current observation."""
        _, end_effector_pos = self.arm_physics.forward_kinematics(self.arm_state.angles)
        sound_energy = self._compute_sound_energy()
        sound_energy_delta = sound_energy - self.previous_sound_energy
        
        return Observation(
            arm_angles=self.arm_state.angles.copy(),
            arm_angular_velocities=self.arm_state.angular_velocities.copy(),
            end_effector_pos=end_effector_pos.copy(),
            sound_energy=sound_energy,
            sound_energy_delta=sound_energy_delta,
            sound_source_positions=self._get_sound_source_positions()
        )
    
    async def step(self, action: np.ndarray) -> Tuple[Observation, float, bool, Dict]:
        """
        Asynchronously step the environment.
        
        Args:
            action: Torques to apply at each joint (shape: [num_links])
            
        Returns:
            observation: New observation
            reward: Reward signal (based on sound energy delta)
            done: Whether episode is done
            info: Additional information
        """
        # Clamp action to valid torque range
        action = np.clip(action, -self.config.max_torque, self.config.max_torque)
        
        # Update sound sources
        self._update_sound_sources()
        
        # Step physics
        self.arm_state = self.arm_physics.step(self.arm_state, action)
        
        # Compute observation
        observation = self._get_observation()
        
        # Reward is the change in sound energy (delta)
        reward = observation.sound_energy_delta
        
        # Update previous sound energy
        self.previous_sound_energy = observation.sound_energy
        
        # Episode never ends (infinite horizon)
        done = False
        
        self.step_count += 1
        
        info = {
            'step_count': self.step_count,
            'total_sound_energy': observation.sound_energy,
            'end_effector_distance_to_source': self._compute_distance_to_nearest_source(observation.end_effector_pos)
        }
        
        # Small async delay to simulate real-world timing
        await asyncio.sleep(0.001)
        
        return observation, reward, done, info
    
    def _compute_distance_to_nearest_source(self, pos: np.ndarray) -> float:
        """Compute distance to nearest sound source."""
        source_positions = self._get_sound_source_positions()
        distances = [np.linalg.norm(pos - sp) for sp in source_positions]
        return min(distances) if distances else float('inf')
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space."""
        # angles + angular_velocities + end_effector_pos + sound_energy + sound_energy_delta
        return self.config.num_links * 2 + 2 + 1 + 1
    
    def get_action_space_size(self) -> int:
        """Get size of action space."""
        return self.config.num_links
    
    def render(self) -> Dict[str, Any]:
        """
        Get current state for visualization.
        
        Returns:
            Dictionary with state information for rendering
        """
        joint_positions, end_effector_pos = self.arm_physics.forward_kinematics(self.arm_state.angles)
        source_positions = self._get_sound_source_positions()
        
        return {
            'joint_positions': joint_positions,
            'end_effector_pos': end_effector_pos,
            'sound_source_positions': source_positions,
            'circle_radius': self.config.circle_radius,
            'arm_state': self.arm_state,
            'sound_energy': self._compute_sound_energy()
        }

