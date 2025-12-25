"""World/Environment interface - represents the simulation state and logic."""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .physics_engine import PhysicsEngine, PhysicsState
from .config import SoundFlowerConfig
from .environment import Observation


@dataclass
class WorldState:
    """Complete state of the world."""
    physics_state: PhysicsState
    observation: Optional[Observation] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = None


class World:
    """
    World/Environment interface.
    
    Represents the simulation world state and provides methods to:
    - Query current state
    - Apply actions
    - Compute observations and rewards
    - Reset to initial state
    """
    
    def __init__(self, config: SoundFlowerConfig):
        """
        Initialize world.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.physics_engine = PhysicsEngine(config)
        
        # Track previous sound energy for delta computation
        self.previous_sound_energy = 0.0
        self.step_count = 0
    
    def get_state(self) -> WorldState:
        """
        Get current world state.
        
        Returns:
            Current world state
        """
        physics_state = self.physics_engine.get_state()
        observation = self._compute_observation(physics_state)
        
        # Compute reward
        reward = observation.sound_energy_delta
        
        # Compute info
        info = {
            'step_count': self.step_count,
            'total_sound_energy': observation.sound_energy,
            'end_effector_distance_to_source': self._compute_distance_to_nearest_source(
                observation.end_effector_pos, observation.sound_source_positions
            ),
            'simulation_time': physics_state.simulation_time
        }
        
        return WorldState(
            physics_state=physics_state,
            observation=observation,
            reward=reward,
            done=False,  # Infinite horizon
            info=info
        )
    
    def apply_action(self, action: np.ndarray):
        """
        Apply action to the world.
        
        Args:
            action: Action to apply (torques for each joint)
        """
        # Clamp action
        action = np.clip(action, -self.config.max_torque, self.config.max_torque)
        
        # Apply to physics engine
        self.physics_engine.set_torques(action)
        
        self.step_count += 1
    
    def _compute_observation(self, physics_state: PhysicsState) -> Observation:
        """Compute observation from physics state."""
        # Forward kinematics
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            physics_state.arm_state.angles
        )
        
        # Sound source positions
        sound_source_positions = []
        for angle in physics_state.sound_source_angles:
            pos = np.array([
                self.config.circle_radius * np.cos(angle),
                self.config.circle_radius * np.sin(angle)
            ])
            sound_source_positions.append(pos)
        
        # Compute sound energy
        sound_energy = 0.0
        if sound_source_positions:
            sound_energy = self.physics_engine.sound_propagation.compute_sound_energy_multiple_sources(
                source_positions=sound_source_positions,
                source_strengths=[self.config.sound_source_strength] * len(sound_source_positions),
                receiver_pos=end_effector_pos
            )
            sound_energy *= self.config.microphone_gain
        
        sound_energy_delta = sound_energy - self.previous_sound_energy
        self.previous_sound_energy = sound_energy
        
        return Observation(
            arm_angles=physics_state.arm_state.angles.copy(),
            arm_angular_velocities=physics_state.arm_state.angular_velocities.copy(),
            end_effector_pos=end_effector_pos.copy(),
            sound_energy=sound_energy,
            sound_energy_delta=sound_energy_delta,
            sound_source_positions=sound_source_positions
        )
    
    def _compute_distance_to_nearest_source(self, pos: np.ndarray, 
                                           source_positions: List[np.ndarray]) -> float:
        """Compute distance to nearest sound source."""
        if not source_positions:
            return float('inf')
        distances = [np.linalg.norm(pos - sp) for sp in source_positions]
        return min(distances)
    
    def get_render_data(self) -> Dict[str, Any]:
        """
        Get data for rendering.
        
        Returns:
            Dictionary with render data
        """
        physics_state = self.physics_engine.get_state()
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            physics_state.arm_state.angles
        )
        
        sound_source_positions = []
        for angle in physics_state.sound_source_angles:
            pos = np.array([
                self.config.circle_radius * np.cos(angle),
                self.config.circle_radius * np.sin(angle)
            ])
            sound_source_positions.append(pos)
        
        # Compute sound energy
        sound_energy = 0.0
        if sound_source_positions:
            sound_energy = self.physics_engine.sound_propagation.compute_sound_energy_multiple_sources(
                source_positions=sound_source_positions,
                source_strengths=[self.config.sound_source_strength] * len(sound_source_positions),
                receiver_pos=end_effector_pos
            )
            sound_energy *= self.config.microphone_gain
        
        return {
            'joint_positions': joint_positions,
            'end_effector_pos': end_effector_pos,
            'sound_source_positions': sound_source_positions,
            'circle_radius': self.config.circle_radius,
            'arm_state': physics_state.arm_state,
            'sound_energy': sound_energy,
            'simulation_time': physics_state.simulation_time,
            'step_count': physics_state.step_count
        }
    
    async def reset(self):
        """Reset world to initial state."""
        await self.physics_engine.reset()
        self.previous_sound_energy = 0.0
        self.step_count = 0
    
    def start_physics(self):
        """Start physics engine."""
        self.physics_engine.start()
    
    def stop_physics(self):
        """Stop physics engine."""
        self.physics_engine.stop()
    
    async def wait_for_physics_stop(self):
        """Wait for physics engine to stop."""
        await self.physics_engine.wait_for_stop()

