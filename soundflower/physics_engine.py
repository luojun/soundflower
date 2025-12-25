"""Asynchronous physics engine for Sound Flower."""

import numpy as np
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
from .physics import ArmPhysics, ArmState, SoundPropagation
from .config import SoundFlowerConfig


@dataclass
class PhysicsState:
    """Complete physics state of the system."""
    arm_state: ArmState
    sound_source_angles: list
    simulation_time: float  # Total simulated time
    step_count: int


class PhysicsEngine:
    """Asynchronous physics engine that runs independently."""
    
    def __init__(self, config: SoundFlowerConfig):
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
            sound_source_angles=[],
            simulation_time=0.0,
            step_count=0
        )
        
        self._initialize_sound_sources()
        
        # Current torques (set by control loop)
        self.current_torques = np.zeros(config.num_links)
        
        # Running state
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Lock for thread-safe state access
        self._lock = asyncio.Lock()
        
        # Callback for state updates (optional)
        self._state_callback: Optional[Callable] = None
    
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
    
    async def _physics_loop(self):
        """Main physics simulation loop."""
        while self.running:
            async with self._lock:
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
                
                # Update simulation time
                self.state.simulation_time += self.config.dt
                self.state.step_count += 1
                
                # Call state callback if set
                if self._state_callback:
                    self._state_callback(self.state)
            
            # Yield to other tasks (physics runs as fast as possible)
            await asyncio.sleep(0)  # Yield control
    
    def start(self):
        """Start the physics engine."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._physics_loop())
    
    def stop(self):
        """Stop the physics engine."""
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def wait_for_stop(self):
        """Wait for physics engine to stop."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def reset(self):
        """Reset physics state."""
        async with self._lock:
            self.state = PhysicsState(
                arm_state=ArmState(
                    angles=np.zeros(self.config.num_links),
                    angular_velocities=np.zeros(self.config.num_links)
                ),
                sound_source_angles=[],
                simulation_time=0.0,
                step_count=0
            )
            self._initialize_sound_sources()
            self.current_torques = np.zeros(self.config.num_links)
    
    def set_state_callback(self, callback: Callable[[PhysicsState], None]):
        """
        Set callback for state updates.
        
        Args:
            callback: Function to call on each physics step
        """
        self._state_callback = callback

