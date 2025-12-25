"""Simulation loop coordinating physics, control, and visualization."""

import numpy as np
import asyncio
from typing import Optional, Callable, Dict, Any
from .physics_engine import PhysicsEngine, PhysicsState
from .config import SoundFlowerConfig
from .environment import Observation


class ControlLoop:
    """Agent control loop running at configurable frequency."""
    
    def __init__(self, physics_engine: PhysicsEngine, agent, 
                 control_frequency: float = 50.0):
        """
        Initialize control loop.
        
        Args:
            physics_engine: Physics engine instance
            agent: Agent with async select_action method
            control_frequency: Control frequency in Hz (10-100)
        """
        self.physics_engine = physics_engine
        self.agent = agent
        self.control_frequency = max(10.0, min(100.0, control_frequency))
        self.control_period = 1.0 / self.control_frequency
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Track previous sound energy for delta computation
        self.previous_sound_energy = 0.0
        
        # Observation callback (for environment interface)
        self._observation_callback: Optional[Callable] = None
    
    def _compute_observation(self, state: PhysicsState) -> Observation:
        """Compute observation from physics state."""
        # Forward kinematics
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            state.arm_state.angles
        )
        
        # Sound source positions
        sound_source_positions = []
        for angle in state.sound_source_angles:
            pos = np.array([
                self.physics_engine.config.circle_radius * np.cos(angle),
                self.physics_engine.config.circle_radius * np.sin(angle)
            ])
            sound_source_positions.append(pos)
        
        # Compute sound energy
        sound_energy = 0.0
        if sound_source_positions:
            sound_energy = self.physics_engine.sound_propagation.compute_sound_energy_multiple_sources(
                source_positions=sound_source_positions,
                source_strengths=[self.physics_engine.config.sound_source_strength] * len(sound_source_positions),
                receiver_pos=end_effector_pos
            )
            sound_energy *= self.physics_engine.config.microphone_gain
        
        sound_energy_delta = sound_energy - self.previous_sound_energy
        self.previous_sound_energy = sound_energy
        
        return Observation(
            arm_angles=state.arm_state.angles.copy(),
            arm_angular_velocities=state.arm_state.angular_velocities.copy(),
            end_effector_pos=end_effector_pos.copy(),
            sound_energy=sound_energy,
            sound_energy_delta=sound_energy_delta,
            sound_source_positions=sound_source_positions
        )
    
    async def _control_loop(self):
        """Main control loop."""
        while self.running:
            # Get current physics state
            state = self.physics_engine.get_state()
            
            # Compute observation
            observation = self._compute_observation(state)
            
            # Agent selects action
            action = await self.agent.select_action(observation)
            
            # Apply torques to physics engine
            self.physics_engine.set_torques(action)
            
            # Call observation callback if set
            if self._observation_callback:
                self._observation_callback(observation, action)
            
            # Wait for next control cycle
            await asyncio.sleep(self.control_period)
    
    def start(self):
        """Start control loop."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._control_loop())
    
    def stop(self):
        """Stop control loop."""
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def wait_for_stop(self):
        """Wait for control loop to stop."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    def set_observation_callback(self, callback: Callable):
        """Set callback for observations."""
        self._observation_callback = callback
    
    def reset(self):
        """Reset control loop state."""
        self.previous_sound_energy = 0.0


class VisualizationLoop:
    """Visualization loop running at configurable frame rate."""
    
    def __init__(self, physics_engine: PhysicsEngine, 
                 render_callback: Callable[[Dict[str, Any]], None],
                 fps: float = 60.0):
        """
        Initialize visualization loop.
        
        Args:
            physics_engine: Physics engine instance
            render_callback: Function to call with render data
            fps: Frame rate (10-100)
        """
        self.physics_engine = physics_engine
        self.render_callback = render_callback
        self.fps = max(10.0, min(100.0, fps))
        self.frame_period = 1.0 / self.fps
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def _get_render_data(self, state: PhysicsState) -> Dict[str, Any]:
        """Get render data from physics state."""
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            state.arm_state.angles
        )
        
        sound_source_positions = []
        for angle in state.sound_source_angles:
            pos = np.array([
                self.physics_engine.config.circle_radius * np.cos(angle),
                self.physics_engine.config.circle_radius * np.sin(angle)
            ])
            sound_source_positions.append(pos)
        
        # Compute sound energy
        sound_energy = 0.0
        if sound_source_positions:
            sound_energy = self.physics_engine.sound_propagation.compute_sound_energy_multiple_sources(
                source_positions=sound_source_positions,
                source_strengths=[self.physics_engine.config.sound_source_strength] * len(sound_source_positions),
                receiver_pos=end_effector_pos
            )
            sound_energy *= self.physics_engine.config.microphone_gain
        
        return {
            'joint_positions': joint_positions,
            'end_effector_pos': end_effector_pos,
            'sound_source_positions': sound_source_positions,
            'circle_radius': self.physics_engine.config.circle_radius,
            'arm_state': state.arm_state,
            'sound_energy': sound_energy,
            'simulation_time': state.simulation_time,
            'step_count': state.step_count
        }
    
    async def _visualization_loop(self):
        """Main visualization loop."""
        while self.running:
            # Get current physics state
            state = self.physics_engine.get_state()
            
            # Get render data
            render_data = self._get_render_data(state)
            
            # Call render callback
            self.render_callback(render_data)
            
            # Wait for next frame
            await asyncio.sleep(self.frame_period)
    
    def start(self):
        """Start visualization loop."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._visualization_loop())
    
    def stop(self):
        """Stop visualization loop."""
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def wait_for_stop(self):
        """Wait for visualization loop to stop."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class SimulationCoordinator:
    """Coordinates physics, control, and visualization loops."""
    
    def __init__(self, physics_engine: PhysicsEngine, 
                 control_loop: Optional[ControlLoop] = None,
                 visualization_loop: Optional[VisualizationLoop] = None):
        """
        Initialize simulation coordinator.
        
        Args:
            physics_engine: Physics engine
            control_loop: Optional control loop
            visualization_loop: Optional visualization loop
        """
        self.physics_engine = physics_engine
        self.control_loop = control_loop
        self.visualization_loop = visualization_loop
    
    async def run(self, duration: Optional[float] = None):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Duration in seconds (None = run indefinitely)
        """
        # Start all loops
        self.physics_engine.start()
        if self.control_loop:
            self.control_loop.start()
        if self.visualization_loop:
            self.visualization_loop.start()
        
        try:
            if duration is not None:
                await asyncio.sleep(duration)
            else:
                # Run indefinitely
                while True:
                    await asyncio.sleep(1.0)
        finally:
            # Stop all loops
            if self.visualization_loop:
                self.visualization_loop.stop()
                await self.visualization_loop.wait_for_stop()
            
            if self.control_loop:
                self.control_loop.stop()
                await self.control_loop.wait_for_stop()
            
            self.physics_engine.stop()
            await self.physics_engine.wait_for_stop()
    
    async def reset(self):
        """Reset all components."""
        await self.physics_engine.reset()
        if self.control_loop:
            self.control_loop.reset()

