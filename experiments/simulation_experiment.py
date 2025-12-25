"""Experiment runner using the new decoupled simulation architecture."""

import asyncio
import numpy as np
from typing import Dict, Optional, Any, Callable
from soundflower.physics_engine import PhysicsEngine
from soundflower.simulation_loop import ControlLoop, VisualizationLoop, SimulationCoordinator
from soundflower.config import SoundFlowerConfig
from soundflower.environment import Observation


class SimulationExperiment:
    """Experiment runner using decoupled physics, control, and visualization."""
    
    def __init__(self, config: SoundFlowerConfig, agent):
        """
        Initialize simulation experiment.
        
        Args:
            config: Configuration object
            agent: Agent with async select_action method
        """
        self.config = config
        self.agent = agent
        
        # Create physics engine
        self.physics_engine = PhysicsEngine(config)
        
        # Create control loop
        self.control_loop = ControlLoop(
            physics_engine=self.physics_engine,
            agent=agent,
            control_frequency=config.control_frequency
        )
        
        # Visualization loop (created on demand)
        self.visualization_loop: Optional[VisualizationLoop] = None
        self.visualizer = None
        
        # Coordinator
        self.coordinator: Optional[SimulationCoordinator] = None
        
        # Statistics
        self.stats = {
            'total_reward': 0.0,
            'steps': 0,
            'simulation_time': 0.0
        }
    
    def set_visualizer(self, render_callback: Callable[[Dict[str, Any]], None], fps: float = 60.0):
        """
        Set up visualization.
        
        Args:
            render_callback: Function to call with render data
            fps: Frame rate
        """
        self.visualization_loop = VisualizationLoop(
            physics_engine=self.physics_engine,
            render_callback=render_callback,
            fps=fps
        )
    
    async def run_headless(self, duration: float, 
                          observation_callback: Optional[Callable[[Observation, np.ndarray], None]] = None):
        """
        Run simulation in headless mode (no visualization, faster than real-time).
        
        Args:
            duration: Duration in simulated seconds
            observation_callback: Optional callback for observations
        """
        if observation_callback:
            self.control_loop.set_observation_callback(observation_callback)
        
        # Create coordinator without visualization
        self.coordinator = SimulationCoordinator(
            physics_engine=self.physics_engine,
            control_loop=self.control_loop,
            visualization_loop=None
        )
        
        # Reset
        await self.coordinator.reset()
        
        # Run for specified duration
        await self.coordinator.run(duration=duration)
        
        # Get final state
        final_state = self.physics_engine.get_state()
        self.stats['simulation_time'] = final_state.simulation_time
        self.stats['steps'] = final_state.step_count
    
    async def run_with_visualization(self, duration: Optional[float] = None):
        """
        Run simulation with visualization.
        
        Args:
            duration: Duration in seconds (None = run indefinitely)
        """
        if not self.visualization_loop:
            raise ValueError("Visualization not set. Call set_visualizer() first.")
        
        # Create coordinator with visualization
        self.coordinator = SimulationCoordinator(
            physics_engine=self.physics_engine,
            control_loop=self.control_loop,
            visualization_loop=self.visualization_loop
        )
        
        # Reset
        await self.coordinator.reset()
        
        # Run
        await self.coordinator.run(duration=duration)
    
    async def run_for_steps(self, max_steps: int, headless: bool = True,
                           observation_callback: Optional[Callable] = None):
        """
        Run simulation for a specific number of control steps.
        
        Args:
            max_steps: Maximum number of control steps
            headless: Whether to run without visualization
            observation_callback: Optional callback for observations
        """
        steps = 0
        
        def step_callback(observation: Observation, action: np.ndarray):
            nonlocal steps
            steps += 1
            if observation_callback:
                observation_callback(observation, action)
        
        if observation_callback:
            self.control_loop.set_observation_callback(step_callback)
        
        # Create coordinator
        self.coordinator = SimulationCoordinator(
            physics_engine=self.physics_engine,
            control_loop=self.control_loop,
            visualization_loop=None if headless else self.visualization_loop
        )
        
        # Reset
        await self.coordinator.reset()
        
        # Run until max_steps reached
        async def run_until_steps():
            while steps < max_steps:
                await asyncio.sleep(0.1)
        
        await asyncio.gather(
            self.coordinator.run(duration=None),
            run_until_steps()
        )
        
        # Stop coordinator
        if self.visualization_loop:
            self.visualization_loop.stop()
            await self.visualization_loop.wait_for_stop()
        
        self.control_loop.stop()
        await self.control_loop.wait_for_stop()
        
        self.physics_engine.stop()
        await self.physics_engine.wait_for_stop()
        
        final_state = self.physics_engine.get_state()
        self.stats['steps'] = steps
        self.stats['simulation_time'] = final_state.simulation_time
    
    def get_state(self):
        """Get current physics state."""
        return self.physics_engine.get_state()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get experiment statistics."""
        return self.stats.copy()

