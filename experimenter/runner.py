"""Runner interface - orchestrates simulation execution."""

import asyncio
from typing import Optional, Callable, Dict, Any
from environment import Environment


class Runner:
    """
    Runner interface for orchestrating simulation.
    
    Coordinates between:
    - Environment (simulation state)
    - Agent (decision making)
    - Animator (visualization, optional)
    
    Runs at configurable frequencies:
    - Physics: runs as fast as possible with configurable time step
    - Control: runs at control_frequency (10-100 Hz)
    - Visualization: runs at visualization_fps (10-100 fps, optional)
    """
    
    def __init__(self, environment: Environment, agent, config):
        """
        Initialize runner.
        
        Args:
            environment: Environment instance
            agent: Agent with async select_action method
            config: Configuration object
        """
        self.environment = environment
        self.agent = agent
        self.config = config
        
        # Control loop state
        self.control_frequency = config.control_frequency
        self.control_period = 1.0 / self.control_frequency
        
        self.running = False
        self._control_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._observation_callback: Optional[Callable] = None
        self._step_callback: Optional[Callable] = None
    
    async def _control_loop(self):
        """Main control loop running at control_frequency."""
        while self.running:
            # Get current environment state
            world_state = self.environment.get_state()
            
            if world_state.observation is None:
                await asyncio.sleep(self.control_period)
                continue
            
            # Agent selects action
            action = await self.agent.select_action(world_state.observation)
            
            # Apply action to environment
            self.environment.apply_action(action)
            
            # Get updated state (for reward, etc.)
            updated_state = self.environment.get_state()
            
            # Call callbacks
            if self._observation_callback:
                self._observation_callback(
                    updated_state.observation,
                    action,
                    updated_state.reward,
                    updated_state.info
                )
            
            if self._step_callback:
                self._step_callback(updated_state)
            
            # Wait for next control cycle
            await asyncio.sleep(self.control_period)
    
    def start(self):
        """Start the runner."""
        if not self.running:
            self.running = True
            self.environment.start_physics()
            self._control_task = asyncio.create_task(self._control_loop())
    
    def stop(self):
        """Stop the runner."""
        self.running = False
        if self._control_task:
            self._control_task.cancel()
        self.environment.stop_physics()
    
    async def wait_for_stop(self):
        """Wait for runner to stop."""
        if self._control_task:
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass
        await self.environment.wait_for_physics_stop()
    
    async def run(self, duration: Optional[float] = None):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Duration in seconds (None = run indefinitely)
        """
        self.start()
        try:
            if duration is not None:
                await asyncio.sleep(duration)
            else:
                # Run indefinitely
                while self.running:
                    await asyncio.sleep(1.0)
        finally:
            self.stop()
            await self.wait_for_stop()
    
    async def run_for_steps(self, max_steps: int):
        """
        Run simulation for specified number of control steps.
        
        Args:
            max_steps: Maximum number of control steps
        """
        steps = 0
        
        def step_callback(state):
            nonlocal steps
            steps += 1
        
        self.set_step_callback(step_callback)
        
        self.start()
        try:
            while steps < max_steps and self.running:
                await asyncio.sleep(0.1)
        finally:
            self.stop()
            await self.wait_for_stop()
    
    def set_observation_callback(self, callback: Callable):
        """
        Set callback for observations.
        
        Args:
            callback: Function(observation, action, reward, info)
        """
        self._observation_callback = callback
    
    def set_step_callback(self, callback: Callable):
        """
        Set callback for each step.
        
        Args:
            callback: Function(world_state)
        """
        self._step_callback = callback
    
    async def reset(self):
        """Reset the runner and environment."""
        await self.environment.reset()

