"""Animator interface - handles visualization."""

import asyncio
from typing import Callable, Dict, Any, Optional
from soundflower import World


class Animator:
    """
    Animator interface for visualization.
    
    Decoupled from simulation logic, runs at configurable frame rate.
    Can be attached/detached from a running simulation.
    """
    
    def __init__(self, world: World, config,
                 render_callback: Callable[[Dict[str, Any]], bool],
                 fps: float = 60.0):
        """
        Initialize animator.
        
        Args:
            world: World instance to animate
            config: Configuration object
            render_callback: Function to call with render data. Should return True to continue, False to quit.
            fps: Frame rate (10-100)
        """
        self.world = world
        self.config = config
        self.render_callback = render_callback
        self.fps = max(10.0, min(100.0, fps))
        self.frame_period = 1.0 / self.fps
        
        self.running = False
        self._animation_task: Optional[asyncio.Task] = None
    
    @property
    def is_running(self) -> bool:
        """Check if Animator is running."""
        return self.running
    
    async def _animation_loop(self):
        """Main animation loop."""
        while self.running:
            # Get render data from world
            render_data = self.world.get_render_data()
            
            # Call render callback - check if it wants to quit
            result = self.render_callback(render_data)
            if result is False:
                # Callback returned False, stop animation
                self.running = False
                break
            
            # Wait for next frame
            await asyncio.sleep(self.frame_period)
    
    def start(self):
        """Start the animator."""
        if not self.running:
            self.running = True
            self._animation_task = asyncio.create_task(self._animation_loop())
    
    def stop(self):
        """Stop the animator."""
        self.running = False
        if self._animation_task:
            self._animation_task.cancel()
    
    async def wait_for_stop(self):
        """Wait for Animator to stop."""
        if self._animation_task:
            try:
                await self._animation_task
            except asyncio.CancelledError:
                pass
    
    def set_fps(self, fps: float):
        """
        Change frame rate.
        
        Args:
            fps: New frame rate
        """
        self.fps = max(10.0, min(100.0, fps))
        self.frame_period = 1.0 / self.fps

