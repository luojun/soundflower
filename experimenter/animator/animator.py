"""Animator interface - handles visualization."""

from typing import Callable, Dict, Any, Optional
from environment import Environment
from .pygame_framer import PygameFramer


class Animator:
    """
    Animator interface for visualization.
    
    Decoupled from simulation logic, runs at configurable frame rate.
    Can be attached/detached from a running simulation.
    """
    
    def __init__(self, config):
        """
        Initialize animator.
        
        Args:
            environment: Environment instance to animate
            config: Configuration object
            render_callback: Function to call with render data. Should return True to continue, False to quit.
            fps: Frame rate (10-100)
        """
        self.framer = PygameFramer(
            circle_radius=config.circle_radius,
            link_lengths=config.link_lengths,
            window_size=(800, 800)
        )

    def start(self):
        self.framer.start()

    def step(self, environment):
        """Animation step."""
        render_data = environment.get_render_data()
        self.framer.render_frame(render_data)
    
    def finish(self):
        self.framer.finish()

