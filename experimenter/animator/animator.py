"""Animator interface - handles visualization."""

from typing import Callable, Dict, Any, Optional
from environment import Environment
from .pygame_framer import PygameFramer


class Animator:
    """
    Animator interface for visualization.

    Decoupled from simulation logic, runs at configurable frame rate.
    Can be attached/detached from a running simulation.
    Supports both single-panel and multi-panel visualization.
    """

    def __init__(self, configs, panels_per_row: int = 1, window_size: Optional[tuple] = None):
        """
        Initialize animator.

        Args:
            configs: Single config or list of configs (one per panel)
            panels_per_row: Number of panels per row (default 1 for single panel)
            window_size: Optional window size, auto-calculated if None
        """
        # Handle single config as 1-element list for unified interface
        if not isinstance(configs, list):
            configs = [configs]

        # Convert config objects to dictionaries expected by PygameFramer
        framer_configs = []
        for config in configs:
            if hasattr(config, 'circle_radius'):  # It's a SoundFlowerConfig object
                framer_config = {
                    'circle_radius': config.circle_radius,
                    'link_lengths': config.link_lengths
                }
            else:  # It's already a dict
                framer_config = config
            framer_configs.append(framer_config)

        self.framer = PygameFramer(framer_configs, panels_per_row, window_size)
        self.environments = []  # Will be set by step() method
        self.agent_names = []   # Will be set by step() method

    def start(self):
        self.framer.start()

    def step(self, environments, agent_names=None):
        """
        Animation step using Environment objects.

        Args:
            environments: Single environment or list of environments (one per panel)
            agent_names: Optional list of agent names for each panel
        """
        # Handle single environment as 1-element list for unified interface
        if not isinstance(environments, list):
            environments = [environments]
        if agent_names is None:
            agent_names = [""] * len(environments)

        # Collect render data from all environments
        render_data_list = []
        for env in environments:
            if env is not None:
                render_data_list.append(env.get_render_data())
            else:
                render_data_list.append(None)

        self.framer.render_frame(render_data_list, agent_names)

    def render(self, render_data_list, agent_names=None):
        """
        Render pre-computed render data (for multi-process scenarios).

        Args:
            render_data_list: List of render data dicts (one per panel)
            agent_names: Optional list of agent names for each panel
        """
        if agent_names is None:
            agent_names = [""] * len(render_data_list)

        self.framer.render_frame(render_data_list, agent_names)

    def finish(self):
        self.framer.finish()

