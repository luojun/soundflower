"""Real-time plotter for simulation metrics using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
import threading


class Plotter:
    """
    Plotter interface for real-time visualization.

    Designed as a shared instance that can track multiple simulation instances.
    Decoupled from simulation logic, runs at configurable frequency.
    """

    # Class-level shared instance and lock for thread safety
    _shared_instance: Optional['Plotter'] = None
    _lock = threading.Lock()

    def __init__(self, config, agent_name: Optional[str] = None, shared: bool = True):
        """
        Initialize plotter.

        Args:
            config: Configuration object
            agent_name: Name of the agent for this simulation instance
            shared: If True, use shared instance for multi-agent scenarios
        """
        self.config = config
        self.agent_name = agent_name

        if shared:
            with Plotter._lock:
                if Plotter._shared_instance is None:
                    # First instance - initialize plots
                    Plotter._shared_instance = self
                    self._is_shared = True
                    self._initialize_plots()
                    # Initialize data storage
                    self._step_rewards: Dict[str, List[tuple]] = defaultdict(list)
                    self._step_energies: Dict[str, List[tuple]] = defaultdict(list)
                    self._cumulative_rewards: Dict[str, List[tuple]] = defaultdict(list)
                    self._cumulative_energies: Dict[str, List[tuple]] = defaultdict(list)
                else:
                    # Use existing shared instance - store reference
                    shared_inst = Plotter._shared_instance
                    # Copy attributes to this instance for convenience
                    self._shared_instance = shared_inst
                    self._is_shared = True
                    self.fig = shared_inst.fig
                    self.axes = shared_inst.axes
                    self.ax_reward = shared_inst.ax_reward
                    self.ax_energy = shared_inst.ax_energy
                    self.ax_cum_reward = shared_inst.ax_cum_reward
                    self.ax_cum_energy = shared_inst.ax_cum_energy
                    # Note: agent_name is already set above, and step() will forward to shared instance
                    return
        else:
            self._is_shared = False
            self._initialize_plots()
            # Initialize data storage
            self._step_rewards: Dict[str, List[tuple]] = defaultdict(list)
            self._step_energies: Dict[str, List[tuple]] = defaultdict(list)
            self._cumulative_rewards: Dict[str, List[tuple]] = defaultdict(list)
            self._cumulative_energies: Dict[str, List[tuple]] = defaultdict(list)

        # Agent color and style mapping
        self._agent_colors = {
            'PointingAgent': '#FF6464',      # Red
            'ApproachingAgent': '#64FF64',    # Green
            'TrackingAgent': '#6464FF',       # Blue
        }
        self._agent_styles = {
            'PointingAgent': '-',            # Solid
            'ApproachingAgent': '--',        # Dashed
            'TrackingAgent': '-.',           # Dash-dot
        }

        # Track cumulative values per agent (only for non-shared or first shared instance)
        if not (self._is_shared and hasattr(self, '_shared_instance')):
            self._agent_cumulative_reward: Dict[str, float] = defaultdict(float)
            self._agent_cumulative_energy: Dict[str, float] = defaultdict(float)

    def _initialize_plots(self):
        """Initialize matplotlib figures and axes."""
        matplotlib.use('TkAgg')  # Use TkAgg backend for interactive mode
        plt.ion()  # Turn on interactive mode

        # Create figure with 4 subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Sound Flower Simulation Metrics', fontsize=14, fontweight='bold')

        # Configure subplots
        self.ax_reward = self.axes[0, 0]
        self.ax_energy = self.axes[0, 1]
        self.ax_cum_reward = self.axes[1, 0]
        self.ax_cum_energy = self.axes[1, 1]

        self.ax_reward.set_title('Per-Step Reward (Normalized)')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)

        self.ax_energy.set_title('Per-Step Sound Energy')
        self.ax_energy.set_xlabel('Step')
        self.ax_energy.set_ylabel('Energy (J)')
        self.ax_energy.grid(True, alpha=0.3)

        self.ax_cum_reward.set_title('Cumulative Reward (Normalized)')
        self.ax_cum_reward.set_xlabel('Step')
        self.ax_cum_reward.set_ylabel('Cumulative Reward')
        self.ax_cum_reward.grid(True, alpha=0.3)

        self.ax_cum_energy.set_title('Cumulative Sound Energy')
        self.ax_cum_energy.set_xlabel('Step')
        self.ax_cum_energy.set_ylabel('Cumulative Energy (J)')
        self.ax_cum_energy.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

    def start(self):
        """Start the plotter."""
        if self._is_shared and self._shared_instance is not self:
            return
        # Plots are already initialized, nothing more to do

    def step(self, step_count: int, reward: float, sound_energy: float,
             cumulative_reward: float, cumulative_sound_energy: float):
        """
        Update plots with new data point.

        Args:
            step_count: Current step number
            reward: Normalized reward for this step
            sound_energy: Raw sound energy for this step (Joules)
            cumulative_reward: Cumulative normalized reward
            cumulative_sound_energy: Cumulative sound energy (Joules)
        """
        # If using shared instance and this is not the main instance, forward to main
        # but preserve this instance's agent_name
        if self._is_shared and hasattr(self, '_shared_instance') and self._shared_instance is not self:
            # Temporarily set agent_name on shared instance, call step, then restore
            original_name = self._shared_instance.agent_name
            self._shared_instance.agent_name = self.agent_name
            self._shared_instance.step(
                step_count, reward, sound_energy,
                cumulative_reward, cumulative_sound_energy
            )
            self._shared_instance.agent_name = original_name
            return

        if self.agent_name is None:
            return

        # Store data
        self._step_rewards[self.agent_name].append((step_count, reward))
        self._step_energies[self.agent_name].append((step_count, sound_energy))
        self._cumulative_rewards[self.agent_name].append((step_count, cumulative_reward))
        self._cumulative_energies[self.agent_name].append((step_count, cumulative_sound_energy))

        # Update plots
        self._update_plots()

    def _update_plots(self):
        """Update all plots with current data."""
        # Clear axes
        self.ax_reward.clear()
        self.ax_energy.clear()
        self.ax_cum_reward.clear()
        self.ax_cum_energy.clear()

        # Redraw titles and labels
        self.ax_reward.set_title('Per-Step Reward (Normalized)')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)

        self.ax_energy.set_title('Per-Step Sound Energy')
        self.ax_energy.set_xlabel('Step')
        self.ax_energy.set_ylabel('Energy (J)')
        self.ax_energy.grid(True, alpha=0.3)

        self.ax_cum_reward.set_title('Cumulative Reward (Normalized)')
        self.ax_cum_reward.set_xlabel('Step')
        self.ax_cum_reward.set_ylabel('Cumulative Reward')
        self.ax_cum_reward.grid(True, alpha=0.3)

        self.ax_cum_energy.set_title('Cumulative Sound Energy')
        self.ax_cum_energy.set_xlabel('Step')
        self.ax_cum_energy.set_ylabel('Cumulative Energy (J)')
        self.ax_cum_energy.grid(True, alpha=0.3)

        # Plot data for each agent
        for agent_name in self._step_rewards.keys():
            color = self._agent_colors.get(agent_name, '#808080')
            style = self._agent_styles.get(agent_name, '-')

            # Per-step reward
            if self._step_rewards[agent_name]:
                steps, rewards = zip(*self._step_rewards[agent_name])
                self.ax_reward.plot(steps, rewards, color=color, linestyle=style,
                                   label=agent_name, linewidth=1.5)

            # Per-step energy
            if self._step_energies[agent_name]:
                steps, energies = zip(*self._step_energies[agent_name])
                self.ax_energy.plot(steps, energies, color=color, linestyle=style,
                                  label=agent_name, linewidth=1.5)

            # Cumulative reward
            if self._cumulative_rewards[agent_name]:
                steps, cum_rewards = zip(*self._cumulative_rewards[agent_name])
                self.ax_cum_reward.plot(steps, cum_rewards, color=color, linestyle=style,
                                       label=agent_name, linewidth=1.5)

            # Cumulative energy
            if self._cumulative_energies[agent_name]:
                steps, cum_energies = zip(*self._cumulative_energies[agent_name])
                self.ax_cum_energy.plot(steps, cum_energies, color=color, linestyle=style,
                                       label=agent_name, linewidth=1.5)

        # Add legends
        self.ax_reward.legend(loc='best', fontsize=8)
        self.ax_energy.legend(loc='best', fontsize=8)
        self.ax_cum_reward.legend(loc='best', fontsize=8)
        self.ax_cum_energy.legend(loc='best', fontsize=8)

        # Refresh display
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish(self):
        """Finish plotting (keep plots open)."""
        if self._is_shared and hasattr(self, '_shared_instance') and self._shared_instance is not self:
            return
        # Keep plots open for inspection
        # plt.ioff()  # Turn off interactive mode if desired
        # plt.show(block=True)  # Block until window is closed

