"""Abstract plotter interface and implementations for visualization."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import os
from collections import defaultdict
import threading

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
except ImportError:
    plt = None
    matplotlib = None
    np = None


class Plotter(ABC):
    """
    Abstract base class for plotters.

    Defines the interface that all plotter implementations must follow.
    """

    def __init__(self, config, agent_name: Optional[str] = None):
        """
        Initialize plotter.

        Args:
            config: Configuration object
            agent_name: Name of the agent for this simulation instance
        """
        self.config = config
        self.agent_name = agent_name

    @abstractmethod
    def start(self):
        """Start plotting."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def finish(self):
        """Finish plotting."""
        pass


class TensorBoardPlotter(Plotter):
    """
    TensorBoard plotter implementation.

    Logs metrics to TensorBoard files for non-blocking visualization.
    Designed for multi-process scenarios where matplotlib would cause lag.
    """

    def __init__(self, config, agent_name: Optional[str] = None):
        """
        Initialize TensorBoard plotter.

        Args:
            config: Configuration object
            agent_name: Name of the agent for this simulation instance
        """
        super().__init__(config, agent_name)

        # Create TensorBoard writer if available
        self.writer = None
        if SummaryWriter is not None and agent_name is not None:
            log_dir = os.path.join("logs", agent_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        elif SummaryWriter is None:
            print(f"Warning: tensorboardX not installed. Install with: pip install tensorboardX")

    def start(self):
        """Start plotting (no-op for TensorBoard, files are written on step)."""
        pass

    def step(self, step_count: int, reward: float, sound_energy: float,
             cumulative_reward: float, cumulative_sound_energy: float):
        """
        Log metrics to TensorBoard.

        Args:
            step_count: Current step number
            reward: Normalized reward for this step
            sound_energy: Raw sound energy for this step (Joules)
            cumulative_reward: Cumulative normalized reward
            cumulative_sound_energy: Cumulative sound energy (Joules)
        """
        if self.writer is None:
            return

        self.writer.add_scalar('reward', reward, step_count)
        self.writer.add_scalar('energy', sound_energy, step_count)
        self.writer.add_scalar('cumulative_reward', cumulative_reward, step_count)
        self.writer.add_scalar('cumulative_energy', cumulative_sound_energy, step_count)

    def finish(self):
        """Finish plotting (close TensorBoard writer)."""
        if self.writer is not None:
            self.writer.close()


class MatplotlibPlotter(Plotter):
    """
    Matplotlib plotter implementation for real-time visualization.

    Designed as a shared instance that can track multiple simulation instances.
    Decoupled from simulation logic, runs at configurable frequency.
    """

    # Class-level shared instance and lock for thread safety
    _shared_instance: Optional['MatplotlibPlotter'] = None
    _lock = threading.Lock()

    def __init__(self, config, agent_name: Optional[str] = None):
        """
        Initialize matplotlib plotter.

        Args:
            config: Configuration object
            agent_name: Name of the agent for this simulation instance
        """
        if plt is None or np is None:
            raise ImportError("matplotlib and numpy are required for MatplotlibPlotter")
        super().__init__(config, agent_name)

        with MatplotlibPlotter._lock:
            if MatplotlibPlotter._shared_instance is None:
                # First instance - initialize plots
                MatplotlibPlotter._shared_instance = self
                self._is_shared = True
                self._initialize_plots()
                # Initialize data storage as numpy arrays (x, y tuples)
                self._step_rewards: Dict[str, Tuple[np.ndarray, np.ndarray]] = defaultdict(lambda: (np.array([]), np.array([])))
                self._step_energies: Dict[str, Tuple[np.ndarray, np.ndarray]] = defaultdict(lambda: (np.array([]), np.array([])))
                self._cumulative_rewards: Dict[str, Tuple[np.ndarray, np.ndarray]] = defaultdict(lambda: (np.array([]), np.array([])))
                self._cumulative_energies: Dict[str, Tuple[np.ndarray, np.ndarray]] = defaultdict(lambda: (np.array([]), np.array([])))
                # Line objects for incremental updates
                self._line_objects: Dict[str, Dict[str, Optional[plt.Line2D]]] = defaultdict(lambda: {
                    'reward': None, 'energy': None, 'cum_reward': None, 'cum_energy': None
                })
                # Update tracking
                self._known_agents = set()
                self._legend_needs_update = False
                self.MAX_DATA_POINTS = 2000
            else:
                # Use existing shared instance - store reference
                shared_inst = MatplotlibPlotter._shared_instance
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
                # Data storage and line objects are on shared_inst, accessed via forwarding
                return

        # Agent color and style mapping
        # Base colors for agent types
        self._base_colors = {
            'PointingAgent': '#FF6464',      # Red
            'ApproachingAgent': '#64FF64',    # Green
            'TrackingAgent': '#6464FF',       # Blue
        }
        # Base styles for link configurations
        self._base_styles = {
            '2-link': '-',                   # Solid
            '3-link': '--',                  # Dashed
        }

        # Track cumulative values per agent (only for non-shared or first shared instance)
        if not (self._is_shared and hasattr(self, '_shared_instance')):
            self._agent_cumulative_reward: Dict[str, float] = defaultdict(float)
            self._agent_cumulative_energy: Dict[str, float] = defaultdict(float)

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for an agent based on its name."""
        # Extract base agent name (before underscore)
        base_name = agent_name.split('_')[0]
        return self._base_colors.get(base_name, '#808080')  # Default gray

    def _get_agent_style(self, agent_name: str) -> str:
        """Get line style for an agent based on its name."""
        # Extract link configuration (after underscore, if present)
        if '_' in agent_name:
            link_config = agent_name.split('_')[1]
            return self._base_styles.get(link_config, '-')  # Default solid
        return '-'  # Default solid

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
        self._layout_fixed = True  # Layout fixed, don't recalculate
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

        # Track new agents
        if self.agent_name not in self._known_agents:
            self._known_agents.add(self.agent_name)
            self._legend_needs_update = True

        # Append data to numpy arrays with rolling window limit
        self._append_data(self.agent_name, step_count, reward, sound_energy,
                         cumulative_reward, cumulative_sound_energy)

        # Update plots incrementally
        self._update_plots()

    def _append_data(self, agent_name: str, step: int, reward: float,
                    energy: float, cum_reward: float, cum_energy: float):
        """Append data point with rolling window limit."""
        def append_limited(arr_tuple, x, y):
            x_arr, y_arr = arr_tuple
            if len(x_arr) == 0:
                return np.array([x], dtype=np.float64), np.array([y], dtype=np.float64)
            x_new = np.append(x_arr, x)
            y_new = np.append(y_arr, y)
            if len(x_new) > self.MAX_DATA_POINTS:
                return x_new[-self.MAX_DATA_POINTS:], y_new[-self.MAX_DATA_POINTS:]
            return x_new, y_new

        self._step_rewards[agent_name] = append_limited(
            self._step_rewards[agent_name], step, reward)
        self._step_energies[agent_name] = append_limited(
            self._step_energies[agent_name], step, energy)
        self._cumulative_rewards[agent_name] = append_limited(
            self._cumulative_rewards[agent_name], step, cum_reward)
        self._cumulative_energies[agent_name] = append_limited(
            self._cumulative_energies[agent_name], step, cum_energy)

    def _update_plots(self):
        """Update plots incrementally using set_data()."""
        # Update each plot type incrementally
        self._update_plot('reward', self.ax_reward, self._step_rewards)
        self._update_plot('energy', self.ax_energy, self._step_energies)
        self._update_plot('cum_reward', self.ax_cum_reward, self._cumulative_rewards)
        self._update_plot('cum_energy', self.ax_cum_energy, self._cumulative_energies)

        # Update axes limits efficiently
        self._update_axes_limits()

        # Update legend only if needed
        if self._legend_needs_update:
            self._update_legend()
            self._legend_needs_update = False

        # Refresh display (non-blocking)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_plot(self, plot_type: str, ax, data_dict: Dict):
        """Update a single plot incrementally."""
        for agent_name in data_dict.keys():
            x_data, y_data = data_dict[agent_name]
            if len(x_data) == 0:
                continue

            color = self._get_agent_color(agent_name)
            style = self._get_agent_style(agent_name)

            line = self._line_objects[agent_name][plot_type]
            if line is None:
                # Create line on first update
                line, = ax.plot(x_data, y_data, color=color, linestyle=style,
                               label=agent_name, linewidth=1.5)
                self._line_objects[agent_name][plot_type] = line
            else:
                # Update existing line data
                line.set_data(x_data, y_data)

    def _update_axes_limits(self):
        """Update axes limits efficiently."""
        # Update limits for each axis based on current data
        for ax, data_dict in [
            (self.ax_reward, self._step_rewards),
            (self.ax_energy, self._step_energies),
            (self.ax_cum_reward, self._cumulative_rewards),
            (self.ax_cum_energy, self._cumulative_energies)
        ]:
            all_x = []
            all_y = []
            for x_arr, y_arr in data_dict.values():
                if len(x_arr) > 0:
                    all_x.extend(x_arr)
                    all_y.extend(y_arr)

            if all_x and all_y:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                y_range = y_max - y_min if y_max != y_min else 1.0
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    def _update_legend(self):
        """Update legend only when agents change."""
        for ax in [self.ax_reward, self.ax_energy, self.ax_cum_reward, self.ax_cum_energy]:
            ax.legend(loc='best', fontsize=10, ncol=2)

    def finish(self):
        """Finish plotting (keep plots open)."""
        if self._is_shared and hasattr(self, '_shared_instance') and self._shared_instance is not self:
            return
        # Keep plots open for inspection
        # plt.ioff()  # Turn off interactive mode if desired
        # plt.show(block=True)  # Block until window is closed


def create_plotter(plotter_type: str, config, agent_name: Optional[str] = None):
    """
    Factory function to create the appropriate plotter.

    Args:
        plotter_type: 'matplotlib' or 'tensorboard'
        config: Configuration object
        agent_name: Name of the agent for this simulation instance

    Returns:
        Plotter instance
    """
    if plotter_type == 'matplotlib':
        return MatplotlibPlotter(config, agent_name)
    elif plotter_type == 'tensorboard':
        return TensorBoardPlotter(config, agent_name)
    else:
        raise ValueError(f"Unknown plotter type: {plotter_type}")
