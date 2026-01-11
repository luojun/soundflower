from environment import State
from experimenter import SoundFlowerConfig

class Logger:
    """
    Interface for logging.
    """

    def __init__(self, agent_name: str = None):
        """
        Initialize logger.

        Args:
            agent_name: Optional name/type of agent for identification in multi-instance scenarios
        """
        self.agent_name = agent_name

    def _get_agent_prefix(self) -> str:
        """Get prefix string for agent identification."""
        if self.agent_name:
            return f"[{self.agent_name}] "
        return ""

    def log_config(self, config: SoundFlowerConfig):
        prefix = self._get_agent_prefix()
        print(f"\n{prefix}Configuration:")
        print(f"{prefix}  Physics time step: {config.dt}s")
        print(f"{prefix}  Control frequency: {config.control_frequency} Hz")
        print(f"{prefix}  Animation FPS: {config.animation_frequency}")

    def log_step(self, state: State, simulation_time: float, step_count: int):
        """Log the state with formatted output."""
        obs = state.observation
        if obs:
            prefix = self._get_agent_prefix()
            print(f"{prefix}[Step {step_count:6d} | Time {simulation_time:7.2f}s] "
                  f"Sound Intensity: {obs.sound_intensity:8.2f} W/m² | "
                  f"Sound Energy: {obs.sound_energy:12.4f} J | "
                  f"Reward: {state.reward:12.4f}")

    def log_final(self, soundflower):
        prefix = self._get_agent_prefix()
        print(f"\n{prefix}" + "=" * 60)
        print(f"{prefix}Final Statistics:")
        print(f"{prefix}  Simulation time: {soundflower.simulation_time:.2f}s")
        print(f"{prefix}  Total steps: {soundflower.step_count}")
        print(f"{prefix}  Total reward: {soundflower.cumulative_reward:.4f}")
        print(f"{prefix}  Cumulative sound energy: {soundflower.cumulative_sound_energy:.4f} J")
        print(f"{prefix}" + "=" * 60)
