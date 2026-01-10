from environment import State
from experimenter import SoundFlowerConfig

class Logger:
    """
    Interface for logging.
    """

    def __init__(self):
        pass

    def log_config(self, config: SoundFlowerConfig):
        print("\nConfiguration:")
        print(f"  Physics time step: {config.dt}s")
        print(f"  Control frequency: {config.control_frequency} Hz")
        print(f"  Animation FPS: {config.animation_frequency}")

    def log_step(self, state: State, simulation_time: float, step_count: int):
        """Log the state with formatted output."""
        obs = state.observation
        if obs:
            print(f"[Step {step_count:6d} | Time {simulation_time:7.2f}s] "
                  f"Sound Intensity: {obs.sound_intensity:8.4f} | "
                  f"Sound Energy: {obs.sound_energy:8.6f} | "
                  f"Energy Delta: {obs.sound_energy_delta:8.6f}")

    def log_final(self, soundflower):
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Simulation time: {soundflower.simulation_time:.2f}s")
        print(f"  Total steps: {soundflower.step_count}")
        print(f"  Total reward: {soundflower.cumulative_reward:.4f}")
        print(f"  Cumulative sound energy: {soundflower.cumulative_sound_energy:.6f}")
        print("=" * 60)
