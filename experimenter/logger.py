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

    def log_step(self, state: State):
        """Log the state."""
        print(f"State: {state}")

    def log_final(self, soundflower):
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Simulation time: {soundflower.simulation_time:.2f}s")
        print(f"  Total steps: {soundflower.step_count}")
        print(f"  Total reward: {soundflower.cumulative_reward:.4f}")
        print("=" * 60)
