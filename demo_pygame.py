"""Demo script with Pygame real-time animation for Sound Flower environment."""

import asyncio
from soundflower.environment import SoundFlowerEnvironment
from agents.heuristic_agent import HeuristicAgent
from experiments import Experiment, create_default_config
from visualization import PygameVisualizer


async def main():
    """Run demo with Pygame animation."""
    print("=" * 60)
    print("Sound Flower - Pygame Real-time Animation Demo")
    print("=" * 60)
    
    # Create configuration (faster sound source for better visualization)
    config = create_default_config(sound_source_angular_velocity=0.3)
    
    # Create environment
    env = SoundFlowerEnvironment(config)
    
    # Create agent (config passed for max_torque)
    agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
    
    # Create experiment
    experiment = Experiment(env, agent, config)
    
    # Print configuration
    experiment.print_config()
    
    # Create Pygame visualizer
    visualizer = PygameVisualizer(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        window_size=(800, 800),
        fps=60
    )
    
    print("\n" + "=" * 60)
    print("Starting real-time animation...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  ESC or Q: Quit")
    print("=" * 60)
    
    try:
        # Run animation
        await visualizer.animate(env, agent, max_steps=10000, steps_per_frame=1)
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user.")
    finally:
        visualizer.close()
        print("\nAnimation closed.")


if __name__ == "__main__":
    asyncio.run(main())
