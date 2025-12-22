"""Demo script with Pygame real-time animation for Sound Flower environment."""

import asyncio
import numpy as np
from soundflower.environment import SoundFlowerEnvironment
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent
from visualization import PygameVisualizer


async def main():
    """Run demo with Pygame animation."""
    print("=" * 60)
    print("Sound Flower - Pygame Real-time Animation Demo")
    print("=" * 60)
    
    # Create configuration
    config = SoundFlowerConfig(
        num_links=2,
        link_lengths=[0.6, 0.4],
        link_masses=[1.0, 0.8],
        joint_frictions=[0.1, 0.15],
        circle_radius=1.0,
        num_microphones=1,
        microphone_gain=1.0,
        num_sound_sources=1,
        sound_source_strength=2.0,
        sound_attenuation_coeff=1.0,
        dt=0.01,
        max_torque=5.0,
        sound_source_angular_velocity=0.3,  # Sound source moves
        sound_source_initial_angle=np.pi / 4  # Start at 45 degrees
    )
    
    print("\nConfiguration:")
    print(f"  Number of links: {config.num_links}")
    print(f"  Link lengths: {config.link_lengths}")
    print(f"  Circle radius: {config.circle_radius}")
    print(f"  Total arm length: {sum(config.link_lengths):.2f}")
    print(f"  Sound source angular velocity: {config.sound_source_angular_velocity:.2f} rad/s")
    
    # Create environment
    env = SoundFlowerEnvironment(config)
    
    # Create agent
    agent = HeuristicAgent(env, kp=8.0, kd=1.0)
    
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

