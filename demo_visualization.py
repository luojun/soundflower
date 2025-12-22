"""Demo script with visualization for Sound Flower environment."""

import asyncio
import numpy as np
from soundflower.environment import SoundFlowerEnvironment
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent
from visualization import SoundFlowerVisualizer


async def main():
    """Run demo with visualization."""
    print("=" * 60)
    print("Sound Flower - Visualization Demo")
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
        sound_source_angular_velocity=0.2,  # Sound source moves slowly
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
    
    # Create visualizer
    visualizer = SoundFlowerVisualizer(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        figsize=(10, 10)
    )
    
    print("\n" + "=" * 60)
    print("Running animated visualization (200 steps)...")
    print("Close the plot window to exit.")
    print("=" * 60)
    
    # Run async animation
    try:
        await visualizer.animate_async(env, agent, max_steps=200, update_interval=0.05)
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user.")
    
    visualizer.close()
    
    print("\n" + "=" * 60)
    print("Creating static snapshot visualization...")
    print("=" * 60)
    
    # Reset and create a static snapshot
    observation = env.reset()
    for _ in range(50):  # Run for 50 steps
        action = await agent.select_action(observation)
        observation, reward, done, info = await env.step(action)
    
    render_data = env.render()
    visualizer.plot_state(render_data, observation=observation, show=True)
    
    print("\n" + "=" * 60)
    print("Creating animation from episode...")
    print("=" * 60)
    
    # Collect render data for animation
    observation = env.reset()
    render_data_sequence = []
    
    for step in range(100):
        render_data = env.render()
        render_data_sequence.append(render_data)
        
        action = await agent.select_action(observation)
        observation, reward, done, info = await env.step(action)
        
        if done:
            break
    
    # Create animation
    anim = visualizer.create_animation(render_data_sequence, interval=50)
    
    print("Animation created! Close the plot window to exit.")
    visualizer.fig.show()
    
    try:
        input("\nPress Enter to close...")
    except KeyboardInterrupt:
        pass
    
    visualizer.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())

