"""Demo script for Sound Flower environment with heuristic agent."""

import asyncio
import numpy as np
from soundflower.environment import SoundFlowerEnvironment
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent
from visualization import SoundFlowerVisualizer


async def main():
    """Run demo of Sound Flower environment."""
    print("=" * 60)
    print("Sound Flower - Robotic Arm Sound Source Tracking Demo")
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
    
    print("\nRunning episode...")
    print("-" * 60)
    
    # Run episode
    episode_stats = await agent.run_episode(max_steps=2000, render=False)
    
    print("\nEpisode Statistics:")
    print(f"  Total steps: {episode_stats['steps']}")
    print(f"  Total reward: {episode_stats['total_reward']:.4f}")
    print(f"  Final sound energy: {episode_stats['final_sound_energy']:.4f}")
    print(f"  Final distance to source: {episode_stats['final_distance_to_source']:.4f}")
    
    # Run a shorter episode with rendering to show behavior
    print("\n" + "=" * 60)
    print("Running visualization episode (100 steps)...")
    print("=" * 60)
    
    observation = env.reset()
    print(f"\nInitial state:")
    print(f"  Arm angles: {observation.arm_angles}")
    print(f"  End effector position: {observation.end_effector_pos}")
    print(f"  Sound source position: {observation.sound_source_positions[0]}")
    print(f"  Initial sound energy: {observation.sound_energy:.4f}")
    
    # Create visualizer for static snapshot
    visualizer = SoundFlowerVisualizer(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        figsize=(10, 10)
    )
    
    total_reward = 0.0
    for step in range(100):
        action = await agent.select_action(observation)
        observation, reward, done, info = await env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            render_data = env.render()
            print(f"\nStep {step}:")
            print(f"  Arm angles: {observation.arm_angles}")
            print(f"  End effector: ({observation.end_effector_pos[0]:.3f}, {observation.end_effector_pos[1]:.3f})")
            print(f"  Sound source: ({observation.sound_source_positions[0][0]:.3f}, {observation.sound_source_positions[0][1]:.3f})")
            print(f"  Sound energy: {observation.sound_energy:.4f}")
            print(f"  Distance to source: {info['end_effector_distance_to_source']:.4f}")
            print(f"  Cumulative reward: {total_reward:.4f}")
    
    # Show final state visualization
    print("\nDisplaying final state visualization...")
    final_render_data = env.render()
    visualizer.plot_state(final_render_data, observation=observation, show=True)
    visualizer.close()
    
    print(f"\nFinal cumulative reward: {total_reward:.4f}")
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: For animated visualization, run: python demo_visualization.py")


if __name__ == "__main__":
    asyncio.run(main())

