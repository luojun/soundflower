"""Demo script for Sound Flower environment with heuristic agent."""

import asyncio
from soundflower.environment import SoundFlowerEnvironment
from agents.heuristic_agent import HeuristicAgent
from experiments import Experiment, create_default_config
from visualization import SoundFlowerVisualizer


async def main():
    """Run demo of Sound Flower environment."""
    print("=" * 60)
    print("Sound Flower - Robotic Arm Sound Source Tracking Demo")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config(sound_source_angular_velocity=0.2)
    
    # Create environment
    env = SoundFlowerEnvironment(config)
    
    # Create agent (config passed for max_torque)
    agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
    
    # Create experiment
    experiment = Experiment(env, agent, config)
    
    # Print configuration
    experiment.print_config()
    
    print("\nRunning episode...")
    print("-" * 60)
    
    # Run episode
    episode_stats = await experiment.run_episode(max_steps=2000, render=False)
    experiment.print_episode_stats(episode_stats)
    
    # Run a shorter episode with rendering to show behavior
    print("\n" + "=" * 60)
    print("Running visualization episode (100 steps)...")
    print("=" * 60)
    
    stats = await experiment.run_episode_with_logging(max_steps=100, log_interval=20)
    
    # Show final state visualization
    print("\nDisplaying final state visualization...")
    visualizer = SoundFlowerVisualizer(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        figsize=(10, 10)
    )
    final_render_data = env.render()
    visualizer.plot_state(final_render_data, observation=stats['final_observation'], show=True)
    visualizer.close()
    
    print(f"\nFinal cumulative reward: {stats['total_reward']:.4f}")
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: For animated visualization, run: python3 demo_visualization.py")


if __name__ == "__main__":
    asyncio.run(main())
