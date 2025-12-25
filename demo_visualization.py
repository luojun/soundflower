"""Demo script with visualization for Sound Flower environment."""

import asyncio
from soundflower.environment import SoundFlowerEnvironment
from agents.heuristic_agent import HeuristicAgent
from experiments import Experiment, create_default_config
from visualization import SoundFlowerVisualizer


async def main():
    """Run demo with visualization."""
    print("=" * 60)
    print("Sound Flower - Visualization Demo")
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
    render_data_sequence = await experiment.collect_render_data(max_steps=100)
    
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
