"""Demo showing the new decoupled World/Runner/Renderer architecture."""

import asyncio
from soundflower.world import World
from soundflower.runner import Runner
from soundflower.renderer import Renderer
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent
from experiments import create_default_config
from visualization.pygame_visualizer_v2 import PygameVisualizerV2


async def main():
    """Run demo with decoupled architecture."""
    print("=" * 60)
    print("Sound Flower - Decoupled Architecture Demo")
    print("=" * 60)
    print("\nThis demo shows the separation of:")
    print("  - World: Simulation state and logic")
    print("  - Runner: Orchestrates simulation execution")
    print("  - Renderer: Handles visualization (optional)")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config(sound_source_angular_velocity=0.3)
    config.control_frequency = 50.0  # 50 Hz control
    config.visualization_fps = 60.0  # 60 fps visualization
    
    print("\nConfiguration:")
    print(f"  Physics time step: {config.dt}s")
    print(f"  Control frequency: {config.control_frequency} Hz")
    print(f"  Visualization FPS: {config.visualization_fps}")
    
    # Create World (simulation state and logic)
    world = World(config)
    
    # Create Agent
    agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
    
    # Create Runner (orchestrates simulation)
    runner = Runner(world, agent, config)
    
    # Create Renderer (visualization)
    visualizer = PygameVisualizerV2(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        window_size=(800, 800),
        fps=config.visualization_fps
    )
    
    renderer = Renderer(
        world=world,
        config=config,
        render_callback=visualizer.render_callback,
        fps=config.visualization_fps
    )
    
    print("\n" + "=" * 60)
    print("Starting simulation...")
    print("Controls:")
    print("  SPACE: Pause/Resume (renderer)")
    print("  ESC or Q: Quit")
    print("=" * 60)
    
    # Reset world
    await world.reset()
    
    # Start all components
    world.start_physics()
    runner.start()
    renderer.start()
    
    try:
        # Run until user quits or 30 seconds pass
        # Check if renderer is still running (user can quit via keyboard)
        start_time = asyncio.get_event_loop().time()
        max_duration = 30.0
        
        while renderer.is_running:
            await asyncio.sleep(0.1)  # Check every 100ms
            
            # Check if max duration reached
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_duration:
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Stop all components
        renderer.stop()
        await renderer.wait_for_stop()
        
        runner.stop()
        await runner.wait_for_stop()
        
        world.stop_physics()
        await world.wait_for_physics_stop()
        
        visualizer.close()
        
        # Print final statistics
        final_state = world.get_state()
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Simulation time: {final_state.info['simulation_time']:.2f}s")
        print(f"  Control steps: {final_state.info['step_count']}")
        print(f"  Final sound energy: {final_state.observation.sound_energy:.4f}")
        print("=" * 60)


async def headless_demo():
    """Run headless (no visualization) - can run faster than real-time."""
    print("=" * 60)
    print("Sound Flower - Headless Mode Demo")
    print("=" * 60)
    print("\nRunning without visualization for faster execution...")
    
    config = create_default_config()
    config.control_frequency = 100.0  # Higher control frequency for headless
    
    world = World(config)
    agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
    runner = Runner(world, agent, config)
    
    # Track statistics
    total_reward = 0.0
    steps = 0
    
    def step_callback(state):
        nonlocal total_reward, steps
        total_reward += state.reward
        steps += 1
        if steps % 100 == 0:
            print(f"Step {steps}: Reward={state.reward:.4f}, "
                  f"Sound Energy={state.observation.sound_energy:.4f}, "
                  f"Time={state.info['simulation_time']:.2f}s")
    
    runner.set_step_callback(step_callback)
    
    await world.reset()
    world.start_physics()
    runner.start()
    
    try:
        # Run for 10 seconds of simulated time
        # This will run much faster than real-time since there's no rendering
        await asyncio.sleep(10.0)
    finally:
        runner.stop()
        await runner.wait_for_stop()
        world.stop_physics()
        await world.wait_for_physics_stop()
        
        print(f"\nTotal steps: {steps}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Average reward per step: {total_reward/steps:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        asyncio.run(headless_demo())
    else:
        asyncio.run(main())

