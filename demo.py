"""Demo script for Sound Flower environment with optional animation."""

import asyncio
import sys
from soundflower import World
from experimenter import Runner, create_default_config
from experimenter.animator import PygameFramer, Animator
from agents.heuristic_agent import HeuristicAgent


async def main(headless: bool = False):
    """Run demo with optional animation."""
    print("=" * 60)
    print("Sound Flower - Robotic Arm Sound Source Tracking Demo")
    print("=" * 60)
    
    if headless:
        print("\nRunning in headless mode (no animation, faster execution)...")
    else:
        print("\nRunning with animation...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  ESC or Q: Quit")
    
    print("=" * 60)
    
    # Create configuration
    config = create_default_config(sound_source_angular_velocity=0.3)
    config.control_frequency = 100.0 if headless else 50.0
    config.visualization_fps = 60.0
    
    print("\nConfiguration:")
    print(f"  Physics time step: {config.dt}s")
    print(f"  Control frequency: {config.control_frequency} Hz")
    if not headless:
        print(f"  Animation FPS: {config.visualization_fps}")
    
    # Create World (simulation state and logic)
    world = World(config)
    
    # Create Agent
    agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
    
    # Create Runner (orchestrates simulation)
    runner = Runner(world, agent, config)
    
    # Create Animator and Framer (only if not headless)
    animator = None
    framer = None
    
    if not headless:
        framer = PygameFramer(
            circle_radius=config.circle_radius,
            link_lengths=config.link_lengths,
            window_size=(800, 800),
            fps=config.visualization_fps
        )
        
        animator = Animator(
            world=world,
            config=config,
            render_callback=framer.render_callback,
            fps=config.visualization_fps
        )
    
    # Track statistics
    total_reward = 0.0
    steps = 0
    
    def step_callback(state):
        nonlocal total_reward, steps
        total_reward += state.reward
        steps += 1
        if headless and steps % 100 == 0:
            print(f"Step {steps}: Reward={state.reward:.4f}, "
                  f"Sound Energy={state.observation.sound_energy:.4f}, "
                  f"Time={state.info['simulation_time']:.2f}s")
    
    runner.set_step_callback(step_callback)
    
    # Reset world
    await world.reset()
    
    # Start all components
    world.start_physics()
    runner.start()
    if animator:
        animator.start()
    
    try:
        if headless:
            # Run for 10 seconds of simulated time
            await asyncio.sleep(10.0)
        else:
            # Run until user quits or 30 seconds pass
            start_time = asyncio.get_event_loop().time()
            max_duration = 30.0
            
            while animator.is_running:
                await asyncio.sleep(0.1)
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_duration:
                    break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Stop all components
        if animator:
            animator.stop()
            await animator.wait_for_stop()
        
        runner.stop()
        await runner.wait_for_stop()
        
        world.stop_physics()
        await world.wait_for_physics_stop()
        
        if framer:
            framer.close()
        
        # Print final statistics
        final_state = world.get_state()
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Simulation time: {final_state.info['simulation_time']:.2f}s")
        print(f"  Control steps: {steps}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final sound energy: {final_state.observation.sound_energy:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    headless = len(sys.argv) > 1 and sys.argv[1] == "--headless"
    asyncio.run(main(headless=headless))
