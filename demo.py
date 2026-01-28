"""Demo script for Sound Flower environment with optional animation."""

import sys, time
import pygame
from environment import Environment
from agents.continual_linear_rl_agent import ContinualLinearRLAgent
from experimenter import create_default_config, Logger
from experimenter.animator import Animator
from experimenter.plotter import create_plotter
from soundflower import SoundFlower


def main(headless: bool = False):
    """Run demo with optional animation."""
    print("=" * 60)
    print("Sound Flower - Continual Linear RL Baseline (Sensorimotor)")
    print("=" * 60)

    if headless:
        print("\nRunning in headless mode ...")
    else:
        print("\nRunning with animation ...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  RETURN: Pause and step forward 1 step")
        print("  1: Pause and step forward 10 steps")
        print("  2: Pause and step forward 100 steps")
        print("  3: Pause and step forward 1000 steps")
        print("  4: Pause and step forward 10000 steps")
        print("  ESC or Q: Quit")
        print("\nVariability Controls:")
        print("  S: Cycle number of active sources (1 -> 2 -> 3 -> 1)")
        print("  R: Increase orbit radius max, Shift+R: Decrease orbit radius min")
        print("  V: Increase orbital speed max, Shift+V: Decrease orbital speed min (more negative)")

    print("=" * 60)

    # Create 2-link sensorimotor configuration
    config = create_default_config(sound_source_angular_velocity=0.3)
    config.num_links = 2
    config.link_lengths = [0.6, 0.4]
    config.link_masses = [6.0, 6.0]
    config.joint_frictions = [1.0, 1.0]
    config.observation_mode = "sensorimotor"
    config.__post_init__()  # Ensure configuration is validated
    environment = Environment(config)
    agent = ContinualLinearRLAgent()
    # Identify logger and plotter by agent type
    agent_name = agent.__class__.__name__
    logger = Logger(agent_name=agent_name)
    animator = None
    plotter = None
    if not headless:
        animator = Animator(configs=config, window_size=(800, 800))
        # Create matplotlib plotter for single-agent demo
        plotter = create_plotter('matplotlib', config, agent_name)

    soundflower = SoundFlower(config, environment, agent, logger=logger, animator=animator, plotter=plotter)

    soundflower.start()
    paused = False
    should_quit = False

    try:
        while True:
            if animator and pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        should_quit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_RETURN:
                            paused = True
                            soundflower.forward(1)
                        elif event.key == pygame.K_1:
                            paused = True
                            soundflower.forward(10)
                        elif event.key == pygame.K_2:
                            paused = True
                            soundflower.forward(100)
                        elif event.key == pygame.K_3:
                            paused = True
                            soundflower.forward(1000)
                        elif event.key == pygame.K_4:
                            paused = True
                            soundflower.forward(10000)
                        elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                            should_quit = True
                        # Variability controls
                        elif event.key == pygame.K_s:
                            # Cycle number of active sources: 1 -> 2 -> 3 -> 1
                            current = environment.physics_engine.config.num_active_sources
                            new_num = (current % 3) + 1
                            environment.physics_engine.set_num_active_sources(new_num)
                            print(f"Active sources: {new_num}")
                        elif event.key == pygame.K_r:
                            # Increase orbit radius max
                            config = environment.physics_engine.config
                            new_max = config.orbit_radius_max + 0.1
                            environment.physics_engine.set_orbit_radius_range(
                                config.orbit_radius_min, new_max
                            )
                            print(f"Orbit radius range: [{config.orbit_radius_min:.2f}, {new_max:.2f}]")
                        elif event.key == pygame.K_r and event.mod & pygame.KMOD_SHIFT:
                            # Decrease orbit radius min
                            config = environment.physics_engine.config
                            new_min = max(0.1, config.orbit_radius_min - 0.1)
                            environment.physics_engine.set_orbit_radius_range(
                                new_min, config.orbit_radius_max
                            )
                            print(f"Orbit radius range: [{new_min:.2f}, {config.orbit_radius_max:.2f}]")
                        elif event.key == pygame.K_v:
                            # Increase orbital speed max
                            config = environment.physics_engine.config
                            new_max = config.orbital_speed_max + 0.1
                            environment.physics_engine.set_orbital_speed_range(
                                config.orbital_speed_min, new_max
                            )
                            print(f"Orbital speed range: [{config.orbital_speed_min:.2f}, {new_max:.2f}] rad/s")
                        elif event.key == pygame.K_v and event.mod & pygame.KMOD_SHIFT:
                            # Decrease orbital speed min (more negative = faster counterclockwise)
                            config = environment.physics_engine.config
                            new_min = config.orbital_speed_min - 0.1
                            environment.physics_engine.set_orbital_speed_range(
                                new_min, config.orbital_speed_max
                            )
                            print(f"Orbital speed range: [{new_min:.2f}, {config.orbital_speed_max:.2f}] rad/s")

            if should_quit:
                break

            if paused:
                time.sleep(0.01)
            else:
                soundflower.step()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        soundflower.finish()


if __name__ == "__main__":
    headless = len(sys.argv) > 1 and sys.argv[1] == "--headless"
    main(headless=headless)
