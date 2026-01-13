"""Demo script for Sound Flower environment with optional animation."""

import sys, time
import pygame
import numpy as np
from environment import Environment
from agents.tracking_agent import TrackingAgent
from experimenter import create_default_config, Logger
from experimenter.animator import Animator
from experimenter.plotter import Plotter
from soundflower import SoundFlower


def main(headless: bool = False):
    """Run demo with optional animation."""
    print("=" * 60)
    print("Sound Flower - Robotic Arm Sound Source Tracking Demo")
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

    print("=" * 60)

    # Create 3-link configuration
    config = create_default_config(sound_source_angular_velocity=0.3)
    config.num_links = 3
    config.link_lengths = [0.5, 0.4, 0.3]
    config.link_masses = [6.0, 6.0, 3.0]
    config.joint_frictions = [1.0, 1.0, 1.0]
    config.__post_init__()  # Ensure configuration is validated
    environment = Environment(config)
    agent = TrackingAgent(
        link_lengths=np.array(config.link_lengths)
    )
    # Identify logger and plotter by agent type
    agent_name = agent.__class__.__name__
    logger = Logger(agent_name=agent_name)
    animator = None
    plotter = None
    if not headless:
        animator = Animator(config=config)
        # Create non-shared plotter for single-agent demo
        plotter = Plotter(config, agent_name=agent_name, shared=False)

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
