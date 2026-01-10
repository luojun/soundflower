"""Demo script for Sound Flower environment with optional animation."""

import sys, time
import pygame
from environment import Environment
from agents.heuristic_agent import HeuristicAgent
from experimenter import create_default_config, Logger
from experimenter.animator import Animator
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

    config = create_default_config(sound_source_angular_velocity=0.3)
    environment = Environment(config)
    agent = HeuristicAgent()
    logger = Logger()
    animator = None
    if not headless:
        animator = Animator(config=config)

    soundflower = SoundFlower(config, environment, agent, logger=logger, animator=animator)

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
