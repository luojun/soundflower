"""SoundFlower interface - orchestrates the SoundFlower mini-world."""

import asyncio
from typing import Optional, Callable, Dict, Any
from environment import Environment
from experimenter.animator import Animator
from experimenter.logger import Logger


class SoundFlower:
    """
    Runner interface for orchestrating the SoundFlower mini-world.

    Coordinates:
    - Environment (simulation state)
    - Agent (decision making)
    - Logger (logging, optional)
    - Animator (animation, optional)

    Runs at configurable frequencies:
    - Environment: physics runs as fast as possible with a configurable time step size <= 0.01s
    - agent: runs at control_frequency (0.1-100 Hz) in simulated time
    - Logger: runs at logging_frequency (0.01-100 Hz, optional) in simulated time
    - Animation: runs at animation_frequency (0.01-100 Hz, optional) in simulated time
    """

    def __init__(self, config, environment: Environment, agent, logger: Optional[Logger] = None,
                 animator: Optional[Animator] = None) -> None:
        """
        Initialize runner.

        Args:
            config: Configuration object
            environment: Environment instance
            agent: Agent with async select_action method
            logger: Logger with async logging
            animator: Animator with async animation
        """
        self.config = config
        self.environment = environment
        self.agent = agent
        self.logger = logger
        self.animator = animator

        if self.agent:
            self.control_period = 1.0 / config.control_frequency
            self.cumulative_reward = 0.0
        if self.logger:
            self.logging_period = 1.0 / config.logging_frequency
        if self.animator:
            self.animation_period = 1.0 / config.animation_frequency

        # Update simulation time
        self.simulation_time = 0.0
        self.step_count = 0


    def start(self):
        if self.logger:
            self.logger.log_config(self.config)

        if self.animator:
            self.animator.start()

        self.time_since_last_aciton = 0.0
        self.time_since_last_log = 0.0
        self.time_since_last_frame = 0.0


    def step(self):
        self.environment.step()

        if self.agent:
            self.time_since_last_aciton += self.config.dt

            if self.time_since_last_aciton >= self.control_period:
                environment_state = self.environment.get_state()
                self.cumulative_reward += environment_state.reward
                action = self.agent.select_action(environment_state.observation)
                self.environment.apply_action(action)

                self.time_since_last_aciton = 0.0

        if self.logger:
            self.time_since_last_log += self.config.dt

            if self.time_since_last_log >= self.logging_period:
                self.logger.log_step(self.environment.get_state())
                self.time_since_last_log = 0.0

        if self.animator:
            self.time_since_last_frame += self.config.dt

            if self.time_since_last_frame >= self.animation_period:
                self.animator.step(self.environment)
                self.time_since_last_frame = 0.0

        self.simulation_time += self.config.dt
        self.step_count += 1

    def forward(self, n_steps: int):
        """
        Step forward by N steps, then refresh logging and animation once.

        Args:
            n_steps: Number of steps to advance
        """
        # Step forward N times using the normal step() method.
        # Logging and animation still happen normally.
        for _ in range(n_steps):
            self.step()

        # Force refresh of logging and animation after all steps
        # (regardless of timing periods)
        if self.logger:
            self.logger.log_step(self.environment.get_state())
            self.time_since_last_log = 0.0

        if self.animator:
            self.animator.step(self.environment)
            self.time_since_last_frame = 0.0


    def finish(self):
        if self.animator:
            self.animator.finish()

        if self.logger:
            self.logger.log_final(self)
