"""Demo script running all three agents in two body configurations simultaneously for
side-by-side comparison."""

import sys
import time
import pygame
import numpy as np
import multiprocessing
from typing import Dict, Any, Optional
from environment import Environment
from agents.pointing_agent import PointingAgent
from agents.approaching_agent import ApproachingAgent
from agents.tracking_agent import TrackingAgent
from experimenter import create_default_config, Logger
from experimenter.animator import Animator
from experimenter.plotter import create_plotter
from soundflower import SoundFlower


# Command types for inter-process communication
CMD_STEP = "step"
CMD_PAUSE = "pause"
CMD_RESUME = "resume"
CMD_FORWARD = "forward"
CMD_QUIT = "quit"


def simulation_worker(config, agent_class, agent_name, link_name, command_queue, data_queue):
    """
    Worker process that runs a single SoundFlower simulation.

    Args:
        config: Configuration object
        agent_class: Agent class to instantiate
        agent_name: Name of the agent
        link_name: Link configuration name
        command_queue: Queue for receiving commands from main process
        data_queue: Queue for sending render data to main process
    """
    try:
        # Create environment and agent in this process
        environment = Environment(config)
        agent = agent_class(link_lengths=np.array(config.link_lengths))
        logger = Logger(agent_name=agent_name)

        # Create TensorBoard plotter (non-blocking file I/O)
        plotter = create_plotter('tensorboard', config, agent_name)

        # Create SoundFlower with TensorBoard plotter
        soundflower = SoundFlower(
            config, environment, agent,
            logger=logger, animator=None, plotter=plotter
        )
        soundflower.start()

        paused = False
        running = True

        # Send initial render data
        render_data = environment.get_render_data()
        data_queue.put({
            'type': 'render',
            'agent_name': agent_name,
            'render_data': render_data,
            'config': config
        })

        while running:
            # Check for commands (non-blocking)
            try:
                while True:
                    cmd = command_queue.get_nowait()
                    if cmd['type'] == CMD_STEP:
                        pass  # Continue to step below
                    elif cmd['type'] == CMD_PAUSE:
                        paused = True
                    elif cmd['type'] == CMD_RESUME:
                        paused = False
                    elif cmd['type'] == CMD_FORWARD:
                        n_steps = cmd['n_steps']
                        for _ in range(n_steps):
                            soundflower.step()
                        # Send updated data after forward steps
                        render_data = environment.get_render_data()
                        data_queue.put({
                            'type': 'render',
                            'agent_name': agent_name,
                            'render_data': render_data,
                            'config': config
                        })
                    elif cmd['type'] == CMD_QUIT:
                        running = False
                        break
            except:
                pass  # No commands available

            if not running:
                break

            if not paused:
                soundflower.step()

                # Send render data periodically (every few steps to reduce queue overhead)
                if soundflower.step_count % 5 == 0:  # Send every 5 steps
                    render_data = environment.get_render_data()
                    data_queue.put({
                        'type': 'render',
                        'agent_name': agent_name,
                        'render_data': render_data,
                        'config': config
                    })

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

        soundflower.finish()
        plotter.finish()
    except Exception as e:
        print(f"Error in simulation worker {agent_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal completion
        data_queue.put({'type': 'done', 'agent_name': agent_name})


class MultiAgentDemo:
    """Runs multiple agents simultaneously with separate visualizations."""

    def __init__(self, headless: bool = False):
        """
        Initialize multi-agent demo.

        Args:
            headless: If True, run without animation
        """
        self.headless = headless

        # Create configurations for 2-link and 3-link arms
        config_2link = create_default_config(sound_source_angular_velocity=0.3)
        config_2link.num_links = 2
        config_2link.link_lengths = [0.5, 0.3]
        config_2link.link_masses = [6.0, 3.0]
        config_2link.joint_frictions = [1.0, 1.0]
        config_2link.__post_init__()

        config_3link = create_default_config(sound_source_angular_velocity=0.3)
        config_3link.num_links = 3
        config_3link.link_lengths = [0.5, 0.4, 0.3]
        config_3link.link_masses = [6.0, 6.0, 3.0]
        config_3link.joint_frictions = [1.0, 1.0, 1.0]
        config_3link.__post_init__()

        # Create process metadata and queues for six simulations (3 agents × 2 link configs)
        self.processes = []
        self.command_queues = []
        self.data_queue = multiprocessing.Queue()
        self.simulation_metadata = []  # Store metadata for each simulation
        self.render_data_cache = {}  # Cache render data from processes
        self.config_cache = {}  # Cache configs for rendering

        agent_configs = [
            ("PointingAgent", PointingAgent, (255, 100, 100)),  # Red
            ("ApproachingAgent", ApproachingAgent, (100, 255, 100)),  # Green
            ("TrackingAgent", TrackingAgent, (100, 100, 255)),  # Blue
        ]
        link_configs = [
            (config_2link, "2-link"),
            (config_3link, "3-link"),
        ]

        for link_config, link_name in link_configs:
            for agent_name, agent_class, color in agent_configs:
                full_agent_name = f"{agent_name}_{link_name}"

                # Create command queue for this simulation
                command_queue = multiprocessing.Queue()
                self.command_queues.append(command_queue)

                # Store metadata
                self.simulation_metadata.append({
                    'agent_name': full_agent_name,
                    'color': color,
                    'link_name': link_name,
                    'config': link_config
                })

                # Create and start worker process
                process = multiprocessing.Process(
                    target=simulation_worker,
                    args=(link_config, agent_class, full_agent_name, link_name, command_queue, self.data_queue)
                )
                process.start()
                self.processes.append(process)

        # Initialize animator for multi-panel visualization
        if not headless:
            # Collect all configs for the animator
            all_configs = []
            for metadata in self.simulation_metadata:
                config = metadata['config']  # Use config from metadata directly
                all_configs.append(config)  # Pass full config objects, Animator will convert them

            self.animator = Animator(all_configs, panels_per_row=3)


    def start(self):
        """Start all simulations and animator."""
        # Start the animator if not headless
        if not self.headless:
            self.animator.start()

    def step(self):
        """Step all simulations forward by sending step commands."""
        # Send step commands to all processes
        for command_queue in self.command_queues:
            command_queue.put({'type': CMD_STEP})

        # Process incoming data from worker processes
        self._process_data_queue()

    def _process_data_queue(self):
        """Process data from worker processes (non-blocking)."""
        while True:
            try:
                data = self.data_queue.get_nowait()
                if data['type'] == 'render':
                    agent_name = data['agent_name']
                    self.render_data_cache[agent_name] = data['render_data']
                    self.config_cache[agent_name] = data['config']
                elif data['type'] == 'done':
                    pass  # Process finished
            except:
                break  # No more data available

    def render(self):
        """Render all panels using the unified animator."""
        if self.headless:
            return

        # Process any pending data from worker processes
        self._process_data_queue()

        # Collect render data and agent names for all panels
        render_data_list = []
        agent_names = []

        for metadata in self.simulation_metadata:
            agent_name = metadata['agent_name']
            render_data = self.render_data_cache.get(agent_name)
            if render_data is not None:
                render_data_list.append(render_data)
                agent_names.append(agent_name)
            else:
                # No data available for this panel yet
                render_data_list.append(None)
                agent_names.append(agent_name)

        # Render all panels using the animator
        self.animator.render(render_data_list, agent_names)

    def forward(self, n_steps: int):
        """Step all simulations forward by N steps."""
        # Send forward commands to all processes
        for command_queue in self.command_queues:
            command_queue.put({'type': CMD_FORWARD, 'n_steps': n_steps})

        # Process incoming data (workers will send updated render data)
        self._process_data_queue()

    def handle_events(self, paused: bool = False):
        """Handle pygame events for all windows."""
        should_quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    # Send pause/resume commands to all processes
                    cmd_type = CMD_PAUSE if paused else CMD_RESUME
                    for command_queue in self.command_queues:
                        command_queue.put({'type': cmd_type})
                elif event.key == pygame.K_RETURN:
                    # Step forward 1 step (works regardless of pause state)
                    self.forward(1)
                elif event.key == pygame.K_1:
                    # Step forward 10 steps (works regardless of pause state)
                    self.forward(10)
                elif event.key == pygame.K_2:
                    # Step forward 100 steps (works regardless of pause state)
                    self.forward(100)
                elif event.key == pygame.K_3:
                    # Step forward 1000 steps (works regardless of pause state)
                    self.forward(1000)
                elif event.key == pygame.K_4:
                    # Step forward 10000 steps (works regardless of pause state)
                    self.forward(10000)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    should_quit = True

        return should_quit, paused

    def finish(self):
        """Finish all simulations."""
        # Send quit commands to all processes
        for command_queue in self.command_queues:
            try:
                command_queue.put({'type': CMD_QUIT})
            except:
                pass

        # Wait for processes to finish (with timeout)
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()

        if not self.headless:
            self.animator.finish()

    def run(self):
        """Run the multi-agent demo."""
        print("=" * 60)
        print("Sound Flower - Multi-Agent Comparison Demo")
        print("=" * 60)
        print("\nRunning six instances simultaneously:")
        print("  Row 1 (2-link arms):")
        print("    - PointingAgent (Red): Only orients toward sound source")
        print("    - ApproachingAgent (Green): Only minimizes distance")
        print("    - TrackingAgent (Blue): Both points and minimizes distance")
        print("  Row 2 (3-link arms):")
        print("    - PointingAgent (Red): Only orients toward sound source")
        print("    - ApproachingAgent (Green): Only minimizes distance")
        print("    - TrackingAgent (Blue): Both points and minimizes distance")
        print("=" * 60)
        print("\nTensorBoard logging enabled.")
        print("To view metrics, run: tensorboard --logdir=logs")
        print("=" * 60)

        if not self.headless:
            print("\nControls (applies to all agents):")
            print("  SPACE: Pause/Resume")
            print("  RETURN: Pause and step forward 1 step")
            print("  1: Pause and step forward 10 steps")
            print("  2: Pause and step forward 100 steps")
            print("  3: Pause and step forward 1000 steps")
            print("  4: Pause and step forward 10000 steps")
            print("  ESC or Q: Quit")
            print("=" * 60)

        self.start()

        # Give processes time to initialize
        time.sleep(0.1)

        paused = False
        should_quit = False

        try:
            while True:
                if not self.headless:
                    should_quit, paused = self.handle_events(paused)

                if should_quit:
                    break

                if paused:
                    time.sleep(0.01)
                    if not self.headless:
                        self.render()
                else:
                    self.step()
                    if not self.headless:
                        self.render()
                    else:
                        # In headless mode, still process data queue periodically
                        time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            self.finish()


def main():
    """Main entry point."""
    headless = len(sys.argv) > 1 and sys.argv[1] == "--headless"
    demo = MultiAgentDemo(headless=headless)
    demo.run()


if __name__ == "__main__":
    main()
