"""Demo script running all three agents simultaneously for side-by-side comparison."""

import sys
import time
import math
import pygame
import numpy as np
import multiprocessing
from typing import Dict, Any, Optional
from environment import Environment
from agents.pointing_agent import PointingAgent
from agents.approaching_agent import ApproachingAgent
from agents.tracking_agent import TrackingAgent
from experimenter import create_default_config, Logger
from experimenter.plotter import Plotter
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
        data_queue: Queue for sending render/plotter data to main process
    """
    try:
        # Create environment and agent in this process
        environment = Environment(config)
        agent = agent_class(link_lengths=np.array(config.link_lengths))
        logger = Logger(agent_name=agent_name)

        # Don't create plotter in worker - main process handles plotting
        soundflower = SoundFlower(
            config, environment, agent,
            logger=logger, animator=None, plotter=None
        )
        soundflower.start()

        paused = False
        running = True
        time_since_last_plot = 0.0
        plotting_period = 1.0 / config.plotting_frequency if config.plotting_frequency > 0 else float('inf')

        # Send initial render data
        render_data = environment.get_render_data()
        environment_state = environment.get_state()
        data_queue.put({
            'type': 'render',
            'agent_name': agent_name,
            'render_data': render_data,
            'config': config
        })

        data_queue.put({
            'type': 'plotter',
            'agent_name': agent_name,
            'step_count': soundflower.step_count,
            'reward': environment_state.reward,
            'sound_energy': environment_state.observation.sound_energy,
            'cumulative_reward': soundflower.cumulative_reward,
            'cumulative_sound_energy': soundflower.cumulative_sound_energy
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
                        environment_state = environment.get_state()
                        data_queue.put({
                            'type': 'render',
                            'agent_name': agent_name,
                            'render_data': render_data,
                            'config': config
                        })
                        data_queue.put({
                            'type': 'plotter',
                            'agent_name': agent_name,
                            'step_count': soundflower.step_count,
                            'reward': environment_state.reward,
                            'sound_energy': environment_state.observation.sound_energy,
                            'cumulative_reward': soundflower.cumulative_reward,
                            'cumulative_sound_energy': soundflower.cumulative_sound_energy
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
                time_since_last_plot += config.dt

                # Send render data periodically (every few steps to reduce queue overhead)
                if soundflower.step_count % 5 == 0:  # Send every 5 steps
                    render_data = environment.get_render_data()
                    data_queue.put({
                        'type': 'render',
                        'agent_name': agent_name,
                        'render_data': render_data,
                        'config': config
                    })

                # Send plotter data at plotting frequency
                if time_since_last_plot >= plotting_period:
                    environment_state = environment.get_state()
                    data_queue.put({
                        'type': 'plotter',
                        'agent_name': agent_name,
                        'step_count': soundflower.step_count,
                        'reward': environment_state.reward,
                        'sound_energy': environment_state.observation.sound_energy,
                        'cumulative_reward': soundflower.cumulative_reward,
                        'cumulative_sound_energy': soundflower.cumulative_sound_energy
                    })
                    time_since_last_plot = 0.0

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

        soundflower.finish()
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

        # Create shared plotter instance for all agents (only in main process)
        self.shared_plotter = None
        if not headless:
            # Create first plotter instance (will become shared)
            base_config = create_default_config(sound_source_angular_velocity=0.3)
            self.shared_plotter = Plotter(base_config, agent_name=None, shared=True)

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
        self.plotters = {}  # Store plotter instances per agent
        self.plotter_data_buffer = {}  # Buffer plotter data for batch processing
        self.last_plot_update_time = time.time()  # Initialize to current time
        self.plot_update_interval = 0.5  # Update plots at most every 0.5 seconds

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

                # Create plotter instance for this agent (in main process)
                if self.shared_plotter:
                    plotter = Plotter(link_config, agent_name=full_agent_name, shared=True)
                    self.plotters[full_agent_name] = plotter

                # Create and start worker process
                process = multiprocessing.Process(
                    target=simulation_worker,
                    args=(link_config, agent_class, full_agent_name, link_name, command_queue, self.data_queue)
                )
                process.start()
                self.processes.append(process)

        # Initialize pygame for multi-panel visualization
        if not headless:
            pygame.init()
            self.panel_width = 400
            self.panel_height = 400
            self.panels_per_row = 3
            self.num_rows = 2
            self.window_width = self.panel_width * self.panels_per_row
            self.window_height = self.panel_height * self.num_rows
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Sound Flower - Multi-Agent Comparison (2-link & 3-link)")
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.wave_phase = 0.0

    def _world_to_screen(self, world_pos: np.ndarray, panel_offset_x: int, panel_offset_y: int, config) -> tuple:
        """Convert environment coordinates to screen coordinates for a panel."""
        world_size = max(config.circle_radius, sum(config.link_lengths)) * 1.2
        screen_x = panel_offset_x + self.panel_width // 2 + int(world_pos[0] * self.panel_width / (2 * world_size))
        screen_y = panel_offset_y + self.panel_height // 2 - int(world_pos[1] * self.panel_height / (2 * world_size))
        return screen_x, screen_y

    def _world_to_screen_radius(self, world_radius: float, config) -> int:
        """Convert environment radius to screen radius."""
        world_size = max(config.circle_radius, sum(config.link_lengths)) * 1.2
        return int(world_radius * min(self.panel_width, self.panel_height) / (2 * world_size))

    def _render_panel(self, panel_index: int):
        """Render a single panel for one agent."""
        # Get metadata and cached render data
        metadata = self.simulation_metadata[panel_index]
        agent_name = metadata['agent_name']
        render_data = self.render_data_cache.get(agent_name)
        config = self.config_cache.get(agent_name)

        if render_data is None or config is None:
            return  # No data available yet

        # Calculate panel position (2 rows × 3 columns)
        row = panel_index // self.panels_per_row
        col = panel_index % self.panels_per_row
        panel_offset_x = col * self.panel_width
        panel_offset_y = row * self.panel_height

        # Colors
        colors = {
            'background': (20, 20, 30),
            'circle': (100, 100, 120),
            'arm': metadata['color'],
            'joint': tuple(min(255, c + 50) for c in metadata['color']),
            'end_effector': (100, 255, 100),
            'sound_source': (255, 100, 100),
            'sound_wave': (255, 150, 150),
            'text': (255, 255, 255),
            'grid': (40, 40, 50)
        }

        # Draw panel background
        pygame.draw.rect(self.screen, colors['background'],
                        (panel_offset_x, panel_offset_y, self.panel_width, self.panel_height))

        # Draw grid
        world_size = max(config.circle_radius, sum(config.link_lengths)) * 1.2
        grid_spacing = world_size / 5
        screen_spacing = int(grid_spacing * min(self.panel_width, self.panel_height) / (2 * world_size))
        center_x = panel_offset_x + self.panel_width // 2
        center_y = panel_offset_y + self.panel_height // 2

        for i in range(-5, 6):
            x = center_x + i * screen_spacing
            pygame.draw.line(self.screen, colors['grid'], (x, panel_offset_y), (x, panel_offset_y + self.panel_height), 1)
        for i in range(-5, 6):
            y = center_y + i * screen_spacing
            pygame.draw.line(self.screen, colors['grid'],
                           (panel_offset_x, y), (panel_offset_x + self.panel_width, y), 1)

        # Draw circle boundary
        center = (center_x, center_y)
        radius = self._world_to_screen_radius(config.circle_radius, config)
        pygame.draw.circle(self.screen, colors['circle'], center, radius, 2)

        # Draw arm
        joint_positions = render_data['joint_positions']
        end_effector_pos = render_data['end_effector_pos']
        for i in range(len(joint_positions) - 1):
            start = self._world_to_screen(joint_positions[i], panel_offset_x, panel_offset_y, config)
            end = self._world_to_screen(joint_positions[i + 1], panel_offset_x, panel_offset_y, config)
            pygame.draw.line(self.screen, colors['arm'], start, end, 5)

        if len(joint_positions) > 0:
            start = self._world_to_screen(joint_positions[-1], panel_offset_x, panel_offset_y, config)
            end = self._world_to_screen(end_effector_pos, panel_offset_x, panel_offset_y, config)
            pygame.draw.line(self.screen, colors['arm'], start, end, 5)

        # Draw joints
        for i, joint_pos in enumerate(joint_positions):
            screen_pos = self._world_to_screen(joint_pos, panel_offset_x, panel_offset_y, config)
            if i == 0:
                pygame.draw.circle(self.screen, colors['joint'], screen_pos, 8)
            else:
                pygame.draw.circle(self.screen, colors['joint'], screen_pos, 6)

        # Draw end effector
        screen_pos = self._world_to_screen(end_effector_pos, panel_offset_x, panel_offset_y, config)
        pygame.draw.circle(self.screen, colors['end_effector'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)

        # Draw sound sources
        sound_source_positions = render_data.get('sound_source_positions', [])
        for source_pos in sound_source_positions:
            screen_pos = self._world_to_screen(source_pos, panel_offset_x, panel_offset_y, config)
            # Draw sound waves
            num_waves = 3
            for i in range(num_waves):
                wave_radius = 15 + 10 * i + 5 * math.sin(self.wave_phase + i * math.pi / 2)
                alpha = max(0, 255 - 80 * i - int(50 * abs(math.sin(self.wave_phase))))
                color = tuple(min(255, c + alpha // 3) for c in colors['sound_wave'][:3])
                pygame.draw.circle(self.screen, color, screen_pos, int(wave_radius), 2)
            # Draw sound source
            pygame.draw.circle(self.screen, colors['sound_source'], screen_pos, 8)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 8, 2)

        # Draw agent name and info
        text_surface = self.font.render(agent_name, True, colors['text'])
        self.screen.blit(text_surface, (panel_offset_x + 10, panel_offset_y + 10))

        sound_intensity = render_data.get('sound_intensity', 0.0)
        text_surface = self.small_font.render(f"Intensity: {sound_intensity:.4f}", True, colors['text'])
        self.screen.blit(text_surface, (panel_offset_x + 10, panel_offset_y + 35))

        if sound_source_positions:
            distances = [np.linalg.norm(end_effector_pos - sp) for sp in sound_source_positions]
            distance_to_source = min(distances)
            text_surface = self.small_font.render(f"Distance: {distance_to_source:.3f}", True, colors['text'])
            self.screen.blit(text_surface, (panel_offset_x + 10, panel_offset_y + 53))

    def start(self):
        """Start all simulations (processes already started in __init__)."""
        # Processes are started in __init__, just need to initialize plotter
        if self.shared_plotter:
            self.shared_plotter.start()

    def step(self):
        """Step all simulations forward by sending step commands."""
        # Send step commands to all processes
        for command_queue in self.command_queues:
            command_queue.put({'type': CMD_STEP})

        # Process incoming data from worker processes
        self._process_data_queue()

        # Update wave phase for animation
        if not self.headless:
            self.wave_phase += 0.1
            if self.wave_phase > 2 * np.pi:
                self.wave_phase -= 2 * np.pi

    def _process_data_queue(self):
        """Process data from worker processes (non-blocking)."""
        current_time = time.time()
        should_update_plots = (current_time - self.last_plot_update_time) >= self.plot_update_interval

        while True:
            try:
                data = self.data_queue.get_nowait()
                if data['type'] == 'render':
                    agent_name = data['agent_name']
                    self.render_data_cache[agent_name] = data['render_data']
                    self.config_cache[agent_name] = data['config']
                elif data['type'] == 'plotter':
                    # Buffer plotter data for batch processing
                    agent_name = data['agent_name']
                    if agent_name in self.plotters:
                        self.plotter_data_buffer[agent_name] = data
                elif data['type'] == 'done':
                    pass  # Process finished
            except:
                break  # No more data available

        # Batch update plots if enough time has passed
        if should_update_plots and self.plotter_data_buffer:
            for agent_name, plot_data in self.plotter_data_buffer.items():
                if agent_name in self.plotters:
                    self.plotters[agent_name].step(
                        plot_data['step_count'],
                        plot_data['reward'],
                        plot_data['sound_energy'],
                        plot_data['cumulative_reward'],
                        plot_data['cumulative_sound_energy']
                    )
            self.plotter_data_buffer.clear()
            self.last_plot_update_time = current_time

    def render(self):
        """Render all panels."""
        if self.headless:
            return

        # Process any pending data from worker processes
        self._process_data_queue()

        # Clear screen
        self.screen.fill((20, 20, 30))

        # Render each panel
        for i in range(len(self.simulation_metadata)):
            self._render_panel(i)

        # Update display
        pygame.display.flip()

    def forward(self, n_steps: int):
        """Step all simulations forward by N steps."""
        # Send forward commands to all processes
        for command_queue in self.command_queues:
            command_queue.put({'type': CMD_FORWARD, 'n_steps': n_steps})

        # Process incoming data (workers will send updated render/plotter data)
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

        if self.shared_plotter:
            self.shared_plotter.finish()

        if not self.headless:
            pygame.quit()

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

