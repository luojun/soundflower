"""Demo script running all three agents simultaneously for side-by-side comparison."""

import sys
import time
import math
import pygame
import numpy as np
from environment import Environment
from agents.pointing_agent import PointingAgent
from agents.approaching_agent import ApproachingAgent
from agents.tracking_agent import TrackingAgent
from experimenter import create_default_config, Logger
from experimenter.plotter import Plotter
from soundflower import SoundFlower


class MultiAgentDemo:
    """Runs multiple agents simultaneously with separate visualizations."""

    def __init__(self, headless: bool = False):
        """
        Initialize multi-agent demo.

        Args:
            headless: If True, run without animation
        """
        self.headless = headless

        # Create shared plotter instance for all agents
        shared_plotter = None
        if not headless:
            # Create first plotter instance (will become shared)
            base_config = create_default_config(sound_source_angular_velocity=0.3)
            shared_plotter = Plotter(base_config, agent_name=None, shared=True)

        # Create configurations for 2-link and 3-link arms
        config_2link = create_default_config(sound_source_angular_velocity=0.3)
        config_2link.num_links = 2
        config_2link.link_lengths = [0.5, 0.3]
        config_2link.link_masses = [1.0, 0.6]
        config_2link.joint_frictions = [0.1, 0.15]
        config_2link.__post_init__()

        config_3link = create_default_config(sound_source_angular_velocity=0.3)
        config_3link.num_links = 3
        config_3link.link_lengths = [0.5, 0.4, 0.3]
        config_3link.link_masses = [1.0, 0.8, 0.6]
        config_3link.joint_frictions = [0.1, 0.12, 0.15]
        config_3link.__post_init__()

        # Create six separate environments and agents (3 agents × 2 link configs)
        self.soundflowers = []
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
                environment = Environment(link_config)
                full_agent_name = f"{agent_name}_{link_name}"
                # Pass link_lengths and min_distance_to_source to agents
                if agent_class == ApproachingAgent:
                    agent = agent_class(
                        link_lengths=np.array(link_config.link_lengths),
                        min_distance_to_source=link_config.min_distance_to_source
                    )
                elif agent_class == TrackingAgent:
                    agent = agent_class(
                        link_lengths=np.array(link_config.link_lengths),
                        min_distance_to_source=link_config.min_distance_to_source
                    )
                else:
                    agent = agent_class(link_lengths=np.array(link_config.link_lengths))
                logger = Logger(agent_name=full_agent_name)

                # Create plotter instance for this agent (will use shared instance)
                plotter = None
                if shared_plotter:
                    plotter = Plotter(link_config, agent_name=full_agent_name, shared=True)

                soundflower = SoundFlower(
                    link_config, environment, agent,
                    logger=logger, animator=None, plotter=plotter  # We'll handle rendering ourselves
                )
                soundflower.agent_name = full_agent_name
                soundflower.agent_color = color
                soundflower.link_config_name = link_name
                self.soundflowers.append(soundflower)

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

    def _render_panel(self, soundflower: SoundFlower, panel_index: int):
        """Render a single panel for one agent."""
        # Calculate panel position (2 rows × 3 columns)
        row = panel_index // self.panels_per_row
        col = panel_index % self.panels_per_row
        panel_offset_x = col * self.panel_width
        panel_offset_y = row * self.panel_height
        render_data = soundflower.environment.get_render_data()
        config = soundflower.environment.config

        # Colors
        colors = {
            'background': (20, 20, 30),
            'circle': (100, 100, 120),
            'arm': soundflower.agent_color,
            'joint': tuple(min(255, c + 50) for c in soundflower.agent_color),
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
        text_surface = self.font.render(soundflower.agent_name, True, colors['text'])
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
        """Start all simulations."""
        for soundflower in self.soundflowers:
            soundflower.start()

    def step(self):
        """Step all simulations forward."""
        for soundflower in self.soundflowers:
            soundflower.step()

        # Update wave phase for animation
        if not self.headless:
            self.wave_phase += 0.1
            if self.wave_phase > 2 * np.pi:
                self.wave_phase -= 2 * np.pi

    def render(self):
        """Render all panels."""
        if self.headless:
            return

        # Clear screen
        self.screen.fill((20, 20, 30))

        # Render each panel
        for i, soundflower in enumerate(self.soundflowers):
            self._render_panel(soundflower, i)

        # Update display
        pygame.display.flip()

    def forward(self, n_steps: int):
        """Step all simulations forward by N steps."""
        for soundflower in self.soundflowers:
            soundflower.forward(n_steps)

    def handle_events(self, paused: bool = False):
        """Handle pygame events for all windows."""
        should_quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RETURN:
                    paused = True
                    self.forward(1)
                elif event.key == pygame.K_1:
                    paused = True
                    self.forward(10)
                elif event.key == pygame.K_2:
                    paused = True
                    self.forward(100)
                elif event.key == pygame.K_3:
                    paused = True
                    self.forward(1000)
                elif event.key == pygame.K_4:
                    paused = True
                    self.forward(10000)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    should_quit = True

        return should_quit, paused

    def finish(self):
        """Finish all simulations."""
        for soundflower in self.soundflowers:
            soundflower.finish()
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

