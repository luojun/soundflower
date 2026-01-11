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
        self.config = create_default_config(sound_source_angular_velocity=0.3)

        # Create three separate environments and agents
        self.soundflowers = []
        agent_configs = [
            ("PointingAgent", PointingAgent, (255, 100, 100)),  # Red
            ("ApproachingAgent", ApproachingAgent, (100, 255, 100)),  # Green
            ("TrackingAgent", TrackingAgent, (100, 100, 255)),  # Blue
        ]

        for agent_name, agent_class, color in agent_configs:
            environment = Environment(self.config)
            agent = agent_class()
            logger = Logger(agent_name=agent_name)

            soundflower = SoundFlower(
                self.config, environment, agent,
                logger=logger, animator=None  # We'll handle rendering ourselves
            )
            soundflower.agent_name = agent_name
            soundflower.agent_color = color
            self.soundflowers.append(soundflower)

        # Initialize pygame for multi-panel visualization
        if not headless:
            pygame.init()
            self.panel_width = 400
            self.panel_height = 400
            self.window_width = self.panel_width * 3
            self.window_height = self.panel_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Sound Flower - Multi-Agent Comparison")
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.wave_phase = 0.0

    def _world_to_screen(self, world_pos: np.ndarray, panel_offset_x: int) -> tuple:
        """Convert environment coordinates to screen coordinates for a panel."""
        world_size = max(self.config.circle_radius, sum(self.config.link_lengths)) * 1.2
        screen_x = panel_offset_x + self.panel_width // 2 + int(world_pos[0] * self.panel_width / (2 * world_size))
        screen_y = self.panel_height // 2 - int(world_pos[1] * self.panel_height / (2 * world_size))
        return screen_x, screen_y

    def _world_to_screen_radius(self, world_radius: float) -> int:
        """Convert environment radius to screen radius."""
        world_size = max(self.config.circle_radius, sum(self.config.link_lengths)) * 1.2
        return int(world_radius * min(self.panel_width, self.panel_height) / (2 * world_size))

    def _render_panel(self, soundflower: SoundFlower, panel_index: int):
        """Render a single panel for one agent."""
        panel_offset_x = panel_index * self.panel_width
        render_data = soundflower.environment.get_render_data()

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
                        (panel_offset_x, 0, self.panel_width, self.panel_height))

        # Draw grid
        world_size = max(self.config.circle_radius, sum(self.config.link_lengths)) * 1.2
        grid_spacing = world_size / 5
        screen_spacing = int(grid_spacing * min(self.panel_width, self.panel_height) / (2 * world_size))
        center_x = panel_offset_x + self.panel_width // 2
        center_y = self.panel_height // 2

        for i in range(-5, 6):
            x = center_x + i * screen_spacing
            pygame.draw.line(self.screen, colors['grid'], (x, 0), (x, self.panel_height), 1)
        for i in range(-5, 6):
            y = center_y + i * screen_spacing
            pygame.draw.line(self.screen, colors['grid'],
                           (panel_offset_x, y), (panel_offset_x + self.panel_width, y), 1)

        # Draw circle boundary
        center = (center_x, center_y)
        radius = self._world_to_screen_radius(self.config.circle_radius)
        pygame.draw.circle(self.screen, colors['circle'], center, radius, 2)

        # Draw arm
        joint_positions = render_data['joint_positions']
        end_effector_pos = render_data['end_effector_pos']
        for i in range(len(joint_positions) - 1):
            start = self._world_to_screen(joint_positions[i], panel_offset_x)
            end = self._world_to_screen(joint_positions[i + 1], panel_offset_x)
            pygame.draw.line(self.screen, colors['arm'], start, end, 5)

        if len(joint_positions) > 0:
            start = self._world_to_screen(joint_positions[-1], panel_offset_x)
            end = self._world_to_screen(end_effector_pos, panel_offset_x)
            pygame.draw.line(self.screen, colors['arm'], start, end, 5)

        # Draw joints
        for i, joint_pos in enumerate(joint_positions):
            screen_pos = self._world_to_screen(joint_pos, panel_offset_x)
            if i == 0:
                pygame.draw.circle(self.screen, colors['joint'], screen_pos, 8)
            else:
                pygame.draw.circle(self.screen, colors['joint'], screen_pos, 6)

        # Draw end effector
        screen_pos = self._world_to_screen(end_effector_pos, panel_offset_x)
        pygame.draw.circle(self.screen, colors['end_effector'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)

        # Draw sound sources
        sound_source_positions = render_data.get('sound_source_positions', [])
        for source_pos in sound_source_positions:
            screen_pos = self._world_to_screen(source_pos, panel_offset_x)
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
        self.screen.blit(text_surface, (panel_offset_x + 10, 10))

        sound_intensity = render_data.get('sound_intensity', 0.0)
        text_surface = self.small_font.render(f"Intensity: {sound_intensity:.4f}", True, colors['text'])
        self.screen.blit(text_surface, (panel_offset_x + 10, 35))

        if sound_source_positions:
            distances = [np.linalg.norm(end_effector_pos - sp) for sp in sound_source_positions]
            distance_to_source = min(distances)
            text_surface = self.small_font.render(f"Distance: {distance_to_source:.3f}", True, colors['text'])
            self.screen.blit(text_surface, (panel_offset_x + 10, 53))

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

    def handle_events(self):
        """Handle pygame events for all windows."""
        should_quit = False
        paused = False

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
        print("\nRunning three agents simultaneously:")
        print("  - PointingAgent (Red): Only orients toward sound source")
        print("  - ApproachingAgent (Green): Only minimizes distance")
        print("  - TrackingAgent (Blue): Both points and minimizes distance")
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
                    should_quit, paused = self.handle_events()

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

