"""Pygame framer for the new decoupled simulation architecture."""

import pygame
import numpy as np
import math
from typing import Dict, Any, Optional


class PygameFramer:
    """Pygame framer that works with the decoupled simulation loop."""

    def __init__(self, configs, panels_per_row: int = 1, window_size: Optional[tuple] = None):
        """
        Initialize visualizer supporting single or multiple panels.

        Args:
            configs: Single config dict or list of config dicts (one per panel)
            panels_per_row: Number of panels per row (default 1 for single panel)
            window_size: Optional window size, auto-calculated if None
        """
        # Handle single config as 1-element list for unified interface
        if not isinstance(configs, list):
            configs = [configs]
        self.configs = configs
        self.num_panels = len(configs)
        self.panels_per_row = panels_per_row
        self.num_rows = (self.num_panels + panels_per_row - 1) // panels_per_row

        # Auto-calculate window size if not provided
        if window_size is None:
            panel_width, panel_height = 400, 400
            window_size = (panel_width * panels_per_row, panel_height * self.num_rows)
        self.window_size = window_size

        # Calculate panel dimensions
        self.panel_width = self.window_size[0] // panels_per_row
        self.panel_height = self.window_size[1] // self.num_rows

        # Pre-calculate world sizes and bounds for each panel
        self.world_sizes = []
        for config in self.configs:
            circle_radius = config['circle_radius']
            link_lengths = config['link_lengths']
            max_reach = sum(link_lengths)
            world_size = max(circle_radius, max_reach) * 1.2
            self.world_sizes.append(world_size)

        # Agent color mapping for multi-panel visualization
        self._agent_colors = {
            'PointingAgent': (255, 100, 100),    # Red
            'ApproachingAgent': (100, 255, 100), # Green
            'TrackingAgent': (100, 100, 255),    # Blue
        }

        # Colors
        self.colors = {
            'background': (20, 20, 30),
            'circle': (100, 100, 120),
            'arm': (100, 150, 255),
            'joint': (150, 200, 255),
            'end_effector': (100, 255, 100),
            'sound_source': (255, 100, 100),
            'sound_wave': (255, 150, 150),
            'text': (255, 255, 255),
            'grid': (40, 40, 50)
        }

        # Animation state
        self.running = True  # Start as running
        self.paused = False
        self.wave_phase = 0.0

        # Current render data
        self.current_render_data: Optional[Dict[str, Any]] = None

    def _world_to_screen(self, world_pos: np.ndarray, panel_index: int) -> tuple:
        """Convert environment coordinates to screen coordinates for a specific panel."""
        world_size = self.world_sizes[panel_index]

        # Calculate panel position
        row = panel_index // self.panels_per_row
        col = panel_index % self.panels_per_row
        panel_offset_x = col * self.panel_width
        panel_offset_y = row * self.panel_height

        # Convert to panel-local coordinates
        screen_x = panel_offset_x + self.panel_width // 2 + int(world_pos[0] * self.panel_width / (2 * world_size))
        screen_y = panel_offset_y + self.panel_height // 2 - int(world_pos[1] * self.panel_height / (2 * world_size))
        return screen_x, screen_y

    def _world_to_screen_radius(self, world_radius: float, panel_index: int) -> int:
        """Convert environment radius to screen radius for a specific panel."""
        world_size = self.world_sizes[panel_index]
        return int(world_radius * min(self.panel_width, self.panel_height) / (2 * world_size))

    def _get_agent_color(self, agent_name: str):
        """Get color for agent type."""
        if '_' in agent_name:
            base_name = agent_name.split('_')[0]
        else:
            base_name = agent_name
        return self._agent_colors.get(base_name, self.colors['arm'])

    def _draw_grid(self, panel_index: int):
        """Draw grid background for a specific panel."""
        world_size = self.world_sizes[panel_index]
        grid_spacing = world_size / 5
        screen_spacing = int(grid_spacing * min(self.panel_width, self.panel_height) / (2 * world_size))

        # Calculate panel position
        row = panel_index // self.panels_per_row
        col = panel_index % self.panels_per_row
        panel_offset_x = col * self.panel_width
        panel_offset_y = row * self.panel_height

        center_x = panel_offset_x + self.panel_width // 2
        center_y = panel_offset_y + self.panel_height // 2

        # Vertical lines
        for i in range(-5, 6):
            x = center_x + i * screen_spacing
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x, panel_offset_y), (x, panel_offset_y + self.panel_height), 1)

        # Horizontal lines
        for i in range(-5, 6):
            y = center_y + i * screen_spacing
            pygame.draw.line(self.screen, self.colors['grid'],
                           (panel_offset_x, y), (panel_offset_x + self.panel_width, y), 1)

    def _draw_circle_boundary(self, panel_index: int):
        """Draw the circular boundary for a specific panel."""
        config = self.configs[panel_index]
        center = self._world_to_screen(np.array([0, 0]), panel_index)
        radius = self._world_to_screen_radius(config['circle_radius'], panel_index)
        pygame.draw.circle(self.screen, self.colors['circle'], center, radius, 2)

    def _draw_arm(self, joint_positions: np.ndarray, end_effector_pos: np.ndarray, panel_index: int, arm_color=None):
        """Draw the robotic arm for a specific panel."""
        if arm_color is None:
            arm_color = self.colors['arm']

        # Draw links
        for i in range(len(joint_positions) - 1):
            start = self._world_to_screen(joint_positions[i], panel_index)
            end = self._world_to_screen(joint_positions[i + 1], panel_index)
            pygame.draw.line(self.screen, arm_color, start, end, 5)

        # Draw final link to end effector
        if len(joint_positions) > 0:
            start = self._world_to_screen(joint_positions[-1], panel_index)
            end = self._world_to_screen(end_effector_pos, panel_index)
            pygame.draw.line(self.screen, arm_color, start, end, 5)

        # Draw joints
        joint_color = tuple(min(255, c + 50) for c in arm_color) if isinstance(arm_color, tuple) else self.colors['joint']
        for i, joint_pos in enumerate(joint_positions):
            screen_pos = self._world_to_screen(joint_pos, panel_index)
            if i == 0:
                pygame.draw.circle(self.screen, joint_color, screen_pos, 8)
            else:
                pygame.draw.circle(self.screen, joint_color, screen_pos, 6)

        # Draw end effector
        screen_pos = self._world_to_screen(end_effector_pos, panel_index)
        pygame.draw.circle(self.screen, self.colors['end_effector'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)

    def _draw_sound_source(self, source_pos: np.ndarray, wave_phase: float, panel_index: int):
        """Draw a sound source with animated waves for a specific panel."""
        screen_pos = self._world_to_screen(source_pos, panel_index)

        # Draw sound waves
        num_waves = 3
        for i in range(num_waves):
            wave_radius = 15 + 10 * i + 5 * math.sin(wave_phase + i * math.pi / 2)
            alpha = max(0, 255 - 80 * i - int(50 * abs(math.sin(wave_phase))))
            color = tuple(min(255, c + alpha // 3) for c in self.colors['sound_wave'][:3])
            pygame.draw.circle(self.screen, color, screen_pos, int(wave_radius), 2)

        # Draw sound source
        pygame.draw.circle(self.screen, self.colors['sound_source'], screen_pos, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 8, 2)

    def _draw_text(self, text: str, pos: tuple, font=None, color=None):
        """Draw text on screen."""
        if font is None:
            font = self.font
        if color is None:
            color = self.colors['text']
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_info_panel(self, render_data: Dict[str, Any], panel_index: int, agent_name: str = ""):
        """Draw information panel for a specific panel."""
        # Calculate panel position
        row = panel_index // self.panels_per_row
        col = panel_index % self.panels_per_row
        panel_offset_x = col * self.panel_width
        panel_offset_y = row * self.panel_height

        y_offset = panel_offset_y + 10
        x_offset = panel_offset_x + 10

        sound_intensity = render_data.get('sound_intensity', 0.0)
        end_effector_pos = render_data.get('end_effector_pos', np.array([0, 0]))

        # Calculate distance to nearest source
        sound_source_positions = render_data.get('sound_source_positions', [])
        if sound_source_positions:
            distances = [np.linalg.norm(end_effector_pos - sp) for sp in sound_source_positions]
            distance_to_source = min(distances)
        else:
            distance_to_source = float('inf')

        # Agent name at top
        if agent_name:
            self._draw_text(agent_name, (x_offset, y_offset), self.small_font, (200, 200, 255))
            y_offset += 20

        self._draw_text(f"Intensity: {sound_intensity:.4f}", (x_offset, y_offset), self.small_font)
        y_offset += 18
        self._draw_text(f"Pos: ({end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f})",
                       (x_offset, y_offset), self.small_font)
        y_offset += 18
        self._draw_text(f"Dist: {distance_to_source:.3f}",
                       (x_offset, y_offset), self.small_font)

    def render_frame(self, render_data_list, agent_names=None) -> bool:
        """
        Callback function for visualization loop.

        Args:
            render_data_list: Single render_data dict or list of render_data dicts (one per panel)
            agent_names: Optional list of agent names for each panel

        Returns:
            True if should continue, False if should quit
        """
        # Handle single render_data as 1-element list for unified interface
        if not isinstance(render_data_list, list):
            render_data_list = [render_data_list]
        if agent_names is None:
            agent_names = [""] * len(render_data_list)

        # Update wave phase
        self.wave_phase += 0.1
        if self.wave_phase > 2 * math.pi:
            self.wave_phase -= 2 * math.pi

        # Store render data
        self.current_render_data = render_data_list

        # Skip rendering if paused
        if self.paused:
            pygame.display.flip()
            return True

        # Clear screen
        self.screen.fill(self.colors['background'])

        # Render each panel
        for panel_index, (render_data, agent_name) in enumerate(zip(render_data_list, agent_names)):
            # Get panel color based on agent type
            arm_color = self._get_agent_color(agent_name)

            # Draw panel background (optional visual separation)
            row = panel_index // self.panels_per_row
            col = panel_index % self.panels_per_row
            panel_x = col * self.panel_width
            panel_y = row * self.panel_height
            # Light border around each panel
            pygame.draw.rect(self.screen, (40, 40, 50),
                           (panel_x, panel_y, self.panel_width, self.panel_height), 1)

            # Always draw grid and circle boundary for each panel
            self._draw_grid(panel_index)
            self._draw_circle_boundary(panel_index)

            # Only draw dynamic elements if render_data is available
            if render_data is not None:
                # Draw arm for this panel
                joint_positions = render_data['joint_positions']
                end_effector_pos = render_data['end_effector_pos']
                self._draw_arm(joint_positions, end_effector_pos, panel_index, arm_color)

                # Draw sound sources for this panel
                sound_source_positions = render_data.get('sound_source_positions', [])
                for source_pos in sound_source_positions:
                    self._draw_sound_source(source_pos, self.wave_phase, panel_index)

                # Draw info panel for this panel
                self._draw_info_panel(render_data, panel_index, agent_name)
            else:
                # Draw agent name even if no data is available
                self._draw_info_panel({
                    'sound_intensity': 0.0,
                    'end_effector_pos': np.array([0, 0]),
                    'sound_source_positions': []
                }, panel_index, agent_name)

        # Draw global controls info (only once, bottom right)
        if self.num_panels > 1:
            controls_x = self.window_size[0] - 150
            controls_y = self.window_size[1] - 60
            self._draw_text("Controls:", (controls_x, controls_y), self.small_font, (150, 150, 150))
            self._draw_text("SPACE: Pause/Resume", (controls_x, controls_y + 18), self.small_font, (150, 150, 150))
            self._draw_text("ESC/Q: Quit", (controls_x, controls_y + 36), self.small_font, (150, 150, 150))

        # Update display
        pygame.display.flip()

    def start(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Sound Flower - Real-time Animation")
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def finish(self):
        """Close the visualizer."""
        pygame.quit()

