"""Pygame animator for the new decoupled simulation architecture."""

import pygame
import numpy as np
import math
from typing import Dict, Any, Optional


class PygameAnimator:
    """Pygame animator that works with the decoupled simulation loop."""
    
    def __init__(self, circle_radius: float, link_lengths: list,
                 window_size: tuple = (800, 800), fps: float = 60.0):
        """
        Initialize visualizer.
        
        Args:
            circle_radius: Radius of the environment circle
            link_lengths: Lengths of arm links
            window_size: Window size (width, height)
            fps: Target frame rate
        """
        self.circle_radius = circle_radius
        self.link_lengths = link_lengths
        self.window_size = window_size
        self.fps = fps
        
        # Calculate world bounds
        max_reach = sum(link_lengths)
        self.world_size = max(circle_radius, max_reach) * 1.2
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Sound Flower - Real-time Animation")
        self.clock = pygame.time.Clock()
        
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
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Animation state
        self.running = True  # Start as running
        self.paused = False
        self.wave_phase = 0.0
        
        # Current render data
        self.current_render_data: Optional[Dict[str, Any]] = None
    
    def _world_to_screen(self, world_pos: np.ndarray) -> tuple:
        """Convert world coordinates to screen coordinates."""
        screen_x = self.window_size[0] // 2 + int(world_pos[0] * self.window_size[0] / (2 * self.world_size))
        screen_y = self.window_size[1] // 2 - int(world_pos[1] * self.window_size[1] / (2 * self.world_size))
        return screen_x, screen_y
    
    def _world_to_screen_radius(self, world_radius: float) -> int:
        """Convert world radius to screen radius."""
        return int(world_radius * min(self.window_size) / (2 * self.world_size))
    
    def _draw_grid(self):
        """Draw grid background."""
        grid_spacing = self.world_size / 5
        screen_spacing = int(grid_spacing * min(self.window_size) / (2 * self.world_size))
        
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2
        
        # Vertical lines
        for i in range(-5, 6):
            x = center_x + i * screen_spacing
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (x, 0), (x, self.window_size[1]), 1)
        
        # Horizontal lines
        for i in range(-5, 6):
            y = center_y + i * screen_spacing
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (0, y), (self.window_size[0], y), 1)
    
    def _draw_circle_boundary(self):
        """Draw the circular boundary."""
        center = (self.window_size[0] // 2, self.window_size[1] // 2)
        radius = self._world_to_screen_radius(self.circle_radius)
        pygame.draw.circle(self.screen, self.colors['circle'], center, radius, 2)
    
    def _draw_arm(self, joint_positions: np.ndarray, end_effector_pos: np.ndarray):
        """Draw the robotic arm."""
        # Draw links
        for i in range(len(joint_positions) - 1):
            start = self._world_to_screen(joint_positions[i])
            end = self._world_to_screen(joint_positions[i + 1])
            pygame.draw.line(self.screen, self.colors['arm'], start, end, 5)
        
        # Draw final link to end effector
        if len(joint_positions) > 0:
            start = self._world_to_screen(joint_positions[-1])
            end = self._world_to_screen(end_effector_pos)
            pygame.draw.line(self.screen, self.colors['arm'], start, end, 5)
        
        # Draw joints
        for i, joint_pos in enumerate(joint_positions):
            screen_pos = self._world_to_screen(joint_pos)
            if i == 0:
                pygame.draw.circle(self.screen, self.colors['joint'], screen_pos, 8)
            else:
                pygame.draw.circle(self.screen, self.colors['joint'], screen_pos, 6)
        
        # Draw end effector
        screen_pos = self._world_to_screen(end_effector_pos)
        pygame.draw.circle(self.screen, self.colors['end_effector'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)
    
    def _draw_sound_source(self, source_pos: np.ndarray, wave_phase: float):
        """Draw a sound source with animated waves."""
        screen_pos = self._world_to_screen(source_pos)
        
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
    
    def _draw_info_panel(self, render_data: Dict[str, Any]):
        """Draw information panel."""
        y_offset = 10
        x_offset = 10
        
        step_count = render_data.get('step_count', 0)
        sound_energy = render_data.get('sound_energy', 0.0)
        end_effector_pos = render_data.get('end_effector_pos', np.array([0, 0]))
        simulation_time = render_data.get('simulation_time', 0.0)
        
        # Calculate distance to nearest source
        sound_source_positions = render_data.get('sound_source_positions', [])
        if sound_source_positions:
            distances = [np.linalg.norm(end_effector_pos - sp) for sp in sound_source_positions]
            distance_to_source = min(distances)
        else:
            distance_to_source = float('inf')
        
        self._draw_text(f"Step: {step_count}", (x_offset, y_offset))
        y_offset += 25
        self._draw_text(f"Time: {simulation_time:.2f}s", (x_offset, y_offset))
        y_offset += 25
        self._draw_text(f"Sound Energy: {sound_energy:.4f}", (x_offset, y_offset))
        y_offset += 25
        self._draw_text(f"End Effector: ({end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f})", 
                       (x_offset, y_offset))
        y_offset += 25
        self._draw_text(f"Distance to Source: {distance_to_source:.3f}", 
                       (x_offset, y_offset))
        y_offset += 30
        
        # Controls
        self._draw_text("Controls:", (x_offset, y_offset), self.small_font, (150, 150, 150))
        y_offset += 20
        self._draw_text("SPACE: Pause/Resume", (x_offset, y_offset), self.small_font, (150, 150, 150))
        y_offset += 18
        self._draw_text("ESC/Q: Quit", (x_offset, y_offset), self.small_font, (150, 150, 150))
    
    def render_callback(self, render_data: Dict[str, Any]) -> bool:
        """
        Callback function for visualization loop.
        
        Args:
            render_data: Render data from renderer
            
        Returns:
            True if should continue, False if should quit
        """
        # Update wave phase
        self.wave_phase += 0.1
        if self.wave_phase > 2 * math.pi:
            self.wave_phase -= 2 * math.pi
        
        # Store render data
        self.current_render_data = render_data
        
        # Handle events
        if not self.handle_events():
            self.running = False
            return False
        
        # Skip rendering if paused
        if self.paused:
            pygame.display.flip()
            self.clock.tick(self.fps)
            return True
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw circle boundary
        self._draw_circle_boundary()
        
        # Draw arm
        joint_positions = render_data['joint_positions']
        end_effector_pos = render_data['end_effector_pos']
        self._draw_arm(joint_positions, end_effector_pos)
        
        # Draw sound sources
        sound_source_positions = render_data.get('sound_source_positions', [])
        for source_pos in sound_source_positions:
            self._draw_sound_source(source_pos, self.wave_phase)
        
        # Draw info panel
        self._draw_info_panel(render_data)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns True if should continue."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True
    
    def close(self):
        """Close the visualizer."""
        pygame.quit()

