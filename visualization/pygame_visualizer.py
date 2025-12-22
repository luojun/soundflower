"""Pygame-based real-time visualizer for Sound Flower environment."""

import pygame
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import math


class PygameVisualizer:
    """Real-time Pygame visualizer for the Sound Flower environment."""
    
    def __init__(self, circle_radius: float, link_lengths: List[float],
                 window_size: Tuple[int, int] = (800, 800),
                 fps: int = 60):
        """
        Initialize Pygame visualizer.
        
        Args:
            circle_radius: Radius of the environment circle
            link_lengths: Lengths of arm links
            window_size: Window size (width, height)
            fps: Target frames per second
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
        self.running = False
        self.paused = False
        
    def _world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_pos: World position (x, y)
            
        Returns:
            Screen position (x, y)
        """
        # Center of screen is origin
        screen_x = self.window_size[0] // 2 + int(world_pos[0] * self.window_size[0] / (2 * self.world_size))
        screen_y = self.window_size[1] // 2 - int(world_pos[1] * self.window_size[1] / (2 * self.world_size))  # Flip Y
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
        """
        Draw the robotic arm.
        
        Args:
            joint_positions: Array of joint positions
            end_effector_pos: End effector position
        """
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
                # Base joint (larger)
                pygame.draw.circle(self.screen, self.colors['joint'], screen_pos, 8)
            else:
                pygame.draw.circle(self.screen, self.colors['joint'], screen_pos, 6)
        
        # Draw end effector (microphone)
        screen_pos = self._world_to_screen(end_effector_pos)
        pygame.draw.circle(self.screen, self.colors['end_effector'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)
    
    def _draw_sound_source(self, source_pos: np.ndarray, strength: float = 1.0, 
                          wave_phase: float = 0.0):
        """
        Draw a sound source with animated waves.
        
        Args:
            source_pos: Sound source position
            strength: Sound source strength (affects wave size)
            wave_phase: Phase for wave animation (0-2π)
        """
        screen_pos = self._world_to_screen(source_pos)
        
        # Draw sound waves (animated concentric circles)
        num_waves = 3
        for i in range(num_waves):
            wave_radius = 15 + 10 * i + 5 * math.sin(wave_phase + i * math.pi / 2)
            alpha = max(0, 255 - 80 * i - int(50 * abs(math.sin(wave_phase))))
            color = tuple(min(255, c + alpha // 3) for c in self.colors['sound_wave'][:3])
            pygame.draw.circle(self.screen, color, screen_pos, int(wave_radius), 2)
        
        # Draw sound source
        pygame.draw.circle(self.screen, self.colors['sound_source'], screen_pos, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 8, 2)
    
    def _draw_text(self, text: str, pos: Tuple[int, int], font=None, color=None):
        """Draw text on screen."""
        if font is None:
            font = self.font
        if color is None:
            color = self.colors['text']
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)
    
    def _draw_info_panel(self, step: int, sound_energy: float, 
                        end_effector_pos: np.ndarray,
                        distance_to_source: float):
        """Draw information panel."""
        y_offset = 10
        x_offset = 10
        
        self._draw_text(f"Step: {step}", (x_offset, y_offset))
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
    
    def render_frame(self, render_data: Dict[str, Any], step: int = 0,
                    wave_phase: float = 0.0):
        """
        Render a single frame.
        
        Args:
            render_data: Data from env.render()
            step: Current step number
            wave_phase: Phase for wave animation
        """
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
        sound_source_positions = render_data['sound_source_positions']
        sound_energy = render_data.get('sound_energy', 0.0)
        
        # Calculate distance to nearest source for info panel
        if sound_source_positions:
            distances = [np.linalg.norm(end_effector_pos - sp) for sp in sound_source_positions]
            distance_to_source = min(distances)
        else:
            distance_to_source = float('inf')
        
        for source_pos in sound_source_positions:
            self._draw_sound_source(source_pos, strength=1.0, wave_phase=wave_phase)
        
        # Draw info panel
        self._draw_info_panel(step, sound_energy, end_effector_pos, distance_to_source)
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events. Returns True if should continue, False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True
    
    async def animate(self, env, agent, max_steps: int = 10000,
                     steps_per_frame: int = 1):
        """
        Animate the environment in real-time.
        
        Args:
            env: SoundFlowerEnvironment instance
            agent: Agent instance
            max_steps: Maximum number of steps
            steps_per_frame: Number of simulation steps per frame
        """
        self.running = True
        self.paused = False
        
        observation = env.reset()
        step = 0
        wave_phase = 0.0
        
        while self.running and step < max_steps:
            # Handle events
            if not self.handle_events():
                break
            
            # Update wave animation phase
            wave_phase += 0.1
            if wave_phase > 2 * math.pi:
                wave_phase -= 2 * math.pi
            
            # Step environment if not paused
            if not self.paused:
                for _ in range(steps_per_frame):
                    action = await agent.select_action(observation)
                    observation, reward, done, info = await env.step(action)
                    step += 1
                    
                    if done:
                        break
            
            # Render frame
            render_data = env.render()
            self.render_frame(render_data, step=step, wave_phase=wave_phase)
            
            # Control frame rate
            self.clock.tick(self.fps)
            
            # Small async delay to allow other tasks
            await asyncio.sleep(0.001)
        
        self.running = False
    
    def close(self):
        """Close the visualizer."""
        pygame.quit()

