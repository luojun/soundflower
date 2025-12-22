"""Visualizer for Sound Flower environment."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Any
import asyncio


class SoundFlowerVisualizer:
    """Visualizer for the Sound Flower environment."""
    
    def __init__(self, circle_radius: float, link_lengths: List[float], 
                 figsize: tuple = (10, 10), show_sound_field: bool = False):
        """
        Initialize visualizer.
        
        Args:
            circle_radius: Radius of the environment circle
            link_lengths: Lengths of arm links
            figsize: Figure size (width, height)
            show_sound_field: Whether to show sound energy field visualization
        """
        self.circle_radius = circle_radius
        self.link_lengths = link_lengths
        self.figsize = figsize
        self.show_sound_field = show_sound_field
        
        # Calculate plot limits (with some padding)
        max_reach = sum(link_lengths)
        plot_limit = max(circle_radius, max_reach) * 1.2
        
        self.xlim = (-plot_limit, plot_limit)
        self.ylim = (-plot_limit, plot_limit)
        
        # Initialize figure and axes
        self.fig = None
        self.ax = None
        self.animation = None
        
    def _setup_plot(self):
        """Set up the matplotlib figure and axes."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.ax.set_aspect('equal')
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title('Sound Flower - Robotic Arm Sound Source Tracking')
            
            # Draw circle boundary
            circle = patches.Circle((0, 0), self.circle_radius, 
                                  fill=False, edgecolor='gray', 
                                  linestyle='--', linewidth=2, label='Boundary')
            self.ax.add_patch(circle)
            
            # Draw center point (mount point)
            self.ax.plot(0, 0, 'ko', markersize=10, label='Mount Point')
    
    def plot_state(self, render_data: Dict[str, Any], 
                   observation: Optional[Any] = None,
                   save_path: Optional[str] = None,
                   show: bool = True):
        """
        Plot a single state snapshot.
        
        Args:
            render_data: Data from env.render()
            observation: Optional observation object for additional info
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        self._setup_plot()
        self.ax.clear()
        
        # Re-setup axes
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        # Draw circle boundary
        circle = patches.Circle((0, 0), self.circle_radius, 
                              fill=False, edgecolor='gray', 
                              linestyle='--', linewidth=2)
        self.ax.add_patch(circle)
        
        # Draw center point
        self.ax.plot(0, 0, 'ko', markersize=10, zorder=10)
        
        # Draw arm
        joint_positions = render_data['joint_positions']
        end_effector_pos = render_data['end_effector_pos']
        
        # Draw links
        for i in range(len(joint_positions) - 1):
            x_coords = [joint_positions[i][0], joint_positions[i+1][0]]
            y_coords = [joint_positions[i][1], joint_positions[i+1][1]]
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Arm' if i == 0 else '')
        
        # Draw joints
        for i, joint_pos in enumerate(joint_positions):
            if i == 0:
                # Base joint (mount point)
                self.ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=12, zorder=10)
            else:
                self.ax.plot(joint_pos[0], joint_pos[1], 'bo', markersize=8, zorder=9)
        
        # Draw end effector with microphone
        self.ax.plot(end_effector_pos[0], end_effector_pos[1], 'go', 
                    markersize=12, label='Microphone', zorder=11)
        
        # Draw sound sources
        sound_source_positions = render_data['sound_source_positions']
        for i, source_pos in enumerate(sound_source_positions):
            # Draw sound source as a red circle with sound waves
            circle_source = patches.Circle((source_pos[0], source_pos[1]), 0.1,
                                          color='red', alpha=0.8, zorder=8)
            self.ax.add_patch(circle_source)
            
            # Draw sound waves (concentric circles)
            for radius in [0.2, 0.3, 0.4]:
                wave = patches.Circle((source_pos[0], source_pos[1]), radius,
                                    fill=False, edgecolor='red', 
                                    linestyle=':', linewidth=1, alpha=0.3)
                self.ax.add_patch(wave)
        
        # Add sound energy text if available
        if 'sound_energy' in render_data:
            sound_energy = render_data['sound_energy']
            self.ax.text(0.02, 0.98, f'Sound Energy: {sound_energy:.4f}',
                        transform=self.ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)
    
    def create_animation(self, render_data_sequence: List[Dict[str, Any]],
                        interval: int = 50, save_path: Optional[str] = None):
        """
        Create an animation from a sequence of render data.
        
        Args:
            render_data_sequence: List of render data dictionaries
            interval: Animation interval in milliseconds
            save_path: Optional path to save the animation
            
        Returns:
            animation: Matplotlib animation object
        """
        self._setup_plot()
        
        def init():
            """Initialize animation."""
            # Draw circle boundary
            circle = patches.Circle((0, 0), self.circle_radius, 
                                  fill=False, edgecolor='gray', 
                                  linestyle='--', linewidth=2)
            self.ax.add_patch(circle)
            
            # Draw center point
            self.ax.plot(0, 0, 'ko', markersize=10, zorder=10)
            
            return []
        
        def animate(frame):
            """Animate frame."""
            # Clear axes (except static elements)
            self.ax.clear()
            
            # Re-setup axes
            self.ax.set_aspect('equal')
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title(f'Sound Flower - Step {frame}')
            
            # Draw circle boundary
            circle = patches.Circle((0, 0), self.circle_radius, 
                                  fill=False, edgecolor='gray', 
                                  linestyle='--', linewidth=2)
            self.ax.add_patch(circle)
            
            # Draw center point
            self.ax.plot(0, 0, 'ko', markersize=10, zorder=10)
            
            # Get current frame data
            render_data = render_data_sequence[frame]
            
            # Draw arm
            joint_positions = render_data['joint_positions']
            end_effector_pos = render_data['end_effector_pos']
            
            # Draw links
            for i in range(len(joint_positions) - 1):
                x_coords = [joint_positions[i][0], joint_positions[i+1][0]]
                y_coords = [joint_positions[i][1], joint_positions[i+1][1]]
                self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Arm' if i == 0 else '')
            
            # Draw joints
            for i, joint_pos in enumerate(joint_positions):
                if i == 0:
                    # Base joint already drawn
                    pass
                else:
                    self.ax.plot(joint_pos[0], joint_pos[1], 'bo', markersize=8, zorder=9)
            
            # Draw end effector
            self.ax.plot(end_effector_pos[0], end_effector_pos[1], 
                        'go', markersize=12, label='Microphone', zorder=11)
            
            # Draw sound sources
            sound_source_positions = render_data['sound_source_positions']
            for source_pos in sound_source_positions:
                circle = patches.Circle((source_pos[0], source_pos[1]), 0.1,
                                      color='red', alpha=0.8, zorder=8)
                self.ax.add_patch(circle)
                
                # Draw sound waves
                for radius in [0.2, 0.3, 0.4]:
                    wave = patches.Circle((source_pos[0], source_pos[1]), radius,
                                        fill=False, edgecolor='red', 
                                        linestyle=':', linewidth=1, alpha=0.3)
                    self.ax.add_patch(wave)
            
            # Add sound energy text
            if 'sound_energy' in render_data:
                sound_energy = render_data['sound_energy']
                self.ax.text(0.02, 0.98, f'Step: {frame}\nSound Energy: {sound_energy:.4f}',
                           transform=self.ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add legend
            self.ax.legend(loc='upper right')
            
            return []
        
        self.animation = FuncAnimation(self.fig, animate, init_func=init,
                                     frames=len(render_data_sequence),
                                     interval=interval, blit=False, repeat=True)
        
        if save_path:
            self.animation.save(save_path, writer='pillow', fps=20)
        
        return self.animation
    
    async def animate_async(self, env, agent, max_steps: int = 1000,
                           update_interval: float = 0.05):
        """
        Animate environment in real-time using async updates.
        
        Args:
            env: SoundFlowerEnvironment instance
            agent: Agent instance
            max_steps: Maximum number of steps
            update_interval: Time between updates in seconds
        """
        self._setup_plot()
        
        observation = env.reset()
        
        plt.ion()  # Turn on interactive mode
        
        try:
            for step in range(max_steps):
                # Clear axes (will redraw everything)
                self.ax.clear()
                
                # Re-setup axes
                self.ax.set_aspect('equal')
                self.ax.set_xlim(self.xlim)
                self.ax.set_ylim(self.ylim)
                self.ax.grid(True, alpha=0.3)
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                self.ax.set_title(f'Sound Flower - Step {step}')
                
                # Draw circle boundary
                circle = patches.Circle((0, 0), self.circle_radius, 
                                      fill=False, edgecolor='gray', 
                                      linestyle='--', linewidth=2)
                self.ax.add_patch(circle)
                
                # Draw center point
                self.ax.plot(0, 0, 'ko', markersize=10, zorder=10)
                
                # Get render data
                render_data = env.render()
                
                # Draw arm
                joint_positions = render_data['joint_positions']
                end_effector_pos = render_data['end_effector_pos']
                
                # Draw links
                for i in range(len(joint_positions) - 1):
                    x_coords = [joint_positions[i][0], joint_positions[i+1][0]]
                    y_coords = [joint_positions[i][1], joint_positions[i+1][1]]
                    self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Arm' if i == 0 else '')
                
                # Draw joints
                for i, joint_pos in enumerate(joint_positions):
                    if i > 0:
                        self.ax.plot(joint_pos[0], joint_pos[1], 'bo', markersize=8, zorder=9)
                
                # Draw end effector
                self.ax.plot(end_effector_pos[0], end_effector_pos[1], 
                            'go', markersize=12, label='Microphone', zorder=11)
                
                # Draw sound sources
                sound_source_positions = render_data['sound_source_positions']
                for source_pos in sound_source_positions:
                    circle = patches.Circle((source_pos[0], source_pos[1]), 0.1,
                                          color='red', alpha=0.8, zorder=8)
                    self.ax.add_patch(circle)
                    
                    # Draw sound waves
                    for radius in [0.2, 0.3, 0.4]:
                        wave = patches.Circle((source_pos[0], source_pos[1]), radius,
                                            fill=False, edgecolor='red', 
                                            linestyle=':', linewidth=1, alpha=0.3)
                        self.ax.add_patch(wave)
                
                # Add info text
                sound_energy = render_data.get('sound_energy', 0.0)
                self.ax.text(0.02, 0.98, 
                           f'Step: {step}\nSound Energy: {sound_energy:.4f}\n'
                           f'End Effector: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f})',
                           transform=self.ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Add legend
                self.ax.legend(loc='upper right')
                
                # Update display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                # Select action and step environment
                action = await agent.select_action(observation)
                observation, reward, done, info = await env.step(action)
                
                # Wait before next update
                await asyncio.sleep(update_interval)
                
                if done:
                    break
        
        finally:
            plt.ioff()  # Turn off interactive mode
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

