"""Experiment class for running Sound Flower episodes and experiments."""

import asyncio
import numpy as np
from typing import Dict, Optional, Any, List
from soundflower.environment import SoundFlowerEnvironment, Observation
from soundflower.config import SoundFlowerConfig


def create_default_config(sound_source_angular_velocity: float = 0.2) -> SoundFlowerConfig:
    """
    Create default configuration for experiments.
    
    Args:
        sound_source_angular_velocity: Angular velocity of sound source (rad/s)
        
    Returns:
        Default SoundFlowerConfig
    """
    return SoundFlowerConfig(
        num_links=2,
        link_lengths=[0.6, 0.4],
        link_masses=[1.0, 0.8],
        joint_frictions=[0.1, 0.15],
        circle_radius=1.0,
        num_microphones=1,
        microphone_gain=1.0,
        num_sound_sources=1,
        sound_source_strength=2.0,
        sound_attenuation_coeff=1.0,
        dt=0.01,
        max_torque=5.0,
        sound_source_angular_velocity=sound_source_angular_velocity,
        sound_source_initial_angle=np.pi / 4  # Start at 45 degrees
    )


class Experiment:
    """Experiment runner for Sound Flower environment."""
    
    def __init__(self, env: SoundFlowerEnvironment, agent: Any, config: Optional[SoundFlowerConfig] = None):
        """
        Initialize experiment.
        
        Args:
            env: Sound Flower environment
            agent: Agent with async select_action method
            config: Optional configuration (stored for reference)
        """
        self.env = env
        self.agent = agent
        self.config = config or env.config
    
    async def run_episode(self, max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            max_steps: Maximum number of steps
            render: Whether to collect render data
            
        Returns:
            episode_stats: Statistics about the episode
        """
        observation = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        render_data = []
        
        for step in range(max_steps):
            # Select action
            action = await self.agent.select_action(observation)
            
            # Step environment
            observation, reward, done, info = await self.env.step(action)
            
            total_reward += reward
            steps += 1
            
            if render:
                render_data.append(self.env.render())
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_sound_energy': observation.sound_energy,
            'final_distance_to_source': info.get('end_effector_distance_to_source', float('inf')),
            'render_data': render_data if render else None
        }
    
    async def run_episode_with_logging(self, max_steps: int = 1000, 
                                      log_interval: int = 20) -> Dict[str, Any]:
        """
        Run an episode with periodic logging.
        
        Args:
            max_steps: Maximum number of steps
            log_interval: Steps between log messages
            
        Returns:
            episode_stats: Statistics about the episode
        """
        observation = self.env.reset()
        
        print(f"\nInitial state:")
        print(f"  Arm angles: {observation.arm_angles}")
        print(f"  End effector position: {observation.end_effector_pos}")
        if observation.sound_source_positions:
            print(f"  Sound source position: {observation.sound_source_positions[0]}")
        print(f"  Initial sound energy: {observation.sound_energy:.4f}")
        
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            action = await self.agent.select_action(observation)
            observation, reward, done, info = await self.env.step(action)
            total_reward += reward
            steps += 1
            
            if step % log_interval == 0:
                print(f"\nStep {step}:")
                print(f"  Arm angles: {observation.arm_angles}")
                print(f"  End effector: ({observation.end_effector_pos[0]:.3f}, {observation.end_effector_pos[1]:.3f})")
                if observation.sound_source_positions:
                    print(f"  Sound source: ({observation.sound_source_positions[0][0]:.3f}, {observation.sound_source_positions[0][1]:.3f})")
                print(f"  Sound energy: {observation.sound_energy:.4f}")
                print(f"  Distance to source: {info.get('end_effector_distance_to_source', float('inf')):.4f}")
                print(f"  Cumulative reward: {total_reward:.4f}")
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_sound_energy': observation.sound_energy,
            'final_distance_to_source': info.get('end_effector_distance_to_source', float('inf')),
            'final_observation': observation
        }
    
    async def collect_render_data(self, max_steps: int = 100) -> List[Dict[str, Any]]:
        """
        Collect render data for visualization.
        
        Args:
            max_steps: Maximum number of steps
            
        Returns:
            render_data_sequence: List of render data dictionaries
        """
        observation = self.env.reset()
        render_data_sequence = []
        
        for step in range(max_steps):
            render_data = self.env.render()
            render_data_sequence.append(render_data)
            
            action = await self.agent.select_action(observation)
            observation, reward, done, info = await self.env.step(action)
            
            if done:
                break
        
        return render_data_sequence
    
    def print_config(self, config: Optional[SoundFlowerConfig] = None):
        """
        Print configuration information.
        
        Args:
            config: Optional config to print. If None, uses self.config.
        """
        cfg = config or self.config
        print("\nConfiguration:")
        print(f"  Number of links: {cfg.num_links}")
        print(f"  Link lengths: {cfg.link_lengths}")
        print(f"  Circle radius: {cfg.circle_radius}")
        print(f"  Total arm length: {sum(cfg.link_lengths):.2f}")
        print(f"  Sound source angular velocity: {cfg.sound_source_angular_velocity:.2f} rad/s")
    
    def print_episode_stats(self, stats: Dict[str, Any]):
        """Print episode statistics."""
        print("\nEpisode Statistics:")
        print(f"  Total steps: {stats['steps']}")
        print(f"  Total reward: {stats['total_reward']:.4f}")
        print(f"  Final sound energy: {stats['final_sound_energy']:.4f}")
        print(f"  Final distance to source: {stats['final_distance_to_source']:.4f}")

