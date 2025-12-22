"""Simple heuristic agent for sound source tracking."""

import numpy as np
from typing import Optional
import asyncio

from soundflower.environment import Observation, SoundFlowerEnvironment


class HeuristicAgent:
    """Simple heuristic agent that points the arm toward the sound source."""
    
    def __init__(self, environment: SoundFlowerEnvironment, kp: float = 5.0, kd: float = 0.5):
        """
        Initialize heuristic agent.
        
        Args:
            environment: The environment to interact with
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
        """
        self.env = environment
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        
        # Target angle for the end effector (to point at sound source)
        self.target_angle = 0.0
    
    def _compute_target_angle(self, observation: Observation) -> float:
        """
        Compute target angle for end effector to point at nearest sound source.
        
        Args:
            observation: Current observation
            
        Returns:
            target_angle: Target angle in radians
        """
        if not observation.sound_source_positions:
            return 0.0
        
        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        nearest_source = observation.sound_source_positions[nearest_idx]
        
        # Compute angle from center to sound source
        target_angle = np.arctan2(nearest_source[1], nearest_source[0])
        
        return target_angle
    
    def _compute_desired_joint_angles(self, target_angle: float, 
                                     current_angles: np.ndarray) -> np.ndarray:
        """
        Compute desired joint angles using inverse kinematics (simplified).
        
        For a multi-link arm, we use a simple strategy:
        - First joint points toward target
        - Subsequent joints try to extend the arm
        
        Args:
            target_angle: Target angle for end effector
            current_angles: Current joint angles
            
        Returns:
            desired_angles: Desired joint angles
        """
        num_links = len(current_angles)
        desired_angles = np.zeros(num_links)
        
        if num_links == 1:
            # Single link: just point at target
            desired_angles[0] = target_angle
        elif num_links == 2:
            # Two links: first joint points roughly toward target
            # Second joint extends the arm
            desired_angles[0] = target_angle * 0.7  # First joint does most of the pointing
            desired_angles[1] = target_angle * 0.3  # Second joint fine-tunes
        else:
            # Three links: distribute angle across joints
            desired_angles[0] = target_angle * 0.5
            desired_angles[1] = target_angle * 0.3
            desired_angles[2] = target_angle * 0.2
        
        return desired_angles
    
    async def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action using heuristic control.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Torques to apply at each joint
        """
        # Compute target angle
        target_angle = self._compute_target_angle(observation)
        self.target_angle = target_angle
        
        # Compute desired joint angles
        desired_angles = self._compute_desired_joint_angles(
            target_angle, observation.arm_angles
        )
        
        # PD controller: compute torques
        angle_errors = desired_angles - observation.arm_angles
        
        # Proportional term
        proportional_torques = self.kp * angle_errors
        
        # Derivative term (damping)
        derivative_torques = -self.kd * observation.arm_angular_velocities
        
        # Total torques
        torques = proportional_torques + derivative_torques
        
        # Clamp to valid range
        max_torque = self.env.config.max_torque
        torques = np.clip(torques, -max_torque, max_torque)
        
        return torques
    
    async def run_episode(self, max_steps: int = 1000, render: bool = False) -> dict:
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
            action = await self.select_action(observation)
            
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

