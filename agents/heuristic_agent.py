"""Simple heuristic agent for sound source tracking."""

import numpy as np
from typing import Optional
import asyncio

from environment import Observation


class HeuristicAgent:
    """Simple heuristic agent that points the arm toward the sound source."""
    
    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0):
        """
        Initialize heuristic agent.
        
        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping). 
        """
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.max_torque = max_torque
        
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
        torques = np.clip(torques, -self.max_torque, self.max_torque)
        
        return torques

