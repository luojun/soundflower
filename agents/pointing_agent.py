"""Agent that only orients toward sound source without minimizing distance."""

import numpy as np
from environment import Observation
from .base_agent import BaseAgent


class PointingAgent(BaseAgent):
    """
    Agent that only orients the microphone toward the sound source.

    This agent maintains the current distance while pointing the microphone
    in the direction of the sound source for optimal sound reception.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.5, max_torque: float = 10.0,
                 link_lengths: np.ndarray = None):
        """
        Initialize pointing agent.

        Args:
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller
            max_torque: Maximum torque that can be applied (for clamping)
            link_lengths: Array of link lengths for IK computation
        """
        super().__init__(kp=kp, kd=kd, max_torque=max_torque, link_lengths=link_lengths)
        self.target_angle = 0.0

    def select_action(self, observation: Observation) -> np.ndarray:
        """
        Select action to point microphone toward sound source.

        Optimizes orientation toward sound source while ignoring distance to target.
        Distance may change naturally but is not subject to IK optimization pressure.

        Args:
            observation: Current observation

        Returns:
            action: Torques to apply at each joint
        """
        if not observation.sound_source_positions:
            return np.zeros(len(observation.arm_angles))

        # Find nearest sound source
        end_effector_pos = observation.end_effector_pos
        distances = [np.linalg.norm(end_effector_pos - sp) for sp in observation.sound_source_positions]
        nearest_idx = np.argmin(distances)
        source_pos = observation.sound_source_positions[nearest_idx]

        # Maintain current distance from base, optimize orientation toward source
        # Project source direction onto circle centered at base with current radius
        current_distance = np.linalg.norm(end_effector_pos)
        if current_distance > 1e-6:
            # Direction from base to source
            direction_to_source = source_pos / np.linalg.norm(source_pos)
            # Target position: maintain distance, point toward source
            target_pos = direction_to_source * current_distance

            # Solve IK to reach target position (maintains distance, optimizes orientation)
            desired_angles = self._solve_inverse_kinematics(
                target_pos, observation.arm_angles, self.link_lengths
            )
        else:
            # Fallback: if at base, use angle-based approach
            target_angle = self._compute_target_angle_for_pointing(observation)
            desired_angles = self._compute_desired_joint_angles(
                target_angle, observation.arm_angles
            )

        # Compute PD torques
        torques = self._compute_pd_torques(
            desired_angles,
            observation.arm_angles,
            observation.arm_angular_velocities
        )

        return torques

