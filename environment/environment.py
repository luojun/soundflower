"""Environment interface - represents the simulation state and logic."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .physics_engine import PhysicsEngine, PhysicsState


@dataclass
class Observation:
    """Observation from the environment."""
    arm_angles: np.ndarray  # Current joint angles
    arm_angular_velocities: np.ndarray  # Current joint angular velocities
    arm_angular_accelerations: np.ndarray  # Current joint angular accelerations
    last_torques: np.ndarray  # Last applied joint torques (efference copy)
    sound_intensity: float  # Current sound intensity at microphone (power per unit area)
    # Full-mode fields (None in Sensorimotor mode)
    end_effector_pos: Optional[np.ndarray] = None  # End effector position (x, y)
    sound_source_positions: Optional[List[np.ndarray]] = None  # Positions of sound sources
    microphone_orientation: Optional[float] = None  # Orientation angle of microphone (rad)
    link_lengths: Optional[np.ndarray] = None
    link_masses: Optional[np.ndarray] = None
    joint_frictions: Optional[np.ndarray] = None


@dataclass
class State:
    """Complete state of the environment."""
    physics_state: PhysicsState
    observation: Optional[Observation] = None
    sound_intensity: Optional[float] = None
    sound_energy: Optional[float] = None
    sound_energy_delta: Optional[float] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = None


class Environment:
    """
    Environment interface.

    Represents the simulation environment state and provides methods to:
    - Query current state
    - Apply actions
    - Compute observations and rewards
    - Reset to initial state
    """

    def __init__(self, config):
        """
        Initialize.

        Args:
            config: Configuration object
        """
        self.config = config
        self.physics_engine = PhysicsEngine(config)

        # Track previous sound energy for delta computation
        self.previous_sound_energy = 0.0
        self.previous_angular_velocities = np.zeros(config.num_links)
        self.last_action = np.zeros(config.num_links)

    def step(self):
        self.physics_engine.step()

    def get_state(self) -> State:
        """
        Get current environment state.

        Returns:
            Current environment state
        """
        physics_state = self.physics_engine.get_state()
        observation, sound_metrics = self._compute_observation(physics_state)

        # Compute reward
        reward = sound_metrics["sound_energy_delta"]

        # Normalize reward by maximum possible delta
        if self.config.reward_normalization_factor > 0:
            reward = reward / self.config.reward_normalization_factor

        return State(
            physics_state=physics_state,
            observation=observation,
            sound_intensity=sound_metrics["sound_intensity"],
            sound_energy=sound_metrics["sound_energy"],
            sound_energy_delta=sound_metrics["sound_energy_delta"],
            reward=reward,
            done=False,  # Infinite horizon
            info=None
        )

    def apply_action(self, action: np.ndarray):
        """
        Apply action to the environment.

        Args:
            action: Action to apply (torques for each joint)
        """
        # Clip action
        action = np.clip(action, -self.config.max_torque, self.config.max_torque)

        self.last_action = action.copy()
        # Apply to physics engine
        self.physics_engine.set_torques(action)


    def _compute_observation(self, physics_state: PhysicsState) -> Tuple[Observation, Dict[str, float]]:
        """Compute observation from physics state."""
        # Forward kinematics
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            physics_state.arm_state.angles
        )

        # Compute microphone orientation (direction of outmost link from base)
        # Orientation is the cumulative angle of all joints
        microphone_orientation = np.sum(physics_state.arm_state.angles)

        # Sound source positions (vectorized)
        if physics_state.sound_source_angles:
            angles_array = np.array(physics_state.sound_source_angles)
            sound_source_positions = np.column_stack([
                self.config.circle_radius * np.cos(angles_array),
                self.config.circle_radius * np.sin(angles_array)
            ])
        else:
            sound_source_positions = np.array([]).reshape(0, 2)

        # Compute sound intensity and energy from all sources (vectorized)
        sound_intensity = 0.0
        sound_energy = 0.0
        microphone_area = self.config.microphone_area

        if len(sound_source_positions) > 0:
            directions = sound_source_positions - end_effector_pos  # (n_sources, 2)
            distances = np.linalg.norm(directions, axis=1)  # (n_sources,)
            # Use config minimum distance (physical constraint should prevent violations, but
            # this ensures calculation safety even if constraint hasn't been applied yet)
            min_distance = self.config.min_distance_to_source
            distances = np.maximum(distances, min_distance)

            # Compute sound intensities for all sources (vectorized)
            # Inverse square law: intensity = P / (4πr²) where P is power
            source_intensities = (self.physics_engine.sound_propagation.attenuation_coeff *
                                 self.config.sound_source_strength / (4 * np.pi * distances ** 2))  # (n_sources,)
            source_intensities *= self.config.microphone_gain

            sound_intensity = np.sum(source_intensities)

            source_direction_angles = np.arctan2(directions[:, 1], directions[:, 0])  # (n_sources,)

            orientation_errors = source_direction_angles - microphone_orientation  # (n_sources,)
            # Normalize to [-pi, pi]
            orientation_errors = np.arctan2(np.sin(orientation_errors), np.cos(orientation_errors))

            # Orientation factors: cosine of angle difference, clipped to [0, 1]
            orientation_factors = np.maximum(0.0, np.cos(orientation_errors))  # (n_sources,)

            # Energy = Intensity × Area × Time × Orientation_factor
            source_energies = (source_intensities * microphone_area * self.config.dt *
                              orientation_factors)  # (n_sources,)
            sound_energy = np.sum(source_energies)

        sound_energy_delta = sound_energy - self.previous_sound_energy
        self.previous_sound_energy = sound_energy

        # Joint angular accelerations (finite difference)
        arm_angular_accelerations = (physics_state.arm_state.angular_velocities - self.previous_angular_velocities) / self.config.dt
        self.previous_angular_velocities = physics_state.arm_state.angular_velocities.copy()

        # Convert sound_source_positions back to list format for Observation dataclass
        sound_source_positions_list = [pos for pos in sound_source_positions] if len(sound_source_positions) > 0 else []

        observation_mode = getattr(self.config, "observation_mode", "full").lower()
        if observation_mode not in {"sensorimotor", "full"}:
            raise ValueError(f"Unknown observation_mode: {self.config.observation_mode}")

        if observation_mode == "full":
            observation = Observation(
                arm_angles=physics_state.arm_state.angles.copy(),
                arm_angular_velocities=physics_state.arm_state.angular_velocities.copy(),
                arm_angular_accelerations=arm_angular_accelerations.copy(),
                last_torques=self.last_action.copy(),
                sound_intensity=sound_intensity,
                end_effector_pos=end_effector_pos.copy(),
                sound_source_positions=sound_source_positions_list,
                microphone_orientation=microphone_orientation,
                link_lengths=np.array(self.config.link_lengths),
                link_masses=np.array(self.config.link_masses),
                joint_frictions=np.array(self.config.joint_frictions)
            )
        else:
            observation = Observation(
                arm_angles=physics_state.arm_state.angles.copy(),
                arm_angular_velocities=physics_state.arm_state.angular_velocities.copy(),
                arm_angular_accelerations=arm_angular_accelerations.copy(),
                last_torques=self.last_action.copy(),
                sound_intensity=sound_intensity
            )

        sound_metrics = {
            "sound_intensity": sound_intensity,
            "sound_energy": sound_energy,
            "sound_energy_delta": sound_energy_delta
        }

        return observation, sound_metrics

    def get_render_data(self) -> Dict[str, Any]:
        """
        Get data for rendering.

        Returns:
            Dictionary with render data
        """
        physics_state = self.physics_engine.get_state()
        joint_positions, end_effector_pos = self.physics_engine.arm_physics.forward_kinematics(
            physics_state.arm_state.angles
        )

        if physics_state.sound_source_angles:
            angles_array = np.array(physics_state.sound_source_angles)
            sound_source_positions = np.column_stack([
                self.config.circle_radius * np.cos(angles_array),
                self.config.circle_radius * np.sin(angles_array)
            ])
        else:
            sound_source_positions = np.array([]).reshape(0, 2)

        sound_intensity = 0.0
        if len(sound_source_positions) > 0:
            directions = sound_source_positions - end_effector_pos  # (n_sources, 2)
            distances = np.linalg.norm(directions, axis=1)  # (n_sources,)
            # Use config minimum distance (physical constraint should prevent violations, but
            # this ensures calculation safety even if constraint hasn't been applied yet)
            min_distance = self.config.min_distance_to_source
            distances = np.maximum(distances, min_distance)

            # Inverse square law: intensity = P / (4πr²) where P is power
            source_intensities = (self.physics_engine.sound_propagation.attenuation_coeff *
                                 self.config.sound_source_strength / (4 * np.pi * distances ** 2))  # (n_sources,)
            source_intensities *= self.config.microphone_gain
            sound_intensity = np.sum(source_intensities)

        # Convert to list for render data
        sound_source_positions_list = [pos for pos in sound_source_positions] if len(sound_source_positions) > 0 else []

        return {
            'joint_positions': joint_positions,
            'end_effector_pos': end_effector_pos,
            'sound_source_positions': sound_source_positions_list,
            'circle_radius': self.config.circle_radius,
            'arm_state': physics_state.arm_state,
            'sound_intensity': sound_intensity,
            'sound_energy': sound_intensity * self.config.microphone_area * self.config.dt  # Area × dt for rendering
        }

