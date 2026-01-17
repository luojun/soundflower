"""Linear reactive agent with leaky memory for sensorimotor control."""

from __future__ import annotations

import numpy as np
from environment import Observation
from .base_agent import BaseAgent


class LinearReactiveAgent(BaseAgent):
    """
    Minimal linear reactive controller with a small leaky memory state.

    Rationale:
    - Sensorimotor-only inputs (no global geometry).
    - Learning-friendly linear maps.
    - No explicit sinusoids; short-term memory can yield oscillatory coordination
      via non-diagonal dynamics.
    - If A = λI, the memory reduces to an eligibility-trace-like decay.
    """

    def __init__(self,
                 memory_size: int | None = None,
                 decay: float = 0.9,
                 coupling: float = 0.05,
                 input_scale: float = 0.1,
                 output_scale: float = 0.1,
                 seed: int | None = 0,
                 use_intensity_delta: bool = True,
                 low_reward_decay: float = 0.8,
                 reward_threshold: float = 0.0):
        super().__init__()
        self.memory_size = memory_size
        self.decay = decay
        self.coupling = coupling
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.use_intensity_delta = use_intensity_delta
        self.low_reward_decay = low_reward_decay
        self.reward_threshold = reward_threshold

        self._rng = np.random.default_rng(seed)
        self._initialized = False
        self._num_joints = 0
        self._x_dim = 0
        self._z = None
        self._A = None
        self._B = None
        self._W = None
        self._prev_intensity = None

    def _initialize(self, observation: Observation) -> None:
        self._num_joints = len(observation.arm_angles)
        x_dim = 4 * self._num_joints + 1  # angles, velocities, accelerations, last_torques, intensity
        if self.use_intensity_delta:
            x_dim += 1
        self._x_dim = x_dim

        z_dim = self.memory_size if self.memory_size is not None else max(2, 2 * self._num_joints)

        # Leaky memory dynamics; add small rotational coupling for coordination.
        A = self.decay * np.eye(z_dim)
        if self.coupling != 0.0 and z_dim >= 2:
            for i in range(0, z_dim - 1, 2):
                A[i, i + 1] -= self.coupling
                A[i + 1, i] += self.coupling

        self._A = A
        self._B = self._rng.normal(scale=self.input_scale, size=(z_dim, x_dim))
        self._W = self._rng.normal(scale=self.output_scale, size=(self._num_joints, z_dim))
        self._z = np.zeros(z_dim)
        self._prev_intensity = observation.sound_intensity
        self._initialized = True

    def _build_features(self, observation: Observation) -> np.ndarray:
        intensity = observation.sound_intensity
        intensity_delta = 0.0
        if self.use_intensity_delta:
            if self._prev_intensity is not None:
                intensity_delta = intensity - self._prev_intensity
            self._prev_intensity = intensity

        x_parts = [
            observation.arm_angles,
            observation.arm_angular_velocities,
            observation.arm_angular_accelerations,
            observation.last_torques,
            np.array([intensity]),
        ]
        if self.use_intensity_delta:
            x_parts.append(np.array([intensity_delta]))
        return np.concatenate(x_parts)

    def _update_state(self, observation: Observation) -> None:
        intensity_delta = 0.0
        if self.use_intensity_delta and self._prev_intensity is not None:
            intensity_delta = observation.sound_intensity - self._prev_intensity

        if intensity_delta <= self.reward_threshold:
            self._z *= self.low_reward_decay

        x = self._build_features(observation)
        self._z = self._A @ self._z + self._B @ x

    def select_action(self, observation: Observation) -> np.ndarray:
        if not self._initialized:
            self._initialize(observation)
        self._update_state(observation)
        torques = self._W @ self._z
        return torques
