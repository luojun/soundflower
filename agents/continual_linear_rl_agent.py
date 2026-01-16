"""Continual linear RL agent with TD(lambda) and adaptive step sizes."""

from __future__ import annotations

import numpy as np
from environment import Observation
from .linear_reactive_agent import LinearReactiveAgent


class ContinualLinearRLAgent(LinearReactiveAgent):
    """
    Linear continual-RL baseline built on the reactive core.

    Minimal, principled baseline:
    - Keeps the same linear memory dynamics z_{t+1} = A z_t + B x_t (fixed A).
    - Learns a linear policy W online with TD(lambda).
    - Uses per-feature adaptive step sizes (IDBD-style) and gentle weight decay
      to handle non-stationarity without adding extra control logic.
    """

    def __init__(self,
                 memory_size: int | None = None,
                 decay: float = 0.9,
                 coupling: float = 0.05,
                 input_scale: float = 0.1,
                 output_scale: float = 0.1,
                 seed: int | None = 0,
                 use_intensity_delta: bool = True,
                 gamma: float = 0.98,
                 trace_decay: float = 0.9,
                 base_step_size: float = 0.05,
                 meta_step_size: float = 0.005,
                 forget_rate: float = 0.001,
                 min_step_size: float = 1e-6,
                 max_step_size: float = 1.0,
                 delta_rms_decay: float = 0.99,
                 delta_clip: float = 5.0,
                 max_weight_norm: float = 100.0,
                 max_value_norm: float = 100.0,
                 max_trace_norm: float = 100.0):
        super().__init__(
            memory_size=memory_size,
            decay=decay,
            coupling=coupling,
            input_scale=input_scale,
            output_scale=output_scale,
            seed=seed,
            use_intensity_delta=use_intensity_delta,
            low_reward_decay=1.0,
            reward_threshold=-np.inf,
        )
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.base_step_size = base_step_size
        self.meta_step_size = meta_step_size
        self.forget_rate = forget_rate
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.delta_rms_decay = delta_rms_decay
        self.delta_clip = delta_clip
        self.max_weight_norm = max_weight_norm
        self.max_value_norm = max_value_norm
        self.max_trace_norm = max_trace_norm

        self._pending_reward = None
        self._prev_z = None
        self._trace = None
        self._value_weights = None
        self._value_beta = None
        self._value_h = None
        self._policy_beta = None
        self._policy_h = None
        self._delta_rms = 0.0

    def _initialize(self, observation: Observation) -> None:
        super()._initialize(observation)
        z_dim = self._z.shape[0]
        base = max(self.base_step_size, self.min_step_size)
        beta_init = np.log(base)

        self._trace = np.zeros(z_dim)
        self._value_weights = np.zeros(z_dim)
        self._value_beta = np.full(z_dim, beta_init)
        self._value_h = np.zeros(z_dim)
        self._policy_beta = np.full_like(self._W, beta_init)
        self._policy_h = np.zeros_like(self._W)
        self._prev_z = None
        self._pending_reward = None
        self._delta_rms = 0.0

    def _advance_state(self, observation: Observation) -> np.ndarray:
        x = self._build_features(observation)
        self._z = self._A @ self._z + self._B @ x
        return self._z

    def observe(self, reward: float, observation: Observation) -> None:
        self._pending_reward = reward

    def _update_idbd_vector(self, delta: float, z_prev: np.ndarray, trace: np.ndarray) -> None:
        beta_min = np.log(self.min_step_size)
        beta_max = np.log(self.max_step_size)
        self._value_beta = np.clip(self._value_beta, beta_min, beta_max)
        self._value_beta += self.meta_step_size * delta * trace * self._value_h
        self._value_beta = np.clip(self._value_beta, beta_min, beta_max)
        alpha = np.exp(self._value_beta)
        self._value_h = self._value_h * (1.0 - alpha * trace * z_prev) + alpha * delta * trace
        self._value_weights += alpha * delta * trace

    def _update_idbd_matrix(self, delta: float, z_prev: np.ndarray, trace: np.ndarray) -> None:
        trace_row = trace[None, :]
        z_row = z_prev[None, :]
        beta_min = np.log(self.min_step_size)
        beta_max = np.log(self.max_step_size)
        self._policy_beta = np.clip(self._policy_beta, beta_min, beta_max)
        self._policy_beta += self.meta_step_size * delta * self._policy_h * trace_row
        self._policy_beta = np.clip(self._policy_beta, beta_min, beta_max)
        alpha = np.exp(self._policy_beta)
        self._policy_h = self._policy_h * (1.0 - alpha * trace_row * z_row) + alpha * delta * trace_row
        self._W += alpha * delta * trace_row

    def _apply_td_update(self, reward: float, z_prev: np.ndarray, z_current: np.ndarray) -> None:
        self._trace = self.gamma * self.trace_decay * self._trace + z_prev
        if self.max_trace_norm > 0.0:
            trace_norm = float(np.linalg.norm(self._trace))
            if trace_norm > self.max_trace_norm:
                self._trace *= (self.max_trace_norm / (trace_norm + 1e-8))
        value_prev = float(self._value_weights @ z_prev)
        value_curr = float(self._value_weights @ z_current)
        delta_raw = reward + self.gamma * value_curr - value_prev
        self._delta_rms = (
            self.delta_rms_decay * self._delta_rms +
            (1.0 - self.delta_rms_decay) * (delta_raw ** 2)
        )
        delta_scale = np.sqrt(self._delta_rms) + 1e-8
        delta = float(np.clip(delta_raw, -self.delta_clip * delta_scale, self.delta_clip * delta_scale))
        if not np.isfinite(delta):
            raise RuntimeError(
                "Non-finite TD error "
                f"(reward={reward}, value_prev={value_prev}, value_curr={value_curr}, delta_raw={delta_raw})"
            )

        if self.forget_rate > 0.0:
            decay = 1.0 - self.forget_rate
            self._value_weights *= decay
            self._W *= decay

        trace_norm_sq = float(np.dot(self._trace, self._trace))
        if trace_norm_sq > 0.0:
            stability_cap = 1.0 / (trace_norm_sq + 1e-8)
            self._policy_beta = np.minimum(self._policy_beta, np.log(stability_cap))
            self._value_beta = np.minimum(self._value_beta, np.log(stability_cap))

        self._update_idbd_vector(delta, z_prev, self._trace)
        self._update_idbd_matrix(delta, z_prev, self._trace)
        if self.max_weight_norm > 0.0:
            weight_norm = float(np.linalg.norm(self._W))
            if weight_norm > self.max_weight_norm:
                self._W *= (self.max_weight_norm / (weight_norm + 1e-8))
        if self.max_value_norm > 0.0:
            value_norm = float(np.linalg.norm(self._value_weights))
            if value_norm > self.max_value_norm:
                self._value_weights *= (self.max_value_norm / (value_norm + 1e-8))
        if not np.all(np.isfinite(self._W)):
            raise RuntimeError(f"Non-finite policy weights after update: {self._W}")

    def select_action(self, observation: Observation) -> np.ndarray:
        if not self._initialized:
            self._initialize(observation)

        z_current = self._advance_state(observation)
        if not np.all(np.isfinite(z_current)):
            raise RuntimeError(f"Non-finite memory state z: {z_current}")
        if self._pending_reward is not None and self._prev_z is not None:
            self._apply_td_update(self._pending_reward, self._prev_z, z_current)
        self._prev_z = z_current.copy()
        self._pending_reward = None
        torques = self._W @ z_current
        if not np.all(np.isfinite(torques)):
            raise RuntimeError(f"Non-finite torques produced: {torques}")
        return torques
