"""Vanilla continual deep RL agent: small MLP, Adam, one-step TD/actor-critic."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from environment import Observation
from .base_agent import BaseAgent


def _build_features_from_obs(
    observation: Observation,
    use_intensity_delta: bool,
    prev_intensity: float | None,
) -> tuple[np.ndarray, float | None]:
    """Build feature vector matching LinearReactiveAgent (angles, velocities, accelerations, last_torques, intensity, optional intensity_delta)."""
    intensity = observation.sound_intensity
    intensity_delta = 0.0
    new_prev = intensity
    if use_intensity_delta and prev_intensity is not None:
        intensity_delta = intensity - prev_intensity
    parts = [
        observation.arm_angles,
        observation.arm_angular_velocities,
        observation.arm_angular_accelerations,
        observation.last_torques,
        np.array([intensity], dtype=np.float64),
    ]
    if use_intensity_delta:
        parts.append(np.array([intensity_delta], dtype=np.float64))
    return np.concatenate(parts), new_prev


class _Actor(nn.Module):
    """Small MLP with Gaussian policy: μ and log_std per joint."""

    def __init__(self, input_dim: int, num_joints: int, hidden: int = 32, max_torque: float = 2.0):
        super().__init__()
        self.max_torque = max_torque
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, num_joints)
        self.log_std_head = nn.Linear(hidden, num_joints)
        # Small init so initial actions are bounded
        nn.init.uniform_(self.mu_head.weight, -0.05, 0.05)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.uniform_(self.log_std_head.weight, -0.05, 0.05)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        mu = self.mu_head(h) * self.max_torque
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.nn.functional.softplus(log_std) + 1e-6
        return mu, std


class _Critic(nn.Module):
    """Small MLP outputting scalar V(s)."""

    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.05, 0.05)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ContinualDeepRLAgent(BaseAgent):
    """
    Minimal vanilla deep continual RL agent.

    Same interface as the linear continual agent: sensorimotor observation -> torques.
    Uses a small MLP actor (Gaussian policy) and critic, Adam optimizer, one-step TD
    and policy gradient. No replay buffer or target network. Intended as a baseline
    before adding a Stream-X variant (ObGD + Stream AC).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        gamma: float = 0.98,
        lr: float = 1e-3,
        beta_entropy: float = 0.01,
        max_torque: float = 2.0,
        use_intensity_delta: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.lr = lr
        self.beta_entropy = beta_entropy
        self.max_torque = max_torque
        self.use_intensity_delta = use_intensity_delta
        self.hidden_size = hidden_size
        if seed is not None:
            torch.manual_seed(seed)
        self._initialized = False
        self._num_joints = 0
        self._input_dim = 0
        self._prev_intensity: float | None = None
        self._actor: _Actor | None = None
        self._critic: _Critic | None = None
        self._optimizer_policy: torch.optim.Optimizer | None = None
        self._optimizer_value: torch.optim.Optimizer | None = None
        # Stored from last decide for use in next decide (learning step)
        self._prev_feats: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None

    def _initialize(self, observation: Observation) -> None:
        self._num_joints = len(observation.arm_angles)
        self._input_dim = 4 * self._num_joints + 1
        if self.use_intensity_delta:
            self._input_dim += 1
        self._actor = _Actor(
            self._input_dim,
            self._num_joints,
            hidden=self.hidden_size,
            max_torque=self.max_torque,
        )
        self._critic = _Critic(self._input_dim, hidden=self.hidden_size)
        self._optimizer_policy = torch.optim.Adam(self._actor.parameters(), lr=self.lr)
        self._optimizer_value = torch.optim.Adam(self._critic.parameters(), lr=self.lr)
        self._prev_intensity = observation.sound_intensity
        self._initialized = True

    def _build_features(self, observation: Observation) -> np.ndarray:
        feats, self._prev_intensity = _build_features_from_obs(
            observation, self.use_intensity_delta, self._prev_intensity
        )
        return feats

    def decide(self, observation: Observation, reward: float | None = None) -> np.ndarray:
        if not self._initialized:
            self._initialize(observation)
        # Learning: if we have a previous transition (s, a) and reward, run value/policy update
        if self._prev_feats is not None and reward is not None:
            s_prime, self._prev_intensity = _build_features_from_obs(
                observation, self.use_intensity_delta, self._prev_intensity
            )
            s_prime_t = torch.as_tensor(s_prime, dtype=torch.float32).unsqueeze(0)
            V_s = self._critic(self._prev_feats).squeeze(0)
            V_s_prime = self._critic(s_prime_t).squeeze(0)
            delta_t = reward + self.gamma * V_s_prime - V_s
            value_loss = delta_t.pow(2)
            self._optimizer_value.zero_grad()
            value_loss.backward()
            self._optimizer_value.step()
            mu, std = self._actor(self._prev_feats)
            dist = torch.distributions.Normal(mu.squeeze(0), std.squeeze(0))
            log_prob = dist.log_prob(self._prev_action).sum()
            entropy = dist.entropy().sum()
            policy_loss = -delta_t.detach() * log_prob - self.beta_entropy * entropy
            self._optimizer_policy.zero_grad()
            policy_loss.backward()
            self._optimizer_policy.step()
            self._prev_feats = None
            self._prev_action = None
        # Choose action for current observation
        x, self._prev_intensity = _build_features_from_obs(
            observation, self.use_intensity_delta, self._prev_intensity
        )
        x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, std = self._actor(x_t)
            dist = torch.distributions.Normal(mu, std)
            action_t = dist.sample()
        action_np = action_t.squeeze(0).numpy()
        action_np = np.clip(action_np, -self.max_torque, self.max_torque)
        self._prev_feats = x_t.detach()
        self._prev_action = action_t.squeeze(0).detach()
        return action_np
