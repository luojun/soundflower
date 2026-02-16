"""Agent smoke tests for action shapes and finite outputs."""

import numpy as np
import pytest

from agents.approaching_agent import ApproachingAgent
from agents.continual_linear_rl_agent import ContinualLinearRLAgent
from agents.linear_reactive_agent import LinearReactiveAgent
from agents.pointing_agent import PointingAgent
from agents.tracking_agent import TrackingAgent
from environment import Environment
from experimenter.config import create_default_config

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _make_env(num_links: int, observation_mode: str) -> Environment:
    config = create_default_config(sound_source_angular_velocity=0.1)
    config.num_links = num_links
    if num_links == 2:
        config.link_lengths = [0.6, 0.4]
        config.link_masses = [6.0, 6.0]
        config.joint_frictions = [1.0, 1.0]
    elif num_links == 3:
        config.link_lengths = [0.6, 0.4, 0.3]
        config.link_masses = [6.0, 6.0, 3.0]
        config.joint_frictions = [1.0, 1.0, 1.0]
    config.observation_mode = observation_mode
    config.__post_init__()
    return Environment(config)


def _assert_action(agent, observation, num_links: int, reward: float | None = None) -> np.ndarray:
    action = agent.decide(observation, reward)
    assert action.shape == (num_links,)
    assert np.all(np.isfinite(action))
    return action


def test_pointing_agent_action_shape():
    env = _make_env(num_links=2, observation_mode="full")
    observation = env.get_state().observation
    _assert_action(PointingAgent(), observation, num_links=2)


def test_tracking_agent_action_shape():
    env = _make_env(num_links=2, observation_mode="full")
    observation = env.get_state().observation
    _assert_action(TrackingAgent(), observation, num_links=2)


def test_approaching_agent_action_shape():
    env = _make_env(num_links=2, observation_mode="full")
    observation = env.get_state().observation
    _assert_action(ApproachingAgent(), observation, num_links=2)


def test_linear_reactive_agent_action_shape():
    env = _make_env(num_links=2, observation_mode="sensorimotor")
    observation = env.get_state().observation
    _assert_action(LinearReactiveAgent(), observation, num_links=2)


def test_continual_linear_rl_agent_step():
    env = _make_env(num_links=2, observation_mode="sensorimotor")
    agent = ContinualLinearRLAgent()

    state = env.get_state()
    action = _assert_action(agent, state.observation, num_links=2)
    env.apply(action)
    env.step()

    next_state = env.get_state()
    _assert_action(agent, next_state.observation, num_links=2, reward=next_state.reward)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_continual_deep_rl_agent_step():
    from agents.continual_deep_rl_agent import ContinualDeepRLAgent
    env = _make_env(num_links=2, observation_mode="sensorimotor")
    agent = ContinualDeepRLAgent()

    state = env.get_state()
    action = _assert_action(agent, state.observation, num_links=2)
    env.apply(action)
    env.step()

    next_state = env.get_state()
    _assert_action(agent, next_state.observation, num_links=2, reward=next_state.reward)
