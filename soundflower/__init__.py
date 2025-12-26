"""Sound Flower - A 2D robotic arm RL environment for sound source tracking."""

__version__ = "0.1.0"

# Re-export from new package structure for backward compatibility
from .world import World, WorldState, Observation
from experimenter import Runner, SoundFlowerConfig, create_default_config
from experimenter.animator import Animator

__all__ = [
    'World',
    'WorldState',
    'Observation',
    'Runner',
    'Animator',
    'SoundFlowerConfig',
    'create_default_config',
]
