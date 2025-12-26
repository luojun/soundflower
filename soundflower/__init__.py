"""Sound Flower - A 2D robotic arm RL environment for sound source tracking."""

__version__ = "0.1.0"

# Re-export from new package structure for backward compatibility
from experimenter import Runner, SoundFlowerConfig, create_default_config
from experimenter.animator import Animator

__all__ = [
    'Runner',
    'Animator',
    'SoundFlowerConfig',
    'create_default_config',
]
