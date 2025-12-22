### The Sound Flower

Exploration in sensorimotor abstraction for embodied RL Agent.

Reminiscent of the Super Sunflower as well as Rotato.

## Overview

Sound Flower is a 2D robotic arm reinforcement learning environment where an agent controls a multi-link robotic arm to track sound sources. The arm is mounted at the center of a circle, with microphones at its tip. The agent receives rewards based on the sound energy captured by the microphones.

## Features

- **Multi-link robotic arm**: Configurable 1-3 DOF arm with physics simulation
- **Sound propagation**: Inverse square law sound propagation from sources
- **Asynchronous interface**: Unlike OpenAI Gym's synchronous interface, this environment uses async/await for more flexible agent-environment interaction
- **Configurable parameters**: Extensive configuration options for arm dynamics, sound sources, and environment properties
- **Heuristic agent**: Included simple heuristic agent for demonstration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo:

```bash
python demo.py
```

## Project Structure

```
soundflower/
├── soundflower/          # Main package
│   ├── __init__.py
│   ├── config.py         # Configuration parameters
│   ├── physics.py        # Physics simulation (arm dynamics, sound propagation)
│   └── environment.py   # Asynchronous RL environment
├── agents/               # Agent implementations
│   ├── __init__.py
│   └── heuristic_agent.py  # Simple heuristic agent
├── demo.py              # Demo script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

The environment can be configured using `SoundFlowerConfig`:

- **Arm configuration**:
  - `num_links`: Number of arm links (1-3)
  - `link_lengths`: Length of each link
  - `link_masses`: Mass of each link (concentrated at midpoint)
  - `joint_frictions`: Friction coefficient at each joint

- **Environment**:
  - `circle_radius`: Radius of the circle where sound sources appear

- **Microphones**:
  - `num_microphones`: Number of microphones at arm tip
  - `microphone_gain`: Gain of microphones

- **Sound sources**:
  - `num_sound_sources`: Number of sound sources
  - `sound_source_strength`: Base strength of sound sources
  - `sound_attenuation_coeff`: Coefficient for inverse square law
  - `sound_source_angular_velocity`: Angular velocity for moving sources
  - `sound_source_initial_angle`: Initial angle of sound source

- **Physics**:
  - `dt`: Time step for simulation
  - `max_torque`: Maximum torque that can be applied at joints

## Usage

### Creating an Environment

```python
from soundflower.environment import SoundFlowerEnvironment
from soundflower.config import SoundFlowerConfig

config = SoundFlowerConfig(
    num_links=2,
    link_lengths=[0.6, 0.4],
    circle_radius=1.0
)

env = SoundFlowerEnvironment(config)
```

### Running an Episode

```python
import asyncio

async def run_episode():
    observation = env.reset()
    
    for step in range(1000):
        # Your agent selects an action (torques for each joint)
        action = np.array([0.5, -0.3])  # Example action
        
        # Step the environment asynchronously
        observation, reward, done, info = await env.step(action)
        
        # Reward is the change in sound energy (delta)
        print(f"Step {step}: Reward = {reward:.4f}, Sound Energy = {observation.sound_energy:.4f}")

asyncio.run(run_episode())
```

### Observation Space

The observation includes:
- `arm_angles`: Current joint angles (rad)
- `arm_angular_velocities`: Current joint angular velocities (rad/s)
- `end_effector_pos`: End effector position (x, y)
- `sound_energy`: Current sound energy at microphone
- `sound_energy_delta`: Change in sound energy (used for reward)
- `sound_source_positions`: Positions of all sound sources

### Action Space

Actions are torques applied at each joint:
- Shape: `[num_links]`
- Range: `[-max_torque, max_torque]`

## Physics

- **Arm dynamics**: Simplified dynamics with friction at joints
- **Sound propagation**: Inverse square law (energy ∝ 1/r²)
- **No gravity**: 2D environment without gravity
- **No self-collision**: Links don't collide with each other

## Heuristic Agent

The included heuristic agent uses a simple PD controller to point the arm toward the nearest sound source. It can be used as a baseline or starting point for more sophisticated agents.

## License

This project is provided as-is for research and educational purposes.
