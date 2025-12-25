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
- **Visualization**: Comprehensive visualization package with static plots and real-time animations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the basic demo:

```bash
python3 demo.py
```

Run the matplotlib visualization demo:

```bash
python3 demo_visualization.py
```

Run the Pygame real-time animation demo (recommended for dynamic visualization):

```bash
python3 demo_pygame.py
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
├── experiments/          # Experiment runners
│   ├── __init__.py
│   └── experiment.py     # Experiment class for running episodes
├── visualization/        # Visualization package
│   ├── __init__.py
│   ├── visualizer.py     # Matplotlib visualization tools
│   └── pygame_visualizer.py  # Pygame real-time animation
├── demo.py              # Basic demo script
├── demo_visualization.py # Matplotlib visualization demo
├── demo_pygame.py        # Pygame real-time animation demo
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

### Creating an Environment and Agent

```python
from soundflower.environment import SoundFlowerEnvironment
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent
from experiments import Experiment

config = SoundFlowerConfig(
    num_links=2,
    link_lengths=[0.6, 0.4],
    circle_radius=1.0
)

env = SoundFlowerEnvironment(config)
agent = HeuristicAgent(kp=8.0, kd=1.0, max_torque=config.max_torque)
```

### Running an Episode

```python
import asyncio
from experiments import Experiment

# Create experiment
experiment = Experiment(env, agent)

# Run episode
async def run_episode():
    stats = await experiment.run_episode(max_steps=1000, render=False)
    print(f"Total reward: {stats['total_reward']:.4f}")
    print(f"Steps: {stats['steps']}")

asyncio.run(run_episode())
```

### Manual Episode Control

```python
import asyncio

async def manual_episode():
    observation = env.reset()
    
    for step in range(1000):
        # Your agent selects an action (torques for each joint)
        action = await agent.select_action(observation)
        
        # Step the environment asynchronously
        observation, reward, done, info = await env.step(action)
        
        # Reward is the change in sound energy (delta)
        print(f"Step {step}: Reward = {reward:.4f}, Sound Energy = {observation.sound_energy:.4f}")

asyncio.run(manual_episode())
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

**Note:** The agent no longer holds a reference to the environment. The `max_torque` parameter is passed during initialization for action clamping.

## Experiments Package

The `experiments` package provides the `Experiment` class for running episodes and collecting statistics:

```python
from experiments import Experiment

experiment = Experiment(env, agent)

# Run a simple episode
stats = await experiment.run_episode(max_steps=1000, render=False)

# Run with logging
stats = await experiment.run_episode_with_logging(max_steps=100, log_interval=20)

# Collect render data for visualization
render_data = await experiment.collect_render_data(max_steps=100)
```

This separation of concerns makes the code more modular and easier to test.

## Visualization

The visualization package provides multiple ways to visualize the environment:

### Pygame Real-time Animation (Recommended)

The Pygame visualizer provides smooth real-time animation with interactive controls:

```python
from visualization import PygameVisualizer

visualizer = PygameVisualizer(
    circle_radius=config.circle_radius,
    link_lengths=config.link_lengths,
    window_size=(800, 800),
    fps=60
)

# Animate in real-time
await visualizer.animate(env, agent, max_steps=10000, steps_per_frame=1)
```

**Features:**
- Smooth 60 FPS real-time rendering
- Animated sound waves (pulsing circles)
- Interactive controls (SPACE to pause, ESC/Q to quit)
- Real-time display of step count, sound energy, and positions
- Shows arm motion, sound source motion, and dynamic updates

### Matplotlib Visualization

For static plots and recorded animations:

```python
from visualization import SoundFlowerVisualizer

visualizer = SoundFlowerVisualizer(
    circle_radius=config.circle_radius,
    link_lengths=config.link_lengths
)

# Static plot
render_data = env.render()
visualizer.plot_state(render_data, observation=observation, show=True)

# Real-time animation (matplotlib)
await visualizer.animate_async(env, agent, max_steps=200, update_interval=0.05)

# Create animation from data
anim = visualizer.create_animation(render_data_sequence, interval=50)
```

**Visualization Elements:**
- The circular environment boundary
- The robotic arm (links and joints)
- The end effector with microphone
- Sound sources with animated wave visualization
- Real-time sound energy display
- Step counter and position information
- Grid background for reference

## License

This project is provided as-is for research and educational purposes.
