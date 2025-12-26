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
- **Animation**: Real-time animation package for visualizing simulation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo with animation:

```bash
python3 demo.py
```

Run the demo in headless mode (no animation, faster execution):

```bash
python3 demo.py --headless
```

## Project Structure

```
soundflower/
├── environment/         # Core simulation package
│   ├── __init__.py
│   ├── physics.py       # Physics simulation (arm dynamics, sound propagation)
│   └── physics_engine.py  # Asynchronous physics engine
├── experimenter/        # Experiment orchestration package
│   ├── __init__.py
│   ├── config.py        # Configuration parameters
│   └── runner.py        # Runner for orchestrating simulation
├── animator/            # Visualization package
│   ├── __init__.py
│   ├── renderer.py      # Renderer interface
│   └── pygame_framer.py # Pygame real-time framer
├── agents/              # Agent implementations
│   ├── __init__.py
│   └── heuristic_agent.py  # Simple heuristic agent
├── soundflower/         # Main package
│   ├── __init__.py      # Package exports
│   └── world.py         # World/Environment interface
├── demo.py              # Demo script (supports --headless flag)
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

### Creating a World and Agent

```python
from soundflower import World
from experimenter import SoundFlowerConfig, create_default_config
from agents.heuristic_agent import HeuristicAgent

# Create configuration
config = create_default_config(sound_source_angular_velocity=0.3)
# Or create custom config:
# config = SoundFlowerConfig(
#     num_links=2,
#     link_lengths=[0.6, 0.4],
#     circle_radius=1.0
# )

# Create World (simulation state)
world = World(config)

# Create Agent
agent = HeuristicAgent(kp=8.0, kd=1.0, config=config)
```

### Running a Simulation

```python
import asyncio
from experimenter import Runner
from animator import Renderer, PygameFramer

async def run_simulation():
    # Create Runner (orchestrates simulation)
    runner = Runner(world, agent, config)
    
    # Optional: Create Renderer for visualization
    framer = PygameFramer(
        circle_radius=config.circle_radius,
        link_lengths=config.link_lengths,
        window_size=(800, 800),
        fps=60.0
    )
    
    renderer = Renderer(
        world=world,
        config=config,
        render_callback=framer.render_callback,
        fps=60.0
    )
    
    # Reset and start
    await world.reset()
    world.start_physics()
    runner.start()
    renderer.start()  # Optional
    
    # Track statistics
    total_reward = 0.0
    def step_callback(state):
        nonlocal total_reward
        total_reward += state.reward
        print(f"Reward: {state.reward:.4f}, Sound Energy: {state.observation.sound_energy:.4f}")
    
    runner.set_step_callback(step_callback)
    
    # Run for 10 seconds
    await asyncio.sleep(10.0)
    
    # Cleanup
    renderer.stop()
    runner.stop()
    world.stop_physics()
    framer.close()
    
    print(f"Total reward: {total_reward:.4f}")

asyncio.run(run_simulation())
```

### Headless Mode (No Visualization)

```python
import asyncio
from soundflower import World
from experimenter import Runner, create_default_config
from agents.heuristic_agent import HeuristicAgent

async def run_headless():
    config = create_default_config()
    config.control_frequency = 100.0  # Higher frequency for faster execution
    config.headless = True
    
    world = World(config)
    agent = HeuristicAgent(config=config)
    runner = Runner(world, agent, config)
    
    await world.reset()
    world.start_physics()
    runner.start()
    
    # Run for 10 seconds (much faster than real-time)
    await asyncio.sleep(10.0)
    
    runner.stop()
    world.stop_physics()
    
    final_state = world.get_state()
    print(f"Final sound energy: {final_state.observation.sound_energy:.4f}")

asyncio.run(run_headless())
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

## Experimenter Package

The `experimenter` package provides configuration and the `Runner` class for orchestrating simulations:

```python
from experimenter import Runner, create_default_config, SoundFlowerConfig
from soundflower import World
from agents.heuristic_agent import HeuristicAgent

# Create configuration
config = create_default_config(sound_source_angular_velocity=0.3)

# Create world and agent
world = World(config)
agent = HeuristicAgent(config=config)

# Create runner
runner = Runner(world, agent, config)

# Set callbacks for monitoring
def step_callback(state):
    print(f"Step: Reward={state.reward:.4f}, Energy={state.observation.sound_energy:.4f}")

runner.set_step_callback(step_callback)

# Run simulation
await world.reset()
world.start_physics()
runner.start()
await asyncio.sleep(10.0)  # Run for 10 seconds
runner.stop()
world.stop_physics()
```

The `Runner` coordinates between the `World` (simulation state) and the `Agent` (decision making), running at a configurable control frequency.

## Animation

The `animator` package provides real-time visualization of the simulation:

### Pygame Framer

The `PygameFramer` provides smooth real-time animation with interactive controls:

```python
from animator import PygameFramer, Renderer
from soundflower import World
from experimenter import Runner, create_default_config
from agents.heuristic_agent import HeuristicAgent

# Create configuration
config = create_default_config(sound_source_angular_velocity=0.3)

# Create world, runner, and framer
world = World(config)
agent = HeuristicAgent(config=config)
runner = Runner(world, agent, config)

framer = PygameFramer(
    circle_radius=config.circle_radius,
    link_lengths=config.link_lengths,
    window_size=(800, 800),
    fps=60
)

# Create renderer with framer callback
renderer = Renderer(
    world=world,
    config=config,
    render_callback=framer.render_callback,
    fps=60.0
)

# Run simulation
await world.reset()
world.start_physics()
runner.start()
renderer.start()

# Run until user quits or timeout
while renderer.is_running:
    await asyncio.sleep(0.1)

# Cleanup
renderer.stop()
runner.stop()
world.stop_physics()
framer.close()
```

**Features:**
- Smooth 60 FPS real-time rendering
- Animated sound waves (pulsing circles)
- Interactive controls (SPACE to pause, ESC/Q to quit)
- Real-time display of step count, sound energy, and positions
- Shows arm motion, sound source motion, and dynamic updates
- Decoupled from simulation - can run at different frame rates

**Animation Elements:**
- The circular environment boundary
- The robotic arm (links and joints)
- The end effector with microphone
- Sound sources with animated wave visualization
- Real-time sound energy display
- Step counter and position information
- Grid background for reference

## License

This project is provided as-is for research and educational purposes.
