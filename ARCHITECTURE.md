# Sound Flower Architecture

## Overview

Sound Flower uses a decoupled architecture inspired by patterns from projects like OmegaZero, with clear separation between:

1. **World** - Simulation state and logic
2. **Runner** - Orchestrates simulation execution  
3. **Renderer** - Handles visualization (optional)

## Architecture Components

### World (`soundflower/world.py`)

The `World` class represents the simulation state and logic. It:

- Manages the physics engine
- Provides methods to query current state
- Computes observations and rewards
- Applies actions to the simulation
- Can be reset to initial state

**Key Methods:**
- `get_state()` - Get current world state (observation, reward, info)
- `apply_action(action)` - Apply action to the world
- `get_render_data()` - Get data for visualization
- `reset()` - Reset to initial state
- `start_physics()` / `stop_physics()` - Control physics engine

### Runner (`soundflower/runner.py`)

The `Runner` class orchestrates simulation execution. It:

- Coordinates between World and Agent
- Runs control loop at configurable frequency (10-100 Hz)
- Manages callbacks for observations and steps
- Can run for duration or number of steps

**Key Methods:**
- `start()` / `stop()` - Start/stop the runner
- `run(duration)` - Run for specified duration
- `run_for_steps(max_steps)` - Run for specified number of steps
- `set_observation_callback()` - Set callback for observations
- `set_step_callback()` - Set callback for each step

### Renderer (`soundflower/renderer.py`)

The `Renderer` class handles visualization. It:

- Decoupled from simulation logic
- Runs at configurable frame rate (10-100 fps)
- Can be attached/detached from running simulation
- Calls render callback with world state

**Key Methods:**
- `start()` / `stop()` - Start/stop rendering
- `set_fps(fps)` - Change frame rate dynamically

## Decoupled Frequencies

The architecture supports independent frequencies:

1. **Physics**: Runs as fast as possible with configurable time step (`config.dt`)
2. **Control**: Runs at `config.control_frequency` (10-100 Hz)
3. **Visualization**: Runs at `config.visualization_fps` (10-100 fps, optional)

This allows:
- **Headless mode**: Run physics + control without visualization (much faster than real-time)
- **Flexible visualization**: Adjust FPS based on rendering performance
- **Independent control**: Agent can run at different frequency than visualization

## Usage Example

```python
import asyncio
from soundflower import World, Runner, Renderer
from soundflower.config import SoundFlowerConfig
from agents.heuristic_agent import HeuristicAgent

# Create configuration
config = SoundFlowerConfig(
    control_frequency=50.0,  # 50 Hz control
    visualization_fps=60.0,   # 60 fps visualization
    dt=0.01                   # 10ms physics time step
)

# Create World (simulation state)
world = World(config)

# Create Agent
agent = HeuristicAgent(config=config)

# Create Runner (orchestrates simulation)
runner = Runner(world, agent, config)

# Create Renderer (optional, for visualization)
def render_callback(render_data):
    # Render the world state
    pass

renderer = Renderer(world, config, render_callback, fps=60.0)

# Run simulation
async def main():
    await world.reset()
    world.start_physics()
    runner.start()
    renderer.start()  # Optional
    
    try:
        await asyncio.sleep(30.0)  # Run for 30 seconds
    finally:
        renderer.stop()
        runner.stop()
        world.stop_physics()

asyncio.run(main())
```

## Headless Mode

For faster-than-real-time execution without visualization:

```python
# No renderer needed
world = World(config)
runner = Runner(world, agent, config)

await world.reset()
world.start_physics()
runner.start()

# This will run much faster than real-time
await runner.run(duration=10.0)  # 10 seconds simulated time
```

## Benefits

1. **Separation of Concerns**: Each component has a clear responsibility
2. **Flexibility**: Can run with or without visualization
3. **Performance**: Headless mode runs much faster than real-time
4. **Modularity**: Easy to swap components (e.g., different renderers)
5. **Testability**: Each component can be tested independently
6. **Scalability**: Can run multiple simulations in parallel

## Comparison with Previous Architecture

**Before**: Environment tightly coupled physics, control, and visualization

**After**: 
- `World` = Environment state and logic
- `Runner` = Experiment/Episode orchestration
- `Renderer` = Visualization (completely optional)

This matches patterns from projects like OmegaZero where World, Runner, and Renderer are clearly separated.

