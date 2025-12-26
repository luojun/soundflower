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

## Heuristic Agent

The included heuristic agent uses a simple PD controller to point the arm toward the nearest sound source. It can be used as a baseline or starting point for more sophisticated agents.


## License

This project is provided as-is for research and educational purposes.
