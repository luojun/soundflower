### The Sound Flower

Exploration in sensorimotor abstraction for embodied RL agents. 
Reminiscent of "super-sunflower" in chapter 6 of [On the Origin of Objects](https://mitpress.mit.edu/9780262692090/on-the-origin-of-objects/) as well as [Rotato](https://www.semanticscholar.org/paper/Around-the-World-in-80-Steps-Or-How-to-Represent-Milligan-Milligan/9f517de67e4ad87aba2239b646ec67e96de00a8e).

##### Overview

![Animation screenshot with 6 instances running](/assets/animation.png)

Sound Flower is a 2D robotic arm reinforcement learning environment where an agent controls a multi-link robotic arm to track sound sources. The arm is mounted at the center of a circle, with microphones at its tip. The agent receives rewards based on the sound energy captured by the microphones.

![Performance plots for 6 instances](/assets/plots.png)

##### Features

- **Multi-link robotic arm**: Configurable 1-3 DOF arm with physics simulation
- **Sound propagation**: Inverse square law sound propagation from sources
- **Control frequency interface**: Unlike OpenAI Gym's synchronous interface, this environment uses a control interface with configurable frequency for agent-environment interaction
- **Configurable parameters**: Extensive configuration options for arm dynamics, sound sources, and environment properties
- **Heuristic agent**: Included simple heuristic agent for demonstration
- **Animation**: Real-time animation package for visualizing simulation

##### Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo with animation:

```bash
python3 demo.py # Or demo_all.py
```

Run the demo in headless mode (no animation, faster execution):

```bash
python3 demo.py --headless # or demo_all.py --headless
```

##### Heuristic Agents

The included heuristic agents use a simple PD controller to point the arm toward the nearest sound source. They can be used as baselines or starting points for more sophisticated agents.


##### License

This project is provided as-is for research and educational purposes.
