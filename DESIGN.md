# Sound Flower — Design Notes

## Reward and Energy

- **Per-step sound energy (E_t)**: Energy captured by the microphone(s) in one simulation step (Joules). It is computed from sound intensity at the arm tip, microphone area, time step, and orientation factor (inverse-square law and directivity).
- **Sound energy delta (Δ_t)**: Change in per-step energy from the previous step: Δ_t = E_t − E_{t−1}.
- **Reward (learning signal)**: The agent’s per-step reward is the **energy delta** (optionally normalized by a configurable factor). So we reward *improvement* in captured energy from one step to the next, not the absolute level. This encourages the agent to move toward configurations that capture more energy.
- **Sum of rewards (“cumulative reward” in code)**: Sum of rewards over time telescopes to **E_T − E_0** (the difference between the most recent step’s energy and the first step’s energy). It is *not* total energy harvested; it is a diagnostic (net improvement in per-step energy since start).

## Performance

The main **performance metrics** are:

1. **Cumulative sound energy harvested** — Total energy (Joules) captured over the run: Σ_t E_t. This is the primary lifetime performance measure.
2. **Windowed average (last N seconds)** — Average harvest rate over the most recent N seconds (J/s), i.e. (total energy in last N seconds) / N. This measures recent performance and how quickly the agent adapts after changes in the environment or body.

These are good choices for our purposes because:

- We care about **how much** energy the agent harvests (total and recent), not the raw sum of reward deltas. Evaluating on harvest keeps the gap between **learning signal** (energy delta) and **adaptive success** (energy harvested), which helps evaluate the quality of learned representations (e.g. sensorimotor abstraction) by how well they support fast adaptation and sustained harvest.
- Cumulative harvest measures overall success over the run; the windowed average measures recent performance and adaptation speed without being dominated by early transient behavior.

See also [README](README.md) and [TODO](TODO.md) for evaluation experiments (e.g. 1-link validation, sensorimotor abstraction).
