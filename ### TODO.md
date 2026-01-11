### TODO

##### Simulation Performance

- [ ] Difference instances of Soundflower (agent-body-environment triplet) on different threads or processes.
- [ ] Faster inverse kinematics solver?
- [ ] Faster physics engine? (Note: burden is from "collision check" -- minimum distance to a sound source.)

##### Agent Interface

- [ ] Geometric observations: make body configuration observable to accommodate configuration changes.
- [ ] Physical observations: joint angle (relative to "up" link), joint angular velocity, and joint angular acceleration.
- [ ] Action observations: motor commands (torque or angular jerk) -- "efference copy".
- [ ] Action: angular jerk instead of torque?

##### Agent

- [ ] Classical model-based agent: uses both geometric observations and physical observations.
- [ ] Baseline model-free RL agent: uses only physical observations.
- [ ] Continual model-free RL agent: uses only physical observations.
- [ ] RL agents with the "sensorimotor abstraction architecture".

##### Body+Environment Variability

- [ ] Variation in ortibal velocity of sound source.
- [ ] Variation in radius of sound source orbit.
- [ ] Variation in number of sound sources.
- [ ] Variation in friction of joints.
- [ ] Variation in length of body link.
- [ ] Level of observation noise.
- [ ] Level of body+environment dynamics noise.

##### UI

- [ ] A "conductor" interface that changes the range of the body+environment variability.

##### Experiment and Evaluation

- [ ] Run the agents continually.
- [ ] Evaluate in terms of life-time cumulative energy harvest.
- [ ] Evaluate in terms of average latest N seconds of energy harvest.

