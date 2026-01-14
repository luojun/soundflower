### TODO

##### Simulation Performance

- [x] Faster plotting.
- [x] Instances of Soundflower (agent-body-environment triplet) on different threads or processes.
- [ ] Use Tensorboard for demo_all.py.
- [ ] Faster inverse kinematics solver?
- [ ] Faster physics engine? (Note: burden is from "collision check" -- minimum distance to a sound source.)

##### Agent Interface

- [ ] Geometric observations: make body configuration observable to accommodate configuration changes.
- [ ] Physical observations: joint angle (relative to "up" link), joint angular velocity, and joint angular acceleration.
- [ ] Action observations: motor commands (torque or angular jerk) -- "efference copy".
- [ ] Action: angular jerk instead of torque? (TAI!)

##### Agent

- [ ] Classical model-based agent: uses both geometric observations and physical observations.
- [ ] Baseline model-free RL agent: uses only physical observations and action observation.
- [ ] Continual model-free RL agent: uses only physical observations and action observation.
- [ ] Model-free RL agents with the "sensorimotor abstraction architecture".
- [ ] Exploration: link to the OaK architecture.
- [ ] Exploration: Soundflower as a computational neuroscience "animal model" for sensorimotor abstraction.

##### Body+Environment Variability

- [ ] Variation in ortibal velocity of sound source.
- [ ] Variation in radius of sound source orbit.
- [ ] Variation in number of sound sources.
- [ ] Variation in friction of joints.
- [ ] Variation in range of joint rotation.
- [ ] Variation in length of body links.
- [ ] Level of observation noise.
- [ ] Level of body+environment dynamics noise.

##### UI

- [ ] "Conductor" GUI for controlling body+environment variability.
- [ ] "Conductor" keyboard UI for controlling body+environment variability.

##### Experiment and Evaluation

- [ ] Run the agents continually.
- [ ] Evaluate in terms of life-time cumulative energy harvest.
- [ ] Evaluate in terms of average latest N seconds of energy harvest.

