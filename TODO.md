### TODO

##### Simulation Performance

- [x] Faster plotting.
- [x] Instances of Soundflower (agent-body-environment triplet) on different threads or processes.
- [x] Use Tensorboard for demo_all.py.
- [ ] Faster inverse kinematics solver?
- [ ] Faster physics engine? (Note: burden is from "collision check" -- minimum distance to a sound source.)

##### Agent Interface

- [x] Full observation mode: expose morphology params and world-frame geometry.
- [x] Sensorimotor mode: joint angles, angular velocities, angular accelerations, plus sound intensity.
- [x] Action observations: motor commands (torque or angular jerk) -- "efference copy".
- [ ] Action: angular jerk instead of torque? (TAI!)

##### Agent

- [x] Classical model-based agent in full observation mode.
- [x] Baseline linear reactive agent in sensorimotor mode.
- [x] Continual linear model-free RL agent in sensorimotor mode.
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

##### Research Topics

- [ ] Unified active audition control: joint torques + mic array gains/phases.
- [ ] Multi-mic sensing: how observation dimensionality affects stability/reacquisition.
- [ ] Multi-source effects and interaction with mic-array sensing.
- [ ] Auditory-motor contingencies for spatial learning (Strawson/Evans sound world).