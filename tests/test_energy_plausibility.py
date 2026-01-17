"""Test to verify cumulative sound energy is physically plausible."""

import numpy as np
from experimenter.config import create_default_config
from environment import Environment
from agents.approaching_agent import ApproachingAgent
from soundflower.soundflower import SoundFlower


def test_energy_plausibility():
    """
    Test that cumulative sound energy is physically plausible.

    Physical constraints:
    - Maximum possible energy = source_power × simulation_time
    - Actual energy should be less due to distance, orientation, and 4π factor
    - With minimum distance of 0.2m and 4π factor, maximum intensity at closest point is:
      I_max = P / (4π × 0.2²) = P / (4π × 0.04) = P / (0.5026) ≈ 2P
    - Maximum energy per step = I_max × microphone_area × dt
    - Maximum cumulative energy ≈ 2P × A × dt × n_steps = 2P × A × total_time
    """
    config = create_default_config(sound_source_angular_velocity=0.2)
    environment = Environment(config)
    agent = ApproachingAgent()

    soundflower = SoundFlower(
        config=config,
        environment=environment,
        agent=agent,
        logger=None,
        animator=None
    )
    soundflower.start()

    # Run simulation for a fixed duration
    simulation_duration = 10.0  # seconds
    n_steps = int(simulation_duration / config.dt)

    # Run simulation
    for _ in range(n_steps):
        soundflower.step()

    # Calculate maximum physically possible energy
    source_power = config.sound_source_strength  # Watts
    total_time = simulation_duration  # seconds
    max_possible_energy = source_power * total_time  # Joules (if source was at zero distance)

    # With 4π factor and minimum distance constraint:
    # I_max = P / (4π × min_distance²)
    min_distance = config.min_distance_to_source  # meters
    max_intensity = source_power / (4 * np.pi * min_distance ** 2)
    max_energy_per_step = max_intensity * config.microphone_area * config.dt
    max_cumulative_energy = max_energy_per_step * n_steps

    # Get actual cumulative energy
    actual_cumulative_energy = soundflower.cumulative_sound_energy

    print(f"\n=== Energy Plausibility Test ===")
    print(f"Simulation duration: {simulation_duration:.2f} seconds")
    print(f"Number of steps: {n_steps}")
    print(f"Source power: {source_power:.2f} W")
    print(f"Microphone area: {config.microphone_area:.6f} m² ({config.microphone_area * 10000:.2f} cm²)")
    print(f"Minimum distance: {min_distance:.2f} m")
    print(f"\nMaximum possible energy (if source at zero distance): {max_possible_energy:.2f} J")
    print(f"Maximum intensity at minimum distance: {max_intensity:.4f} W/m²")
    print(f"Maximum cumulative energy (with distance/orientation constraints): {max_cumulative_energy:.6f} J")
    print(f"Actual cumulative energy: {actual_cumulative_energy:.6f} J")
    if max_possible_energy > 0:
        print(f"\nEnergy ratio (actual / max possible): {actual_cumulative_energy / max_possible_energy:.6f}")
    if max_cumulative_energy > 0:
        print(f"Energy ratio (actual / max with constraints): {actual_cumulative_energy / max_cumulative_energy:.6f}")

    # Assertions
    assert actual_cumulative_energy <= max_possible_energy, \
        f"Energy {actual_cumulative_energy:.2f} J exceeds maximum possible {max_possible_energy:.2f} J"

    assert actual_cumulative_energy <= max_cumulative_energy * 1.1, \
        f"Energy {actual_cumulative_energy:.2f} J exceeds reasonable maximum {max_cumulative_energy:.2f} J"

    # Additional check: energy should be reasonable
    # Note: energy might be zero if arm doesn't get close enough or orientation is wrong
    assert actual_cumulative_energy >= 0, "Cumulative energy should be non-negative"
    assert actual_cumulative_energy < 1000, \
        f"Energy {actual_cumulative_energy:.6f} J seems unreasonably high for {simulation_duration:.1f}s simulation"

    print("\n✓ All energy plausibility checks passed!")


if __name__ == "__main__":
    test_energy_plausibility()

