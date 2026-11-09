import numpy as np
import pytest

from src.oscillator import add_measurement_noise, simulate_oscillator


def test_undamped_oscillator_conserves_energy_approximately():
    times = np.linspace(0.0, 10.0, 1000)
    mass, stiffness = 1.0, 4.0

    states = simulate_oscillator(
        times, mass=mass, damping=0.0, stiffness=stiffness
    )
    displacement, velocity = states.T
    energy = 0.5 * mass * velocity**2 + 0.5 * stiffness * displacement**2

    assert np.max(np.abs(energy - energy[0])) < 1e-6


def test_noise_is_reproducible():
    signal = np.ones(100)
    first = add_measurement_noise(signal, noise_std=0.1, seed=7)
    second = add_measurement_noise(signal, noise_std=0.1, seed=7)

    np.testing.assert_array_equal(first, second)


def test_invalid_mass_is_rejected():
    with pytest.raises(ValueError):
        simulate_oscillator(np.linspace(0, 1, 10), mass=0.0)
