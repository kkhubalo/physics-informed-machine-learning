import numpy as np
from src.estimation import estimate_parameters
from src.oscillator import simulate_oscillator

def test_recovers_noise_free_parameters():
    times = np.linspace(0.0, 12.0, 300)
    states = simulate_oscillator(times, mass=1.0, damping=0.2, stiffness=4.0,
                                 initial_displacement=1.0, initial_velocity=0.0)
    fit = estimate_parameters(times, states[:, 0], mass=1.0,
                              initial_displacement=1.0, initial_velocity=0.0)
    assert fit.success
    assert abs(fit.damping - 0.2) < 1e-3
    assert abs(fit.stiffness - 4.0) < 1e-3
    assert fit.rmse < 1e-5

def test_recovers_parameters_with_measurement_noise():
    times = np.linspace(0.0, 15.0, 600)
    states = simulate_oscillator(times, mass=1.0, damping=0.2, stiffness=4.0,
                                 initial_displacement=1.0, initial_velocity=0.0)
    observed = states[:, 0] + np.random.default_rng(42).normal(0.0, 0.01, len(times))
    fit = estimate_parameters(times, observed, mass=1.0,
                              initial_displacement=1.0, initial_velocity=0.0)
    assert fit.success
    assert abs(fit.damping - 0.2) < 0.05
    assert abs(fit.stiffness - 4.0) < 0.1
