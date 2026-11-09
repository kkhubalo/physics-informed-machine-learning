"""Numerical model for a damped harmonic oscillator.

The state is y = [x, v], where x is displacement and v is velocity.
The equation of motion is

    m * x_ddot + c * x_dot + k * x = F(t).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def oscillator_rhs(
    time: float,
    state: FloatArray,
    mass: float,
    damping: float,
    stiffness: float,
    force: Callable[[float], float] | None = None,
) -> FloatArray:
    """Return the time derivative [velocity, acceleration]."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if stiffness <= 0:
        raise ValueError("stiffness must be positive")

    displacement, velocity = state
    applied_force = 0.0 if force is None else float(force(time))
    acceleration = (
        applied_force - damping * velocity - stiffness * displacement
    ) / mass

    return np.asarray([velocity, acceleration], dtype=float)


def simulate_oscillator(
    times: FloatArray,
    mass: float = 1.0,
    damping: float = 0.2,
    stiffness: float = 4.0,
    initial_displacement: float = 1.0,
    initial_velocity: float = 0.0,
    force: Callable[[float], float] | None = None,
) -> FloatArray:
    """Simulate displacement and velocity at the requested times.

    Returns
    -------
    ndarray, shape (n_times, 2)
        Column 0 is displacement; column 1 is velocity.
    """
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")

    initial_state = np.asarray(
        [initial_displacement, initial_velocity], dtype=float
    )

    solution = solve_ivp(
        fun=lambda t, y: oscillator_rhs(
            t, y, mass, damping, stiffness, force
        ),
        t_span=(float(times[0]), float(times[-1])),
        y0=initial_state,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution.y.T


def add_measurement_noise(
    displacement: FloatArray,
    noise_std: float,
    seed: int = 42,
) -> FloatArray:
    """Add zero-mean Gaussian measurement noise to displacement."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    displacement = np.asarray(displacement, dtype=float)
    return displacement + rng.normal(0.0, noise_std, size=displacement.shape)
