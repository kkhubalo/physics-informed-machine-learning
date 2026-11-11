"""Classical parameter estimation for a damped harmonic oscillator.

For free decay, displacement data identify c/m and k/m, not all three
absolute coefficients independently. Mass and initial conditions are fixed
inputs; damping and stiffness are estimated by nonlinear least squares.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from src.oscillator import simulate_oscillator

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class ParameterEstimate:
    mass: float
    damping: float
    stiffness: float
    rmse: float
    success: bool
    message: str

def estimate_parameters(times: FloatArray, observed_displacement: FloatArray, *,
                        mass: float = 1.0, initial_displacement: float | None = None,
                        initial_velocity: float = 0.0, initial_damping: float = 0.3,
                        initial_stiffness: float = 3.0) -> ParameterEstimate:
    """Fit damping and stiffness while treating mass and initial state as known."""
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed_displacement, dtype=float)
    if times.ndim != 1 or observed.ndim != 1 or times.size != observed.size:
        raise ValueError("times and observations must be equal-length 1D arrays")
    if times.size < 5 or not np.all(np.isfinite(observed)):
        raise ValueError("at least five finite observations are required")
    if mass <= 0 or initial_damping <= 0 or initial_stiffness <= 0:
        raise ValueError("mass and initial parameter guesses must be positive")
    x0 = float(observed[0] if initial_displacement is None else initial_displacement)

    def residual(log_parameters: FloatArray) -> FloatArray:
        damping, stiffness = np.exp(log_parameters)
        prediction = simulate_oscillator(times, mass=mass, damping=float(damping),
            stiffness=float(stiffness), initial_displacement=x0,
            initial_velocity=initial_velocity)[:, 0]
        return prediction - observed

    result = least_squares(residual, np.log([initial_damping, initial_stiffness]),
        bounds=(np.log([1e-6, 1e-6]), np.log([1e3, 1e3])), x_scale="jac")
    damping, stiffness = np.exp(result.x)
    return ParameterEstimate(float(mass), float(damping), float(stiffness),
        float(np.sqrt(np.mean(result.fun**2))), bool(result.success), str(result.message))
