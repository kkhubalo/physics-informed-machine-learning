import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.neural_models import train_data_driven_model, train_physics_informed_model

def test_data_driven_model_returns_finite_validation_error():
    times = np.linspace(0.0, 2.0, 30)
    displacement = np.exp(-0.1 * times) * np.cos(2.0 * times)
    fit = train_data_driven_model(times, displacement, epochs=5, seed=1)
    assert np.isfinite(fit.training_rmse)
    assert np.isfinite(fit.validation_rmse)

def test_physics_informed_model_learns_positive_parameters():
    times = np.linspace(0.0, 2.0, 30)
    displacement = np.exp(-0.1 * times) * np.cos(2.0 * times)
    fit = train_physics_informed_model(times, displacement, epochs=5, seed=1)
    assert fit.damping is not None and fit.damping > 0
    assert fit.stiffness is not None and fit.stiffness > 0
    assert np.isfinite(fit.validation_rmse)
