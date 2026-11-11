"""Run parameter estimation and compare data-only and physics-informed models."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.estimation import estimate_parameters
from src.neural_models import train_data_driven_model, train_physics_informed_model
from src.oscillator import add_measurement_noise, simulate_oscillator

def main() -> None:
    results_dir = ROOT / "results"
    figure_dir = ROOT / "figures" / "generated"
    results_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    mass, true_damping, true_stiffness = 1.0, 0.2, 4.0
    times = np.linspace(0.0, 15.0, 600)
    states = simulate_oscillator(times, mass=mass, damping=true_damping,
        stiffness=true_stiffness, initial_displacement=1.0, initial_velocity=0.0)
    observed = add_measurement_noise(states[:, 0], noise_std=0.03, seed=42)
    classical = estimate_parameters(times, observed, mass=mass,
        initial_displacement=1.0, initial_velocity=0.0)
    data_fit = train_data_driven_model(times, observed, epochs=2000, seed=42)
    pinn_fit = train_physics_informed_model(times, observed, mass=mass,
        initial_damping=0.3, initial_stiffness=3.0, physics_weight=1.0,
        epochs=2500, seed=42)
    metrics = pd.DataFrame([
        {"method":"classical nonlinear least squares","training_rmse_m":classical.rmse,
         "validation_rmse_m":np.nan,"damping_estimate":classical.damping,
         "stiffness_estimate":classical.stiffness},
        {"method":"data-driven neural network","training_rmse_m":data_fit.training_rmse,
         "validation_rmse_m":data_fit.validation_rmse,"damping_estimate":np.nan,
         "stiffness_estimate":np.nan},
        {"method":"physics-informed neural network","training_rmse_m":pinn_fit.training_rmse,
         "validation_rmse_m":pinn_fit.validation_rmse,"damping_estimate":pinn_fit.damping,
         "stiffness_estimate":pinn_fit.stiffness},
    ])
    metrics.to_csv(results_dir / "model_comparison.csv", index=False)
    dense = np.linspace(times[0], times[-1], 1000)
    with torch.no_grad():
        tt = torch.as_tensor(dense, dtype=torch.float32).reshape(-1, 1)
        data_curve = data_fit.model(tt).cpu().numpy().ravel()
        pinn_curve = pinn_fit.model(tt).cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, observed, s=8, alpha=0.25, label="Noisy observations")
    ax.plot(times, states[:, 0], linewidth=2, label="Ground truth")
    ax.plot(dense, data_curve, linewidth=1.8, label="Data-driven network")
    ax.plot(dense, pinn_curve, linewidth=1.8, label="Physics-informed network")
    ax.set(xlabel="Time (s)", ylabel="Displacement (m)",
           title="Damped oscillator: data-driven vs physics-informed learning")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Classical fit: c={classical.damping:.5f} kg/s; k={classical.stiffness:.5f} kg/s^2")
    print(f"PINN fit: c={pinn_fit.damping:.5f} kg/s; k={pinn_fit.stiffness:.5f} kg/s^2")
    print(metrics.to_string(index=False))
    print("Saved results/model_comparison.csv and figures/generated/model_comparison.png")

if __name__ == "__main__":
    main()
