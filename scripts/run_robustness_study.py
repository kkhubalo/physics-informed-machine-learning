"""Evaluate classical parameter-estimation robustness to observation noise."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.estimation import estimate_parameters
from src.oscillator import simulate_oscillator

def main() -> None:
    output_dir = ROOT / "results"
    figure_dir = ROOT / "figures" / "generated"
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    mass, true_c, true_k = 1.0, 0.2, 4.0
    times = np.linspace(0.0, 15.0, 600)
    clean = simulate_oscillator(times, mass=mass, damping=true_c,
        stiffness=true_k, initial_displacement=1.0, initial_velocity=0.0)[:, 0]
    records = []
    for noise_std in [0.0, 0.01, 0.03, 0.06]:
        for seed in range(10):
            rng = np.random.default_rng(seed)
            observed = clean + rng.normal(0.0, noise_std, len(times))
            fit = estimate_parameters(times, observed, mass=mass,
                initial_displacement=1.0, initial_velocity=0.0)
            records.append({"noise_std_m":noise_std, "seed":seed,
                "damping_estimate":fit.damping, "stiffness_estimate":fit.stiffness,
                "damping_abs_error":abs(fit.damping-true_c),
                "stiffness_abs_error":abs(fit.stiffness-true_k), "fit_rmse_m":fit.rmse})
    raw = pd.DataFrame(records)
    raw.to_csv(output_dir / "robustness_repeats.csv", index=False)
    summary = raw.groupby("noise_std_m").agg(
        damping_mean=("damping_estimate","mean"), damping_std=("damping_estimate","std"),
        stiffness_mean=("stiffness_estimate","mean"), stiffness_std=("stiffness_estimate","std"),
        damping_abs_error_mean=("damping_abs_error","mean"),
        stiffness_abs_error_mean=("stiffness_abs_error","mean")).reset_index()
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].errorbar(summary.noise_std_m, summary.damping_mean,
        yerr=summary.damping_std, marker="o", capsize=4)
    axes[0].axhline(true_c, linestyle="--", label="Ground truth")
    axes[0].set(xlabel="Noise standard deviation (m)", ylabel="Estimated damping (kg/s)")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].errorbar(summary.noise_std_m, summary.stiffness_mean,
        yerr=summary.stiffness_std, marker="o", capsize=4)
    axes[1].axhline(true_k, linestyle="--", label="Ground truth")
    axes[1].set(xlabel="Noise standard deviation (m)", ylabel="Estimated stiffness (kg/s²)")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.suptitle("Parameter-estimation robustness across measurement-noise levels")
    fig.tight_layout(); fig.savefig(figure_dir / "robustness_study.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print("Saved robustness_repeats.csv, robustness_summary.csv and robustness_study.png")

if __name__ == "__main__":
    main()
