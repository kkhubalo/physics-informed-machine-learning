"""Generate a reproducible noisy damped-oscillator dataset."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running this script from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oscillator import add_measurement_noise, simulate_oscillator  # noqa: E402


def main() -> None:
    output_dir = ROOT / "data" / "generated"
    figure_dir = ROOT / "figures" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Illustrative parameters; SI units are used throughout.
    mass = 1.0          # kg
    damping = 0.2       # kg/s
    stiffness = 4.0     # kg/s^2
    noise_std = 0.03    # m

    times = np.linspace(0.0, 15.0, 600)
    states = simulate_oscillator(
        times,
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        initial_displacement=1.0,
        initial_velocity=0.0,
    )

    displacement = states[:, 0]
    noisy_displacement = add_measurement_noise(
        displacement, noise_std=noise_std, seed=42
    )

    dataset = pd.DataFrame(
        {
            "time_s": times,
            "displacement_true_m": displacement,
            "displacement_observed_m": noisy_displacement,
            "velocity_true_m_per_s": states[:, 1],
        }
    )
    csv_path = output_dir / "damped_oscillator.csv"
    dataset.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, displacement, label="Noise-free model", linewidth=2)
    ax.scatter(
        times,
        noisy_displacement,
        label="Noisy observations",
        s=10,
        alpha=0.5,
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Displacement (m)",
        title="Damped harmonic oscillator",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    figure_path = figure_dir / "damped_oscillator.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Saved dataset: {csv_path}")
    print(f"Saved figure:  {figure_path}")
    print(f"Number of observations: {len(dataset)}")


if __name__ == "__main__":
    main()
