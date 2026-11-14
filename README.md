# Physics-Informed Machine Learning for Dynamical Systems

A reproducible computational-physics project investigating how physical laws and machine-learning methods can work together to identify and model dynamical systems from data.

## Research question

Can we recover the parameters and trajectory of a dynamical system from noisy observations, and what is gained by incorporating the governing physical equation into a machine-learning model?

## Case study: damped harmonic oscillator

We study the unforced damped oscillator:

\[m\\ddot{x}(t)+c\\dot{x}(t)+kx(t)=0.\]

The project contains three modelling levels:

1. **Forward numerical simulation:** solve the governing ODE and generate reproducible synthetic observations.
2. **Classical parameter estimation:** recover damping and stiffness from displacement observations using nonlinear least squares.
3. **Scientific machine learning:** compare a data-only neural trajectory model with a physics-informed neural network (PINN) that penalises violations of the ODE.

> **Identifiability:** For unforced motion, displacement data identify c/m and k/m. They cannot determine all three absolute coefficients independently. The benchmark therefore fixes mass and estimates damping and stiffness. See docs/parameter_estimation.md.

## Repository structure

- src/oscillator.py — physical model and numerical simulation
- src/estimation.py — classical parameter estimation
- src/neural_models.py — data-driven and physics-informed MLPs
- scripts/generate_data.py — reproducible synthetic dataset generation
- scripts/run_analysis.py — parameter estimation and neural-model comparison\n- scripts/run_robustness_study.py — repeated parameter-recovery experiments across noise levels
- docs/ — physical model, estimation method and PINN formulation
- tests/ — numerical and estimation tests

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the experiments

Generate the synthetic observations and initial plot:

```bash
python scripts/generate_data.py
```

Run classical parameter estimation and compare the neural models:

```bash
python scripts/run_analysis.py
```

The analysis writes results/model_comparison.csv and figures/generated/model_comparison.png. Both neural models use a deterministic 80/20 index split and fixed random seed.

Run tests:

```bash
python -m pytest -q
```

## Dependencies

NumPy, SciPy, pandas, Matplotlib, Pytest and PyTorch.


## Current results

The current benchmark uses a damped harmonic oscillator with fixed mass
\(m=1.0\,\mathrm{kg}\), reference damping \(c=0.20\,\mathrm{kg/s}\),
reference stiffness \(k=4.00\,\mathrm{kg/s^2}\), and 3% displacement
observation noise (standard deviation 0.03 m, seed 42).

| Method | Damping estimate (kg/s) | Relative damping error | Stiffness estimate (kg/s²) | Relative stiffness error |
|---|---:|---:|---:|---:|
| Classical nonlinear least squares | 0.20044 | 0.22% | 4.00019 | 0.0047% |
| Physics-informed neural network | 0.21582 | 7.91% | 3.98445 | 0.39% |

For the neural models, the PINN achieved a validation RMSE of 0.05314 m,
compared with 0.05624 m for the data-driven neural network (about 5.5% lower
validation RMSE in this run). The classical fit had a training RMSE of
0.02920 m; this is a fit to the observed data and is not directly comparable
to the neural models' held-out validation RMSE.

### Classical-estimator noise robustness

The robustness script performs 10 repeated classical nonlinear least-squares
fits at each observation-noise level. Mean absolute parameter errors were:

| Noise standard deviation (m) | Mean absolute damping error (kg/s) | Mean absolute stiffness error (kg/s²) |
|---:|---:|---:|
| 0.00 | approximately 0 | approximately 0 |
| 0.01 | 0.000301 | 0.000465 |
| 0.03 | 0.000905 | 0.001393 |
| 0.06 | 0.001810 | 0.002778 |

**Scope note:** the repeated noise-robustness experiment currently evaluates
the classical estimator only. A repeated PINN-versus-NLS robustness
comparison across noise levels remains future work. These are synthetic-data
results under the stated model assumptions, not experimental validation.

## Research standards and limitations

- State physical assumptions and SI units explicitly.
- Fix random seeds for reproducibility.
- Compare against an interpretable numerical baseline.
- Report parameter errors and held-out prediction errors, not only plots.
- Treat synthetic-data results as a controlled benchmark, not experimental evidence.
- Do not claim a PINN is better unless measured results support that conclusion.

## Author

Karabo Kelvin Khubalo — MSc Physics, University of Modena and Reggio Emilia; Erasmus Exchange,Radboud University.

Research interests: computational physics, scientific machine learning, numerical modelling and semiconductor/quantum systems.
