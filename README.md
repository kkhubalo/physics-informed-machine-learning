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

## Research standards and limitations

- State physical assumptions and SI units explicitly.
- Fix random seeds for reproducibility.
- Compare against an interpretable numerical baseline.
- Report parameter errors and held-out prediction errors, not only plots.
- Treat synthetic-data results as a controlled benchmark, not experimental evidence.
- Do not claim a PINN is better unless measured results support that conclusion.

## Author

Karabo Khubalo — MSc Physics, University of Modena and Reggio Emilia; Erasmus exchange at Radboud University.

Research interests: computational physics, scientific machine learning, numerical modelling and semiconductor/quantum systems.
