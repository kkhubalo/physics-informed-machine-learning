# Physics-Informed Machine Learning for Dynamical Systems

A computational physics project exploring how physical laws and machine-learning methods can work together to identify and model dynamical systems from data.

## Research question

Can we recover the physical parameters of a dynamical system from noisy observations, and how do physics-based estimation methods compare with data-driven approaches?

## Case study 1: Damped harmonic oscillator

We begin with the forced damped oscillator

\[
m\ddot{x}(t)+c\dot{x}(t)+kx(t)=F(t),
\]

where \(m\) is mass, \(c\) is the damping coefficient, and \(k\) is the spring constant.

The initial study will:
1. Derive and implement the governing equations.
2. Simulate the system numerically.
3. Generate synthetic observations with controllable measurement noise.
4. Estimate unknown parameters from the observations.
5. Compare parameter estimates with the known ground truth.
6. Evaluate robustness as noise levels and sampling conditions change.

## Project structure

```text
.
├── data/                  # Generated datasets (not committed by default)
├── docs/                  # Theory, methods and experiment notes
├── scripts/               # Reproducible experiment entry points
├── src/
│   └── oscillator.py      # Physical model and numerical simulation
├── tests/                 # Model and numerical tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting started

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_data.py
```

## Current status

**Stage 1 — Physical model and synthetic data.** The first implementation focuses on a transparent numerical baseline. Parameter estimation and machine-learning comparisons will be added as subsequent, separately documented stages.

## Research standards

- Keep physical assumptions and units explicit.
- Fix random seeds for reproducibility.
- Separate training, validation and test data when learning models are introduced.
- Compare against interpretable physical and numerical baselines.
- Report parameter errors and uncertainty, not only visual fit.
- Record limitations and failed experiments.

## Author

Karabo Khubalo — MSc Physics, University of Modena and Reggio Emilia; Erasmus exchange at Radboud University.

Research interests: computational physics, scientific machine learning, numerical modelling and semiconductor/quantum systems.
