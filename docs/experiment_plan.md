# Experimental plan

## Research question

How accurately can the parameters of a dynamical system be inferred from noisy observations, and what is gained by incorporating the governing physical equations?

## Stage 1 — Forward simulation

- Implement the oscillator as a first-order ODE.
- Verify limiting cases and numerical behaviour.
- Generate a reproducible synthetic dataset.
- Keep the true parameters recorded for later evaluation.

## Stage 2 — Classical parameter estimation

Estimate damping and stiffness from displacement observations using a numerical optimisation method. Compare recovered values with the known parameters.

## Stage 3 — Noise and sampling study

Repeat estimation across multiple noise levels, observation durations and sampling densities. Report bias, variance and parameter error.

## Stage 4 — Machine-learning comparison

Establish a conventional baseline before introducing a neural model. Compare a data-driven trajectory fit with a physics-informed objective that penalises violations of the governing differential equation.

## Evaluation

- Parameter relative error.
- Displacement prediction RMSE on held-out time points.
- Robustness to measurement noise.
- Sensitivity to sampling density and initialisation.
- Reproducibility across random seeds.

## Reporting

Each experiment should record its assumptions, parameter values, random seed, software versions, evaluation split and limitations. Results should not be presented as experimental evidence: this first case study uses synthetic data.
