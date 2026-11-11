# Experimental plan

## Research question

How accurately can the parameters and trajectory of a dynamical system be inferred from noisy observations, and what changes when the governing physical equation is included in a machine-learning objective?

## Stage 1 — Forward simulation (implemented)

- Implement the oscillator as a first-order ODE.
- Verify limiting cases and numerical behaviour.
- Generate reproducible synthetic observations with known ground-truth parameters.

## Stage 2 — Classical parameter estimation (implemented)

- Fit damping and stiffness by nonlinear least squares.
- Fix mass and initial conditions to known benchmark values.
- Compare recovered coefficients with ground truth.

## Stage 3 — Noise and sampling study (next extension)

Repeat estimation across multiple noise levels, observation durations and sampling densities. Report bias, variance and parameter error over repeated random seeds.

## Stage 4 — Machine-learning comparison (baseline implementation)

- Fit a data-only neural trajectory model.
- Fit a physics-informed neural trajectory model using automatic differentiation and the ODE residual.
- Compare held-out displacement RMSE and recovered physical coefficients.

## Evaluation

- Parameter relative error.
- Displacement prediction RMSE on held-out time points.
- Robustness to measurement noise.
- Sensitivity to sampling density, loss weight and initialisation.
- Reproducibility across random seeds.

## Reporting

Record assumptions, parameter values, random seed, software versions, evaluation split and limitations. This case study uses synthetic data only; it is not experimental evidence. In free decay, displacement alone identifies c/m and k/m, so the benchmark fixes mass.
