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

## Stage 3 — Noise robustness (implemented)

- Repeat estimation over four measurement-noise levels.
- Use ten independent random seeds at each level.
- Report mean estimates, standard deviations and absolute errors.
- Generate a plot showing parameter recovery as noise increases.

## Stage 4 — Machine-learning comparison (baseline implementation)

- Fit a data-only neural trajectory model.
- Fit a physics-informed neural trajectory model using automatic differentiation and the ODE residual.
- Compare held-out displacement RMSE and recovered physical coefficients.

## Next scientific extensions

Vary observation duration and sampling density; perform a systematic sweep of the physics-loss weight; test sensitivity to initial parameter guesses; compare against measured data if a suitable dataset becomes available.

## Evaluation and reporting

Report parameter relative error, held-out displacement RMSE, robustness to measurement noise and reproducibility across random seeds. Record assumptions, parameter values, software versions and evaluation splits. This case study uses synthetic data only; it is not experimental evidence. In free decay, displacement alone identifies c/m and k/m, so the benchmark fixes mass.
