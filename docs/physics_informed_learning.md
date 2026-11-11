# Physics-informed learning

## Learning task

A neural network x_theta(t) approximates oscillator displacement from time. The data-only model minimises mean-squared error on training observations:

\[L_data = (1/N) sum_i (x_theta(t_i)-x_i)^2.\]

The physics-informed model adds the governing-equation residual:

\[r_theta(t)=d^2x_theta/dt^2+(c/m)dx_theta/dt+(k/m)x_theta.\]

\[L=L_data+lambda L_physics,\quad L_physics=(1/N_r)sum_j r_theta(t_j)^2.\]

Derivatives are computed by automatic differentiation. Damping and stiffness are constrained to be positive through logarithmic parameterisation. Mass is fixed because free-decay displacement data do not identify all three absolute coefficients independently.

## Comparison protocol

Both neural models use the same observations, deterministic 80/20 index split and random seed. The data-only model uses observation loss alone; the PINN adds the ODE residual at collocation points. Compare held-out displacement RMSE and the PINN's recovered physical coefficients.

These compact MLPs are baseline models. Results depend on architecture, optimiser, training duration, loss weighting and random seed. Do not claim PINNs universally outperform data-only models; report the measured outcome.
