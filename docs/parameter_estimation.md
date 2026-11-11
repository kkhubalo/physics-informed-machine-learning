# Classical parameter estimation

## Identifiability

For unforced motion, the governing equation can be divided by mass:

\[\ddot{x}+(c/m)\dot{x}+(k/m)x=0.\]

A displacement trajectory identifies the ratios c/m and k/m, not all three absolute values m, c and k independently. This project fixes mass to a known value and estimates damping c and stiffness k. An independent mass measurement or a known external force is needed to identify absolute coefficients without fixing mass.

## Method

For each trial pair (c, k), the ODE is integrated using the benchmark initial conditions and observation times. The residual is the difference between simulated and observed displacement. Positive coefficients are enforced by optimising their logarithms with nonlinear least squares.

Initial displacement and velocity are treated as known in this first benchmark. The tests check recovery from noise-free and noisy synthetic observations.

## Limitation

This is a controlled synthetic-data experiment. Recovery validates the implementation under the assumed model; it is not evidence of performance on measured physical data.
