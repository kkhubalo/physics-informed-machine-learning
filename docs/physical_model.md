# Physical model: damped harmonic oscillator

## Governing equation

For a mass attached to a linear spring and subject to viscous damping and an external force,

\[
m\ddot{x}(t)+c\dot{x}(t)+kx(t)=F(t).
\]

Here:

| Symbol | Meaning | SI unit |
|---|---|---|
| \(m\) | Mass | kg |
| \(c\) | Viscous damping coefficient | kg s\(^{-1}\) |
| \(k\) | Spring stiffness | kg s\(^{-2}\) |
| \(x\) | Displacement | m |
| \(F\) | Applied force | N |

The first implementation uses an unforced system, \(F(t)=0\), with state vector

\[
\mathbf{y}(t)=
\begin{bmatrix}x(t)\\v(t)\end{bmatrix}.
\]

The first-order system is

\[
\frac{d}{dt}
\begin{bmatrix}x\\v\end{bmatrix}
=
\begin{bmatrix}
v\\
-(c/m)v-(k/m)x
\end{bmatrix}.
\]

## Assumptions

- The spring obeys Hooke's law over the simulated displacement range.
- Damping is linear and proportional to velocity.
- The mass, damping and stiffness are constant.
- The initial conditions are known in the initial benchmark.
- Measurement noise is additive, independent, zero-mean Gaussian noise.

## Benchmark parameters

The synthetic-data script uses \(m=1.0\,\mathrm{kg}\), \(c=0.2\,\mathrm{kg/s}\), and \(k=4.0\,\mathrm{kg/s^2}\). These are illustrative values chosen for a stable underdamped response, not measurements from a physical experiment.

The natural angular frequency is \(\omega_0=\sqrt{k/m}\), and the damping ratio is \(\zeta=c/(2\sqrt{mk})\). For the benchmark, \(\omega_0=2\,\mathrm{rad/s}\) and \(\zeta=0.05\).
