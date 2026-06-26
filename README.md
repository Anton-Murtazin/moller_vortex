# moller-vortex

Research code for Moller scattering of on-axis vortex wave packets.

The project uses natural units, \( \hbar=c=1 \). Energies, masses, momenta,
and packet widths are measured in MeV. Impact parameters are measured in
MeV\(^{-1}\).

## Installation

From the project root:

```bash
python -m pip install -e .[dev]
```

Then import the package in scripts or notebooks as:

```python
import moller_vortex as mv
```

## Calculation Modes

Probability functions use one selector:

| `method` | Calculation |
|---|---|
| `"impulse"` | analytic ultrarelativistic S-impulse |
| `"first_order"` | first-order impulse correction |
| `"time"` | ultrarelativistic S_time with analytic theta by default |
| `"massive"` | massive paraxial S_time with the invariant t-channel denominator |

The massive near-forward pole is not regularized by default. Diagnose it by
scanning the integration grid and intervals.

## Minimal Example

```python
import numpy as np
import moller_vortex as mv

packet1 = mv.LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
packet2 = mv.LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)

N1 = mv.normalization_constant(packet1)
N2 = mv.normalization_constant(packet2)

k3 = np.array([0.8, 0.10, 19.7])
k4 = np.array([-0.55, -0.08, -19.6])
impact_b = np.array([0.3, 0.0])

S = mv.S_impulse(
    k3,
    k4,
    packet1,
    packet2,
    lam1=0.5,
    lam2=0.5,
    lam3=0.5,
    lam4=0.5,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
)
```

Massive paraxial time calculation:

```python
time_quad = mv.ExactTimeQuadrature(
    n_theta=128,
    n_kappa=257,
    theta_method="trapezoid",
    kappa_n_sigma=10.0,
)

S = mv.S_time(
    k3,
    k4,
    packet1,
    packet2,
    lam1=0.5,
    lam2=-0.5,
    lam3=0.5,
    lam4=-0.5,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    quadrature=time_quad,
    model="massive",
)
```

## Probability Interface

Use scalar functions for one point and grid functions for maps:

```python
prob_quad = mv.ProbabilityQuadrature(
    k3_perp_range=(0.010, 0.050),
    k3z_range=(10.0 - 50.0 * packet1.sigma_par, 10.0 + 50.0 * packet1.sigma_par),
    k4z_range=(-10.0 - 50.0 * packet2.sigma_par, -10.0 + 50.0 * packet2.sigma_par),
    n_k3_perp=25,
    n_phi=10,
    n_k3z=25,
    n_k4z=25,
)

w = mv.diff_probability(
    np.array([0.02, 0.0]),
    packet1,
    packet2,
    quadrature=prob_quad,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    method="time",
    method_kwargs={
        "quadrature": mv.ExactTimeQuadrature(),
        "batch_size": 16,
    },
)
```

Long scans accept `progress=True` for point-by-point progress and `workers=N`
for independent outer grid points. For time methods, tune `workers` together
with `method_kwargs={"batch_size": ...}` while the working arrays still fit
comfortably in memory and cache.

## Numerical Settings

Normalization uses `normalization_constant(packet)`. If a particular
normalization scan needs different adaptive settings, pass `quad_epsabs`,
`quad_epsrel`, or `quad_limit` directly to that call.

All S-matrix and probability integrations are controlled by:

```python
mv.ExactTimeQuadrature(...)
mv.ProbabilityQuadrature(...)
```

The default `ExactTimeQuadrature` uses analytic theta for
`model="ultrarelativistic"`. The massive model depends on the exact-time roots
inside the invariant denominator, so use numerical theta quadrature.

## Main Notebook

`analysis_workspace.ipynb` contains packet setup, normalization diagnostics,
S-matrix point diagnostics, quadrature scans, differential probability maps,
massive pole diagnostics, and total probability / `ky_average` examples.

## Package Layout

```text
src/moller_vortex/
  constants.py       units, dtypes, and physical constants
  kinematics.py      vector validation, energies, and helicity labels
  packets.py         LGPacket and normalization constants
  amplitudes.py      massive spinor factors
  transverse.py      analytic transverse integrals for impulse methods
  smatrix.py         S-matrix imports
  smatrix_common.py  shared S-matrix factors
  smatrix_impulse.py
  smatrix_first_order.py
  smatrix_time.py
  probability.py     vectorized phase-space probability integrals
  quadrature.py      quadrature dataclasses and node builders

analysis_workspace.ipynb
docs/FUNCTION_GUIDE.md
```
