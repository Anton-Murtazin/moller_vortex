# moller-vortex

Self-contained research code for numerical analysis of Møller scattering of on-axis vortex wave packets in impulse approximation.

The project is written in natural units:

$$
\hbar = c = 1.
$$

Energies, masses and momenta are measured in MeV. Impact parameters are measured in MeV$^{-1}$.

## Installation for local work

From the project root:

```bash
python -m pip install -e .[dev]
```

The `-e` flag installs the project in editable mode: after editing files in `src/moller_vortex/`, the changes are visible without reinstalling.
After installation, notebooks and scripts should import the project directly:

```python
import moller_vortex as mv
```


## Main working file

For calculations, scans and plots, open:

```text
analysis_workspace.ipynb
```

The notebook imports both:

```python
import moller_vortex as mv
from moller_vortex import *
```

so project functions can be used either as `mv.function_name(...)` or directly as `function_name(...)`.

## Minimal workflow

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

S = mv.S_impulse_closed_form(
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

For a one-off calculation, `N1` and `N2` may be omitted; the S-matrix functions will compute them internally. For scans, compute them explicitly and reuse them.

## Built-in numerical checks

The project does not use a separate pytest-style test folder. Numerical checks are ordinary functions inside the package and can be called directly from a notebook or script. They do not decide whether a test has “passed” or “failed”; they only print and return the numerical errors.

```python
accuracy = mv.NumericalAccuracy(
    quad_epsabs=1.0e-10,
    quad_epsrel=1.0e-10,
    quad_limit=300,
    root_residual_atol=1.0e-10,
)

results = mv.run_all_checks(
    accuracy=accuracy,
    n_phi=16,
    verbose=True,
)
```

Individual checks are also available:

```python
mv.check_normalization(accuracy=accuracy)
mv.check_transverse_integral(accuracy=accuracy, n_phi=16)
mv.check_smatrix(accuracy=accuracy, n_phi=16)
```

The returned dictionaries contain raw relative errors. The interpretation of those errors is left to the analysis notebook. The checks compare:

1. normalization in the spherical limit $\sigma_\perp = \sigma_\parallel$ against the closed Bessel-$K$ expression;
2. analytic transverse expressions against direct numerical polar integration;
3. closed impulse $S$ matrix against the same formula with numerical transverse integration.

## Structure

```text
src/moller_vortex/
  constants.py     units, dtypes, electron mass and charge
  accuracy.py      global numerical integration accuracy
  kinematics.py    vector checks, energies, helicity labels
  packets.py       LGPacket and normalization constants
  amplitudes.py    impulse Møller amplitude
  transverse.py    analytic and direct numerical transverse integrals
  smatrix.py       closed and numerically checked impulse S matrix
  checks.py        ordinary check functions for notebooks and scripts
  probability.py   numerical integration of the squared module of S matrix related to the transverse total momentum

notebooks/
  usage_example.ipynb

analysis_workspace.ipynb  main Jupyter workspace for exploratory work
```

## Main documentation

The central reference is:

```text
docs/FUNCTION_GUIDE.md
```

It describes the physical role of the main functions, their implementation, expected inputs and outputs, typical usage patterns for scans, and the built-in check functions.

## Input and output conventions

- transverse vectors are array-like objects with shape `(2,)`;
- three-momenta are array-like objects with shape `(3,)`;
- helicities must be exactly `+0.5` or `-0.5`;
- `LGPacket` stores only physical parameters and does not store a normalization constant;
- compute `N = normalization_constant(packet)` explicitly and pass it to repeated scans;
- the impact parameter belongs to the second incoming packet and is passed to S-matrix functions as `impact_b`.
