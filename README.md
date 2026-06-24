# moller-vortex

Self-contained research code for numerical analysis of Moller scattering of on-axis vortex wave packets.

The project is written in natural units:

$$
\hbar = c = 1.
$$

Energies, masses and momenta are measured in MeV. Impact parameters are measured in

$$
\mathrm{MeV}^{-1}.
$$

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

The notebook imports:

```python
import moller_vortex as mv
```

so project functions are used as `mv.function_name(...)`. This keeps the
notebook namespace explicit and easier to inspect.

## Minimal workflow

```python
import numpy as np
import moller_vortex as mv

packet1 = mv.LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
packet2 = mv.LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)

N1 = mv.normalization_constant(packet1)
N2 = mv.normalization_constant(packet2)

k3 = mv.vec3([0.8, 0.10, 19.7])
k4 = mv.vec3([-0.55, -0.08, -19.6])
impact_b = mv.vec2([0.3, 0.0])

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

To use the paraxial but non-ultrarelativistic t-channel matrix element, use
the exact-time S-matrix with the full Minkowski denominator:

```python
exact_quad = mv.ExactTimeQuadrature(
    n_theta=128,
    n_kappa=257,
    theta_method="trapezoid",
)

S = mv.S_exact_time(
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
    quadrature=exact_quad,
    denominator_mode="minkowski",
    matrix_element="paraxial_massive",
)
```

For probability routines, use `s_matrix="exact_time"` and pass the same options
through `s_matrix_kwargs`. The default `matrix_element="ultrarelativistic"` is
kept for backward compatibility. In this mode the smooth energy prefactors are
evaluated at the incoming packet central energies and external final energies;
the invariant denominator is evaluated on the exact-time roots.

## Central numerical settings

The standard normalization call uses the built-in adaptive `quad` settings:

```python
N = mv.normalization_constant(packet)
```

The S-matrix and probability routines do not accept a shared tolerance object.
For scans, compute `N1` and `N2` explicitly once and pass them into the
S-matrix/probability functions. If `N1` or `N2` is omitted, the package computes
the missing normalization with the standard settings. If one particular
normalization check needs different adaptive tolerances, pass `quad_epsabs`,
`quad_epsrel`, or `quad_limit` directly in that `normalization_constant(...)`
call.

Use `ProbabilityQuadrature` / `ExactTimeQuadrature` for deterministic node
counts and integration ranges.

The quadrature dataclasses only store settings. Nodes and weights are built by:

```python
mv.probability_inner_nodes(quadrature)
mv.probability_outer_nodes(quadrature)
mv.exact_time_nodes(exact_quad, radial_interval=(kappa_min, kappa_max))  # kappa
```

`S_exact_time` uses only the direct kappa disk formula. `ExactTimeQuadrature`
stores `n_kappa/kappa_method`, `theta_method`, and the optional
`kappa_n_sigma` cutoff. The default `theta_method="analytic"` uses the closed
theta integral for `denominator_mode="expanded"`, so `n_theta` is ignored and
only the kappa quadrature remains. Use `theta_method="trapezoid"` when you want
the direct numerical angular check, when using `denominator_mode="exact"`, or
when using `matrix_element="paraxial_massive"` with
`denominator_mode="minkowski"`.

For probability integrals the finite-axis defaults are composite Boole, so use node counts of the form `n = 4*m + 1`, for example `9`, `17`, or `25`. Periodic angular axes use endpoint-free trapezoid quadrature by default.

Long grid-style probability routines accept `progress=True` to update an
in-place point-by-point progress line with completed points, total points,
percent, elapsed time and ETA. For expensive maps and outer probability
integrals, pass `workers=N` to evaluate independent outer grid points in
parallel. This is most useful for `s_matrix="exact_time"`; start with
`workers=2` or `workers=4` and tune together with
`s_matrix_kwargs={"batch_size": ...}` because both settings increase the amount
of work kept in memory at once.

## Built-in numerical checks

The project does not use a separate pytest-style test folder. Numerical checks
are ordinary functions inside the package and can be called directly from a
notebook or script. They do not decide whether a test has "passed" or "failed";
they only print and return the numerical errors.

```python
results = mv.run_all_checks(
    verbose=True,
)
```

Individual checks are also available:

```python
mv.check_normalization()
mv.check_laguerre_derivative()
mv.check_vectorized_paths()
```

The returned dictionaries contain raw relative errors. The interpretation of those errors is left to the analysis notebook. The checks compare:

1. normalization in the spherical limit

$$
\sigma_\perp = \sigma_\parallel
$$

against the closed Bessel-K expression;
2. the Laguerre-derivative formula against the direct finite sum;
3. vectorized S-matrix/probability paths against scalar diagnostic paths;
4. analytic theta exact-time integration against direct numerical theta quadrature.

## Structure

```text
src/moller_vortex/
  constants.py     units, dtypes, electron mass and charge
  kinematics.py    vector checks, energies, helicity labels
  packets.py       LGPacket and normalization constants
  amplitudes.py    impulse Moller amplitude
  transverse.py    closed transverse integrals
  smatrix.py       closed, first-order and exact-time S-matrix routines
  checks.py        ordinary check functions for notebooks and scripts
  probability.py   numerical integration of the squared modulus of S matrix related to the transverse total momentum
  quadrature.py    deterministic quadrature rules and central quadrature dataclasses

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
