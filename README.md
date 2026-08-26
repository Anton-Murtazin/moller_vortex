# moller-vortex

Compact NumPy implementation of the vortex packets defined in the notebook
and a paraxial ultrarelativistic impulse model for Moller scattering.

Natural units are used: `hbar = c = 1`. Masses, energies, momenta, and packet
widths are in MeV; impact parameters are in `MeV^-1`.

## Scope

The implementation contains:

- the user-specified Gaussian vortex state and its relativistic normalization;
- the forward, helicity-conserving ultrarelativistic Moller amplitude;
- the separated longitudinal impulse factor;
- the closed first-order transverse propagator convolution;
- analytic integration over the two final longitudinal momenta;
- differential probability and the mean total momentum.

The packet ansatz used here has separate negative Gaussian factors in energy
and longitudinal momentum. It is therefore the model specified in the
notebook, rather than a literal transcription of every packet formula in the
reference PDF.

The probability functions use

```text
dP/d^2K_perp = integral d^3k3/[(2pi)^3 2E3]
                        d^3k4/[(2pi)^3 2E4]
                        delta^2(K_perp-k3_perp-k4_perp) |S_fi|^2.
```

With momenta expressed in MeV, `dP/d^2K_perp` is returned in `MeV^-2`.

## Structure

```text
src/moller_vortex/
  config.py       constants and calculation parameter dataclasses
  numerics.py     quadrature, vector operations, and progress helpers
  states.py       vortex state and two-dimensional normalization
  scattering.py   longitudinal/transverse factors and S-matrix
  probability.py  differential probability and mean total momentum
moller_vortex.ipynb
```

## Installation

```powershell
python -m pip install -e ".[notebook]"
```

The calculation core depends only on NumPy.

## Quadrature axes

Every finite integration domain is represented by
`Axis(start, stop, points, rule)`. The available rules are `"boole"`,
`"gauss"`, `"trapezoid"`, and `"periodic"`. The default is composite
five-point Boole, for which `points` must be `4 * n + 1`. The `"periodic"`
option is the endpoint-free trapezoidal rule used for azimuthal angles.

## Minimal use

```python
import numpy as np
import moller_vortex as mv

normalization_grid_1 = mv.NormalizationGrid(
    k_perp=mv.Axis(0.0, 0.002, 65),
    k_z=mv.Axis(9.5, 10.5, 97),
)
normalization_grid_2 = mv.NormalizationGrid(
    k_perp=normalization_grid_1.k_perp,
    k_z=mv.Axis(-10.5, -9.5, 97),
)
packet1 = mv.VortexPacket(
    ell=1,
    sigma_perp=2.0e-4,
    sigma_parallel=0.05,
    sigma_energy=0.05,
    k0_z=10.0,
    normalization_grid=normalization_grid_1,
)
packet2 = mv.VortexPacket(
    ell=-1,
    sigma_perp=2.0e-4,
    sigma_parallel=0.05,
    sigma_energy=0.05,
    k0_z=-10.0,
    normalization_grid=normalization_grid_2,
)

momentum_3 = np.array((0.02, 0.0, 10.0))
momentum_4 = np.array((-0.0198, 0.0, -10.0))
S = mv.s_matrix(momentum_3, momentum_4, packet1, packet2)
```

Probability maps and outer momentum integrals are evaluated sequentially;
their remaining transverse integration arrays are vectorized. The two final
longitudinal momenta are integrated analytically in the same paraxial
approximation used to derive the impulse S-matrix. Pass `progress=True` to
`differential_probability_grid`, `total_probability`, or
`mean_total_momentum` to display progress in a notebook or terminal.
