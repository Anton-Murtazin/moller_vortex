# moller-vortex

Compact numerical code for the Lorentz-covariant vortex packets and the
paraxial ultrarelativistic impulse S-matrix in
`QED_with_vortex_packets-2.pdf`.

Natural units are used: `hbar = c = 1`. Masses, energies, momenta, and packet
widths are in MeV; impact parameters are in `MeV^-1`.

## Scope

The implementation follows only the formulas present in the PDF:

- vortex state: Eqs. (4), (14)-(16);
- two-dimensional normalization: Eqs. (17)-(18);
- longitudinal impulse factor: Eq. (28);
- first-order transverse propagator expansion and closed integrals:
  Eqs. (32), (46), and (56);
- S-matrix: Eq. (31);
- approximation diagnostics: Eqs. (57)-(101).

There are no exact-time, massive-spinor, u-channel, higher-order, or
regularized-pole modes. In particular, the code uses the closed transverse
formulas derived after Eq. (32), so it does not introduce an unphysical cutoff
into Eq. (29).

The probability functions retain the phase-space definitions used in the old
project:

```text
dP/d^2K_perp = integral d^3k3/[(2pi)^3 2E3]
                        d^3k4/[(2pi)^3 2E4]
                        delta^2(K_perp-k3_perp-k4_perp) |S_fi|^2
```

The reported total probability is the integral over the explicitly selected
finite domain. Convergence with respect to every interval and point count must
be checked before interpreting it as the full-domain result.

## Structure

```text
src/moller_vortex/
  config.py       global dtype, constants, packets, quadrature axes
  states.py       state, 2D normalization, approximation diagnostics
  scattering.py   longitudinal/transverse factors and S-matrix
  probability.py  differential/total probability and mean total momentum
moller_vortex.ipynb
```

## Installation

```powershell
python -m pip install -e ".[notebook,dev]"
```

The calculation core itself depends only on NumPy.

## Minimal use

```python
import numpy as np
import moller_vortex as mv

packet1 = mv.VortexPacket(1, 2.0e-4, 0.05, 0.05, 10.0)
packet2 = mv.VortexPacket(-1, 2.0e-4, 0.05, 0.05, -10.0)

normalization_grid = mv.NormalizationGrid(
    k_perp=mv.Axis(0.0, 0.002, 64),
    k_z=mv.Axis(9.5, 10.5, 96),
)
N1 = mv.normalization(packet1, normalization_grid)

grid2 = mv.NormalizationGrid(
    k_perp=normalization_grid.k_perp,
    k_z=mv.Axis(-10.5, -9.5, 96),
)
N2 = mv.normalization(packet2, grid2)

k3 = np.array((0.02, 0.0, 10.0))
k4 = np.array((-0.0198, 0.0, -10.0))
S, info = mv.s_matrix(
    k3, k4, packet1, packet2, norms=(N1, N2), return_info=True
)
```

Every finite integration axis has an explicit interval, point count, and rule.
Use `rule="gauss"` for finite non-periodic axes and `rule="periodic"` for
azimuthal angles. Probability map points and outer total-momentum points can be
parallelized with `workers=N`; inner integrals are vectorized and processed in
memory-controlled batches.
