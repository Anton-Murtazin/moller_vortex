# moller-vortex

Minimal numerical implementation of Lorentz-covariant vortex states and the
paraxial ultrarelativistic Moller S-matrix from
`QED_with_vortex_packets-8.pdf`.

Natural units are used: `hbar = c = 1`. Energies, masses, momenta, and packet
widths are in MeV. Impact parameters are in `MeV^-1`.

The package contains only the calculation core:

```text
src/moller_vortex/
  config.py       global dtype, parameters, and quadrature axes
  states.py       vortex state and two-dimensional normalization
  scattering.py   longitudinal factor, transverse integral, and S-matrix
  probability.py  differential/total probabilities and mean momentum
```

Install it with:

```powershell
python -m pip install -e .
```

All finite non-periodic axes use composite Boole quadrature by default:

```python
axis = Axis(start, stop, points)
```

Boole requires `points = 4*m + 1`. Use `rule="periodic"` for angular axes.
The integration interval and number of points are always explicit.

The public API is exported from `moller_vortex`:

- `VortexPacket`, `Axis`, `NormalizationGrid`, `ScatteringGrid`,
  `ProbabilityGrid`;
- `wave_packet`, `normalization`;
- `longitudinal_factor`, `transverse_integral`, `s_matrix`;
- `differential_probability`, `differential_probability_grid`;
- `total_probability`, `mean_total_momentum`.

The global numerical type is set by `REAL_DTYPE` in `config.py`. The complex
type is derived from it automatically.
