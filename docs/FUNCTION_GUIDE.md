# Function Guide

Use:

```python
import moller_vortex as mv
```

The code uses natural units, \( \hbar=c=1 \). Momenta, masses, energies, and
packet widths are in MeV. Impact parameters are in MeV\(^{-1}\).

## Packets

```python
packet = mv.LGPacket(
    ell=5,
    sigma_perp=sigma_perp,
    sigma_par=sigma_par,
    kbar_z=kbar_z,
)

N = mv.normalization_constant(packet)
```

For scans, compute `N1` and `N2` once and pass them to S-matrix and probability
functions.

Useful helpers:

| Function | Purpose |
|---|---|
| `spatial_width_nm_to_momentum_mev(width_nm)` | convert a spatial width to a momentum width |
| `central_energy(packet, m=...)` | central packet energy |
| `normalization_constant(packet, ...)` | on-axis packet normalization |
| `spherical_normalization_constant(packet, ...)` | spherical-width reference |

If a normalization scan needs non-default adaptive settings, pass `quad_epsabs`,
`quad_epsrel`, or `quad_limit` directly to `normalization_constant(...)`.

## Quadratures

```python
prob_quad = mv.ProbabilityQuadrature(
    k3_perp_range=(0.010, 0.050),
    k3z_range=(10.0 - 50.0 * sigma1_par, 10.0 + 50.0 * sigma1_par),
    k4z_range=(-10.0 - 50.0 * sigma2_par, -10.0 + 50.0 * sigma2_par),
    n_k3_perp=25,
    n_phi=10,
    n_k3z=25,
    n_k4z=25,
)

time_quad = mv.ExactTimeQuadrature(
    n_theta=128,
    n_kappa=257,
    theta_method="analytic",
    kappa_method="boole",
    kappa_n_sigma=10.0,
)
```

Use `theta_method="analytic"` for the ultrarelativistic time calculation. Use
`theta_method="trapezoid"` for the massive time calculation.

## S-Matrix Functions

### Impulse

```python
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

Grid version:

```python
S = mv.S_impulse_grid(k3x, k3y, k3z, k4x, k4y, k4z, ...)
```

### First Order

```python
S = mv.S_first_order(
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

Optional `time_step` or `time_step_scale` controls the finite difference used
for the first-order correction.

Grid version:

```python
S = mv.S_first_order_grid(k3x, k3y, k3z, k4x, k4y, k4z, ...)
```

### Time

```python
S = mv.S_time(
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
    quadrature=time_quad,
    model="ultrarelativistic",
)
```

`model="ultrarelativistic"` uses the ultrarelativistic transverse denominator
and supports analytic theta. `model="massive"` uses the massive paraxial spinor
factor with the invariant t-channel denominator and requires numerical theta.

Grid version:

```python
S = mv.S_time_grid(k3x, k3y, k3z, k4x, k4y, k4z, ..., model="massive")
```

Use `batch_size` to control how many broadcast points are evaluated at once in
`S_time_grid`.

## Probability Functions

Probability functions use:

```python
method="impulse"       # mv.S_impulse
method="first_order"   # mv.S_first_order
method="time"          # mv.S_time, model="ultrarelativistic"
method="massive"       # mv.S_time, model="massive"
```

One \(K_\perp\) point:

```python
w = mv.diff_probability(
    K_perp,
    packet1,
    packet2,
    quadrature=prob_quad,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    method="time",
    method_kwargs={
        "quadrature": time_quad,
        "batch_size": 16,
    },
)
```

Map:

```python
values = mv.diff_probability_grid(
    Kx_values,
    Ky_values,
    packet1,
    packet2,
    quadrature=prob_quad,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    method="massive",
    method_kwargs={
        "quadrature": mv.ExactTimeQuadrature(theta_method="trapezoid"),
        "batch_size": 16,
    },
    progress=True,
    workers=8,
)
```

Longitudinal maps, total probability, and transverse-momentum averages:

```python
longitudinal = mv.longitudinal_density_grid(..., method="time")
P = mv.total_probability(..., method="impulse")
Ky = mv.ky_average(..., method="first_order")
```

`progress=True` reports completed points, total points, percent, elapsed time,
and ETA. `workers` parallelizes independent outer points.

## Diagnostics

Before interpreting a result, scan the deterministic settings that control the
integral being used:

| Integral | Parameters to scan |
|---|---|
| packet normalization | `quad_limit`, `quad_epsabs`, `quad_epsrel` |
| phase-space probability | `n_k3_perp`, `n_phi`, `n_k3z`, `n_k4z` |
| phase-space intervals | `k3_perp_range`, `k3z_range`, `k4z_range` |
| ultrarelativistic time | `n_kappa`, `kappa_n_sigma` |
| massive time | `n_kappa`, `n_theta`, `kappa_n_sigma`, near-forward `k3_perp_range` |

The demonstration notebook contains compact scan blocks for these quantities.
