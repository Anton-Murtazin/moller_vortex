# Function Guide

This file is a practical guide to the `moller_vortex` project. It describes the main physical objects, numerical routines, units, and the intended workflow for using the code.

The project is written as an editable Python package. In normal use, work from the project root and install it with

```powershell
python -m pip install -e ".[dev]"
```

Then import the package in notebooks or scripts as

```python
import moller_vortex as mv
```

Avoid `from moller_vortex import *` in notebooks. It can hide name conflicts and makes navigation in VS Code worse.

---

## 1. Units and conventions

The code uses relativistic natural units with

\[
c=1.
\]

The numerical momentum and energy unit is MeV.

| Quantity | Code unit |
|---|---:|
| momentum \(k\), \(K\), \(\sigma\), \(m\), \(E\) | MeV |
| coordinate-space impact parameter \(b\) | MeV\(^{-1}\) |
| differential probability \(w(\mathbf K_\perp)=dP/d^2K_\perp\) | MeV\(^{-2}\) |
| total probability \(P\) | dimensionless |
| \(\langle K_x\rangle,\langle K_y\rangle\) | MeV |

If a paper gives transverse or longitudinal coordinate widths in nm, convert the corresponding momentum scale as

\[
\sigma_p c = \frac{\hbar c}{\sigma_x}.
\]

In code units this is written as

```python
HBARC_MEV_NM = 1.973269804e-4
sigma_p_mev = HBARC_MEV_NM / sigma_x_nm
```

The impact parameter converts as

\[
b[\mathrm{MeV}^{-1}] = \frac{b[\mathrm{nm}]}{\hbar c[\mathrm{MeV\,nm}]}.
\]

In code:

```python
NM_TO_MEV_INV = 1.0 / HBARC_MEV_NM
impact_b = np.array([b_x_nm * NM_TO_MEV_INV, b_y_nm * NM_TO_MEV_INV])
```

The sign convention for the second packet follows the phase

\[
\phi_2(\mathbf k_2)\propto \exp\{+i\mathbf b_\perp\cdot\mathbf k_{2\perp}\}.
\]

Since

\[
\mathbf k_{2\perp}=\mathbf K_\perp-\mathbf k_{1\perp},
\]

this gives

\[
e^{+i\mathbf b_\perp\cdot\mathbf k_{2\perp}}
=
e^{+i\mathbf b_\perp\cdot\mathbf K_\perp}
e^{-i\mathbf b_\perp\cdot\mathbf k_{1\perp}}.
\]

Therefore the transverse Gaussian source is

\[
\mathbf J_0=(\beta+\gamma)\mathbf K_\perp-i\mathbf b_\perp.
\]

---

## 2. Accuracy configuration

### `NumericalAccuracy`

```python
accuracy = mv.NumericalAccuracy(
    quad_epsabs=1.0e-10,
    quad_epsrel=1.0e-10,
    quad_limit=300,
    root_residual_atol=1.0e-10,
)
```

This object stores global numerical accuracy parameters.

Fields:

```python
quad_epsabs: float
quad_epsrel: float
quad_limit: int
root_residual_atol: float
```

`quad_epsabs`, `quad_epsrel`, and `quad_limit` are used by one-dimensional adaptive integrations such as normalization integrals.

`root_residual_atol` is reserved for routines involving root validation in delta-reduced expressions.

The package default is

```python
mv.ACCURACY
```

Use a separate variable name such as `acc` if you want to avoid conflict with the module name `moller_vortex.accuracy`.

---

## 3. Packet definition

### `LGPacket`

```python
packet = mv.LGPacket(
    ell=5,
    sigma_perp=sigma_perp,
    sigma_par=sigma_par,
    kbar_z=10.0,
)
```

Represents an on-axis Laguerre-Gaussian-type momentum-space packet.

Fields:

```python
ell: int
sigma_perp: float
sigma_par: float
kbar_z: float
```

Physical meaning:

- `ell` is the OAM integer.
- `sigma_perp` is the transverse momentum width in MeV.
- `sigma_par` is the longitudinal momentum width in MeV.
- `kbar_z` is the central longitudinal momentum in MeV.

The vortex phase convention is

\[
\ell>0:\quad k_\perp^{|\ell|}e^{+i\ell\phi}=k_+^\ell,
\]

\[
\ell<0:\quad k_\perp^{|\ell|}e^{i\ell\phi}=k_-^{|\ell|}.
\]

Here

\[
k_+=k_x+ik_y,
\qquad
k_-=k_x-ik_y.
\]

### `lg_packet_phi(...)`

Returns the momentum-space packet wave function factor. This is useful for direct checks or for building diagnostic plots of the packet in momentum space.

### `vortex_factor(...)`

Returns the OAM phase and radial factor associated with a packet.

---

## 4. Normalization

### `normalization_constant(packet, m=..., accuracy=...)`

Computes the relativistic normalization constant \(N_\ell\) for an on-axis packet.

Typical use:

```python
N1 = mv.normalization_constant(packet1, accuracy=acc)
N2 = mv.normalization_constant(packet2, accuracy=acc)
```

For scans, compute `N1`, `N2` once and pass them explicitly to avoid recomputing normalization many times.

For narrow packets, the implementation must avoid computing separately the unstable product

\[
\exp\left(\frac{2m^2}{\sigma_\parallel^2}\right)
K_0\left(
\frac{2m\sqrt{m^2+k_\perp^2}}{\sigma_\parallel^2}
\right).
\]

A stable implementation uses

\[
K_\nu(x)=e^{-x}\operatorname{kve}(\nu,x),
\]

so that the exponential cancellation is performed analytically.

The stable product is

\[
\exp\left[
-\frac{2m}{\sigma_\parallel^2}
\left(
\sqrt{m^2+k_\perp^2}-m
\right)
\right]
\operatorname{kve}
\left(
0,
\frac{2m\sqrt{m^2+k_\perp^2}}{\sigma_\parallel^2}
\right).
\]

### `spherical_normalization_constant(packet, m=...)`

Closed expression for the normalization in the spherical limit

\[
\sigma_\perp=\sigma_\parallel.
\]

Use this only for checks or for exactly spherical packets.

---

## 5. Basic kinematics

### `vec2(x)`, `vec3(x)`

Convert input objects to 2D or 3D NumPy arrays. These functions are mostly convenience utilities.

### `energy(k, m=...)`

Returns

\[
E_{\mathbf k}=\sqrt{m^2+\mathbf k^2}.
\]

Input `k` is a 3-vector in MeV.

### `central_energy(packet, m=...)`

Returns

\[
\bar E = \sqrt{m^2+\bar k_z^2}
\]

for the central momentum of a packet.

### `helicity(...)`

Used where spin/helicity labels are needed. In the current impulse approximation, the implemented spin dependence reduces to helicity Kronecker deltas.

### `kron_delta(a, b)`

Kronecker delta helper.

---

## 6. Moller amplitude

### `moller_amplitude_impulse(...)`

Implements the impulse-approximation spinor amplitude used inside the S-matrix routines.

In the current impulse approximation the spin structure used in the probability module is effectively

\[
S_{\lambda_1\lambda_2\lambda_3\lambda_4}
=
S_0
\delta_{\lambda_3\lambda_1}
\delta_{\lambda_4\lambda_2}.
\]

Therefore the unpolarized spin average satisfies

\[
\frac{1}{4}
\sum_{\lambda_1,\lambda_2}
\sum_{\lambda_3,\lambda_4}
|S_{\lambda_1\lambda_2\lambda_3\lambda_4}|^2
=
|S_0|^2.
\]

---

## 7. Transverse integral

The transverse integral is one of the central analytic pieces of the project.

The basic definitions are

\[
A=\alpha+\gamma,
\]

\[
\mathbf J_0=(\beta+\gamma)\mathbf K_\perp-i\mathbf b_\perp,
\]

\[
p_+=\frac{J_{0x}+iJ_{0y}}{A},
\qquad
p_-=\frac{J_{0x}-iJ_{0y}}{A},
\]

\[
K_+=K_x+iK_y,
\qquad
K_-=K_x-iK_y,
\]

\[
q_+=K_+-p_+,
\qquad
q_-=K_--p_-.
\]

The common transverse prefactor is

\[
\mathcal P
=
\frac{1}{k_{3\perp}^2}
\frac{2\pi}{A}
\exp\left(
\frac{\mathbf J_0^2}{2A}
-
\frac{\gamma K_\perp^2}{2}
\right).
\]

Here

\[
\mathbf J_0^2=J_{0x}^2+J_{0y}^2,
\]

which is a bilinear square, not a Hermitian norm.

### `laguerre_derivative(a, b, c1, c2, c12)`

Computes

\[
D_{a,b}(c_1,c_2;c_{12})
=
\left.
\partial_{t_1}^{a}
\partial_{t_2}^{b}
\exp(c_1t_1+c_2t_2-c_{12}t_1t_2)
\right|_{t_1=t_2=0}.
\]

For \(b\ge a\),

\[
D_{a,b}
=
(-c_{12})^a a!\,
c_2^{b-a}
L_a^{b-a}\left(\frac{c_1c_2}{c_{12}}\right).
\]

For \(a>b\),

\[
D_{a,b}
=
(-c_{12})^b b!\,
c_1^{a-b}
L_b^{a-b}\left(\frac{c_1c_2}{c_{12}}\right).
\]

This form is used for opposite-sign OAM cases.

### `laguerre_derivative_sum(...)`

Direct finite-sum implementation of the same derivative. It is mainly a diagnostic check for `laguerre_derivative(...)`.

### `transverse_integral_explicit(...)`

Computes the analytic first-order expanded transverse integral for arbitrary integer \(\ell_1,\ell_2\).

It covers all cases:

\[
(0,0),
\qquad
(0,\pm m),
\qquad
(\pm n,0),
\qquad
(n,m),
\qquad
(-n,-m),
\qquad
(n,-m),
\qquad
(-n,m).
\]

The same-sign cases use direct powers of \(p_\pm,q_\pm\).

The opposite-sign cases use

\[
D^{(+,-)}_{a,b}
=
D_{a,b}\left(p_+,q_-;\frac{2}{A}\right),
\]

\[
D^{(-,+)}_{a,b}
=
D_{a,b}\left(p_-,q_+;\frac{2}{A}\right).
\]

### `transverse_integral_numeric_quad(...)`

Numerical check for the same first-order transverse integral.

This is intended for verification, not for production scans.

---

## 8. S-matrix routines

### `impulse_parameters(...)`

Computes the intermediate parameters entering the impulse-approximation expression.

These include kinematic quantities, transverse Gaussian coefficients, and factors entering the reduced S-matrix.

### `S_impulse_common_factor(...)`

Computes the common factor outside the transverse integral in the impulse approximation.

Important: the transverse factor

\[
\frac{1}{k_{3\perp}^2}\frac{2\pi}{A_0}
\exp\left(
\frac{\mathbf J_{00}^2}{2A_0}
-
\frac{\gamma_0 K_\perp^2}{2}
\right)
\]

belongs to `transverse_integral_explicit(...)` and must not be duplicated in the common factor.

### `S_impulse_closed_form(...)`

Main analytic impulse-approximation S-matrix routine.

Typical call:

```python
S = mv.S_impulse_closed_form(
    k3,
    k4,
    packet1,
    packet2,
    lam1=-0.5,
    lam2=0.5,
    lam3=-0.5,
    lam4=0.5,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)
```

Inputs:

- `k3`, `k4`: final 3-momenta in MeV.
- `packet1`, `packet2`: incoming wave packets.
- `lam1`, `lam2`, `lam3`, `lam4`: helicities.
- `impact_b`: 2-vector in MeV\(^{-1}\).
- `N1`, `N2`: normalization constants. If omitted, they may be recomputed.
- `accuracy`: numerical accuracy object.

### `S_impulse_numeric_transverse_quad(...)`

Same impulse S-matrix expression, but with the transverse integral computed numerically. This is a diagnostic routine used to validate the analytic transverse expression.

---

## 9. Differential probability

The probability module evaluates

\[
w(\mathbf K_\perp)
=
\frac{dP}{d^2K_\perp}
\]

at fixed total final transverse momentum

\[
\mathbf K_\perp
=
\mathbf k_{3\perp}+\mathbf k_{4\perp}.
\]

The implemented expression is

\[
w(\mathbf K_\perp)
=
\int
\frac{d^2 k_{3\perp}\,dk_{3z}\,dk_{4z}}
{(2\pi)^6\,4E_3E_4}
\,
\frac{1}{4}
\sum_{\lambda_1,\lambda_2}
\sum_{\lambda_3,\lambda_4}
|S_{fi}|^2.
\]

The constraint of fixed \(\mathbf K_\perp\) is imposed by

\[
\mathbf k_{4\perp}
=
\mathbf K_\perp-\mathbf k_{3\perp}.
\]

The transverse integration over \(\mathbf k_{3\perp}\) is performed in polar coordinates:

\[
d^2k_{3\perp}
=
\rho\,d\rho\,d\phi.
\]

The radial integrations use Gauss-Legendre quadrature on finite intervals. Angular integrations use periodic trapezoidal quadrature on \([0,2\pi)\).

### `ProbabilityQuadrature`

Stores all quadrature parameters for both the fixed-\(\mathbf K_\perp\) differential probability and the remaining outer \(\mathbf K_\perp\) integration.

Current fields:

```python
ProbabilityQuadrature(
    k3_perp_range=(...),
    k3z_range=(...),
    k4z_range=(...),
    K_perp_range=(...),
    n_k3_perp=...,
    n_phi=...,
    n_k3z=...,
    n_k4z=...,
    n_K_perp=...,
    n_K_phi=...,
)
```

Meanings:

- `k3_perp_range`: integration range for \(\rho=|\mathbf k_{3\perp}|\).
- `k3z_range`: integration range for \(k_{3z}\).
- `k4z_range`: integration range for \(k_{4z}\).
- `K_perp_range`: integration range for \(K=|\mathbf K_\perp|\) in the final outer transverse integral.
- `n_k3_perp`: Gauss-Legendre nodes for \(\rho\).
- `n_phi`: angular nodes for \(\phi\), the angle of \(\mathbf k_{3\perp}\).
- `n_k3z`: Gauss-Legendre nodes for \(k_{3z}\).
- `n_k4z`: Gauss-Legendre nodes for \(k_{4z}\).
- `n_K_perp`: Gauss-Legendre nodes for \(K=|\mathbf K_\perp|\).
- `n_K_phi`: angular nodes for \(\phi_K\), the angle of \(\mathbf K_\perp\).

Example for parameters close to the cited plots:

```python
quadrature = mv.ProbabilityQuadrature(
    k3_perp_range=(0.010, 0.050),
    k3z_range=(10.0 - 8.0 * sigma1_par, 10.0 + 8.0 * sigma1_par),
    k4z_range=(-10.0 - 8.0 * sigma2_par, -10.0 + 8.0 * sigma2_par),
    K_perp_range=(0.0, 3.0e-4),
    n_k3_perp=10,
    n_phi=24,
    n_k3z=10,
    n_k4z=10,
    n_K_perp=10,
    n_K_phi=24,
)
```

### `legendre_nodes_and_weights(interval, n)`

Returns Gauss-Legendre nodes and weights on a finite interval.

If \(x_i,w_i\) are nodes and weights on \([-1,1]\), the map to \([a,b]\) is

\[
t_i=\frac{b-a}{2}x_i+\frac{a+b}{2},
\]

\[
W_i=\frac{b-a}{2}w_i.
\]

### `spin_averaged_s_abs2_impulse(...)`

Computes

\[
\frac{1}{4}
\sum_{\lambda_1,\lambda_2}
\sum_{\lambda_3,\lambda_4}
|S_{fi}|^2
\]

in the impulse approximation.

Because the current spin dependence is only

\[
\delta_{\lambda_3\lambda_1}\delta_{\lambda_4\lambda_2},
\]

the default branch computes one helicity-conserving amplitude and returns its squared modulus.

Use

```python
explicit_spin_sum=True
```

only as a diagnostic, because it performs all 16 helicity combinations.

### `diff_probability(...)`

Computes \(w(\mathbf K_\perp)\) at one fixed value of \(\mathbf K_\perp\).

Example:

```python
K_perp = np.array([0.0, 0.0])

w = mv.diff_probability(
    K_perp,
    packet1,
    packet2,
    quadrature,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)
```

Return dimension:

\[
[w]=\mathrm{MeV}^{-2}.
\]

### `diff_probability_grid(...)`

Computes `diff_probability(...)` on a rectangular \(K_x,K_y\) grid for color plots.

Example:

```python
Kx_values = np.linspace(-3.0e-4, 3.0e-4, 31)
Ky_values = np.linspace(-3.0e-4, 3.0e-4, 31)

W = mv.diff_probability_grid(
    Kx_values,
    Ky_values,
    packet1,
    packet2,
    quadrature,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)
```

`W[iy, ix]` corresponds to

\[
K_x=Kx\_values[ix],
\qquad
K_y=Ky\_values[iy].
\]

For plotting axes in eV, multiply the `extent` by \(10^6\).

### `total_probability(...)`

Computes

\[
P=\int w(\mathbf K_\perp)d^2K_\perp.
\]

The outer integral is done in polar coordinates:

\[
\mathbf K_\perp=K(\cos\phi_K,\sin\phi_K),
\]

\[
d^2K_\perp=K\,dK\,d\phi_K.
\]

The radial \(K\) integral uses Gauss-Legendre quadrature over `quadrature.K_perp_range`.

The angular \(\phi_K\) integral uses periodic trapezoidal quadrature with `quadrature.n_K_phi` nodes.

Example:

```python
P = mv.total_probability(
    packet1,
    packet2,
    quadrature,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)
```

Return dimension: dimensionless.

### `Ky_average(...)`

Computes

\[
\langle K_y\rangle
=
\frac{
\int K_y w(\mathbf K_\perp)d^2K_\perp
}{
\int w(\mathbf K_\perp)d^2K_\perp
}.
\]

Example:

```python
Ky_mean = mv.Ky_average(
    packet1,
    packet2,
    quadrature,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)

print(Ky_mean, "MeV")
print(Ky_mean * 1.0e6, "eV")
```

Return dimension: MeV.

---

## 10. Color plot workflow

Example setup for parameters close to the paper figure:

```python
import numpy as np
import matplotlib.pyplot as plt
import moller_vortex as mv

HBARC_MEV_NM = 1.973269804e-4
NM_TO_MEV_INV = 1.0 / HBARC_MEV_NM

def spatial_width_nm_to_momentum_mev(width_nm: float) -> float:
    return HBARC_MEV_NM / width_nm

sigma1_perp = spatial_width_nm_to_momentum_mev(10.0)
sigma2_perp = spatial_width_nm_to_momentum_mev(2.0)
sigma1_par = spatial_width_nm_to_momentum_mev(5.0)
sigma2_par = spatial_width_nm_to_momentum_mev(1.0)

packet1 = mv.LGPacket(
    ell=5,
    sigma_perp=sigma1_perp,
    sigma_par=sigma1_par,
    kbar_z=10.0,
)

packet2 = mv.LGPacket(
    ell=0,
    sigma_perp=sigma2_perp,
    sigma_par=sigma2_par,
    kbar_z=-10.0,
)

impact_b = np.array([5.0 * NM_TO_MEV_INV, 0.0], dtype=float)

acc = mv.NumericalAccuracy(
    quad_epsabs=1.0e-10,
    quad_epsrel=1.0e-10,
    quad_limit=300,
    root_residual_atol=1.0e-10,
)

N1 = mv.normalization_constant(packet1, accuracy=acc)
N2 = mv.normalization_constant(packet2, accuracy=acc)

quadrature = mv.ProbabilityQuadrature(
    k3_perp_range=(0.010, 0.050),
    k3z_range=(10.0 - 8.0 * sigma1_par, 10.0 + 8.0 * sigma1_par),
    k4z_range=(-10.0 - 8.0 * sigma2_par, -10.0 + 8.0 * sigma2_par),
    K_perp_range=(0.0, 3.0e-4),
    n_k3_perp=10,
    n_phi=24,
    n_k3z=10,
    n_k4z=10,
    n_K_perp=10,
    n_K_phi=24,
)
```

Compute a rectangular colorplot grid:

```python
Kx_values = np.linspace(-3.0e-4, 3.0e-4, 31)
Ky_values = np.linspace(-3.0e-4, 3.0e-4, 31)

W = mv.diff_probability_grid(
    Kx_values,
    Ky_values,
    packet1,
    packet2,
    quadrature,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
    accuracy=acc,
)
```

Plot:

```python
fig, ax = plt.subplots(figsize=(5.5, 4.8))

image = ax.imshow(
    W,
    origin="lower",
    extent=[
        Kx_values[0] * 1.0e6,
        Kx_values[-1] * 1.0e6,
        Ky_values[0] * 1.0e6,
        Ky_values[-1] * 1.0e6,
    ],
    aspect="equal",
)

ax.set_xlabel(r"$K_x$ [eV]")
ax.set_ylabel(r"$K_y$ [eV]")
ax.set_title(r"$w(\mathbf{K}_{\perp})$")

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(r"$w(\mathbf{K}_{\perp})$ [MeV$^{-2}$]")

fig.tight_layout()
plt.show()
```

If values span many orders of magnitude, use logarithmic plotting.

---

## 11. Superkick scan workflow

To compute

\[
\langle K_y\rangle(b_x),
\]

use `Ky_average(...)` for each impact parameter.

Example:

```python
b_x_nm = np.linspace(0.0, 40.0, 41)

Ky_values = np.array([
    mv.Ky_average(
        packet1,
        packet2,
        quadrature,
        impact_b=np.array([b_nm * NM_TO_MEV_INV, 0.0], dtype=float),
        N1=N1,
        N2=N2,
        accuracy=acc,
    )
    for b_nm in b_x_nm
])

fig, ax = plt.subplots(figsize=(5.5, 4.2))

ax.plot(
    b_x_nm,
    Ky_values * 1.0e6,
    marker="o",
    linewidth=1.5,
)

ax.set_xlabel(r"$b_x$ [nm]")
ax.set_ylabel(r"$\langle K_y\rangle$ [eV]")
ax.set_title(r"$\langle K_y\rangle$ as a function of $b_x$")
ax.grid(True)

fig.tight_layout()
plt.show()
```

Runtime estimate:

If a \(31\times31\) colorplot took about 20 minutes, then a scan with

```python
len(b_x_nm) = 41
n_K_perp = 10
n_K_phi = 24
```

uses approximately

\[
41\cdot 10\cdot24=9840
\]

calls to `diff_probability(...)`.

A \(31\times31\) colorplot uses

\[
31\cdot31=961
\]

calls.

So the \(\langle K_y\rangle(b_x)\) scan can take about

\[
\frac{9840}{961}\approx 10.2
\]

times longer than the colorplot at the same internal quadrature settings.

---

## 12. Numerical strategy

The code uses deterministic quadrature.

For finite radial and longitudinal intervals:

\[
\int_a^b f(x)\,dx
\simeq
\sum_i W_i f(x_i),
\]

where \(x_i,W_i\) are Gauss-Legendre nodes and weights.

For angular variables:

\[
\int_0^{2\pi} f(\phi)\,d\phi
\simeq
\frac{2\pi}{N}
\sum_{j=0}^{N-1}
f\left(\frac{2\pi j}{N}\right).
\]

This is the periodic trapezoidal rule. All weights are equal because the endpoint \(2\pi\) is not included and \(0\equiv2\pi\).

Increase the following parameters for convergence:

```python
n_k3_perp
n_phi
n_k3z
n_k4z
n_K_perp
n_K_phi
```

A sensible workflow is:

1. Use small quadrature to verify signs and scales.
2. Increase the internal quadrature for `diff_probability`.
3. Increase the outer \(K\)-quadrature for `total_probability` and `Ky_average`.
4. Check convergence at representative points before running expensive full scans.

---

## 13. Git and editable installation

For ordinary project work, use an editable install:

```powershell
python -m pip install -e ".[dev]"
```

Then changes in `src/moller_vortex/*.py` are picked up after restarting the Python kernel.

When adding a new file, for example

```text
src/moller_vortex/probability.py
```

make sure it is exported in

```text
src/moller_vortex/__init__.py
```

For example:

```python
from .probability import *
```

or preferably explicit imports:

```python
from .probability import (
    ProbabilityQuadrature,
    diff_probability,
    diff_probability_grid,
    total_probability,
    Ky_average,
)
```

After editing `__init__.py`, restart the notebook kernel.

For Git workflow:

```powershell
git status
git add src/moller_vortex/probability.py
git commit -m "Update probability integration routines"
git push
```

Do not commit generated Python cache files:

```text
__pycache__/
*.pyc
```

They should be ignored by `.gitignore`.
