# Function guide for `moller_vortex`

This file is a working guide to the main functions in the project. It is written for continuing analytic and numerical work, not as a restricted public API. The package namespace is intentionally open, so functions can be accessed either as

```python
import moller_vortex as mv

packet = mv.LGPacket(...)
```

or, in exploratory scripts,

```python
from moller_vortex import *

packet = LGPacket(...)
```

The recommended style for longer work is `import moller_vortex as mv`, because it makes it clear which functions come from this project.

---

## 1. Global conventions

### Units

The code uses natural units:

$$
\hbar=c=1.
$$

Masses, energies and momenta are measured in MeV. Lengths and impact parameters are measured in MeV$^{-1}$.

The electromagnetic coupling is defined in the Heaviside-Lorentz convention:

$$
e^2=4\pi\alpha.
$$

The relevant constants live in `constants.py`:

```python
ELECTRON_MASS
ALPHA_EM
ELECTRON_CHARGE
PI
FLOAT_DTYPE
COMPLEX_DTYPE
```

### Vectors

The project uses ordinary NumPy arrays for vectors.

A transverse vector has shape `(2,)`:

```python
K_perp = np.array([Kx, Ky])
```

A three-momentum has shape `(3,)`:

```python
k = np.array([kx, ky, kz])
```

The functions `vec2(...)` and `vec3(...)` check this at function boundaries. They are not physics functions; they only prevent accidental shape errors.

### Helicity labels

Helicities are represented as exact floating labels:

```python
+0.5
-0.5
```

The function `helicity(lam)` accepts only these two values. This is deliberate: helicity is a discrete label, not an approximate floating variable.

---

## 2. Typical workflow

A standard calculation has the following structure.

```python
import numpy as np
import moller_vortex as mv

packet1 = mv.LGPacket(
    ell=1,
    sigma_perp=0.18,
    sigma_par=0.35,
    kbar_z=20.0,
)

packet2 = mv.LGPacket(
    ell=-1,
    sigma_perp=0.18,
    sigma_par=0.35,
    kbar_z=-20.0,
)

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

For repeated scans, compute normalizations once when the packet parameters are fixed. Do not recompute `N1`, `N2` inside every call unless the packet parameters change.

---

## 3. `accuracy.py`

### `NumericalAccuracy`

```python
NumericalAccuracy(
    quad_epsabs=1.0e-10,
    quad_epsrel=1.0e-10,
    quad_limit=300,
    root_residual_atol=1.0e-10,
)
```

This dataclass stores the global numerical precision policy. `quad_epsabs`, `quad_epsrel`, and `quad_limit` are passed to SciPy adaptive quadratures. `root_residual_atol` is reserved for delta-reduced S-matrix routines where a candidate root is accepted only if the energy-conservation residual is small enough.

Physical role: none. It only controls numerical accuracy.

Use:

```python
accuracy = NumericalAccuracy(
    quad_epsabs=1e-11,
    quad_epsrel=1e-11,
    quad_limit=500,
    root_residual_atol=1e-11,
)
N = normalization_constant(packet, accuracy=accuracy)
```

### `ACCURACY`

Default instance of `NumericalAccuracy` used when no explicit accuracy object is provided.

---

## 4. `kinematics.py`

### `vec2(v)`

```python
vec2(v) -> np.ndarray
```

Checks that `v` is a transverse vector of shape `(2,)` and returns it as a `float64` NumPy array.

Use it at boundaries, not repeatedly inside algebraic expressions.

### `vec3(v)`

```python
vec3(v) -> np.ndarray
```

Checks that `v` is a three-vector of shape `(3,)` and returns it as a `float64` NumPy array.

### `energy(k, m=ELECTRON_MASS)`

Computes the on-shell energy:

$$
E_k=\sqrt{m^2+\mathbf k^2}.
$$

Input:

```python
k = np.array([kx, ky, kz])
E = energy(k)
```

The function raises an error for negative mass.

### `helicity(lam)`

Validates a helicity label. It returns `lam` if `lam` is exactly `+0.5` or `-0.5`; otherwise it raises `ValueError`.

### `kron_delta(a, b)`

Discrete Kronecker delta:

$$
\delta_{ab}=\begin{cases}
1, & a=b,\\
0, & a\ne b.
\end{cases}
$$

Used for helicity-conserving impulse amplitudes.

### `relative_error(a, b)`

Numerical relative error with `b` used as the reference value:

$$
\mathrm{err}(a,b)=\frac{|a-b|}{|b|}.
$$

If the reference value is exactly zero, the function returns `0.0` when `a=0` and `np.inf` otherwise. This keeps absolute and relative comparisons conceptually separate.

---

## 5. `packets.py`

### `LGPacket`

```python
LGPacket(
    ell: int,
    sigma_perp: float,
    sigma_par: float,
    kbar_z: float,
)
```

Container for one on-axis momentum-space Laguerre-Gaussian packet. The code assumes

$$
\bar{\mathbf k}_\perp=0.
$$

Fields:

- `ell`: integer orbital angular momentum charge;
- `sigma_perp`: transverse momentum width;
- `sigma_par`: longitudinal momentum width;
- `kbar_z`: central longitudinal momentum.

The object does **not** store the normalization constant. This is intentional: for scans, the physical parameters and the normalization are kept explicit.

Validation rule:

$$
\sigma_\perp>0,
\qquad
\sigma_\parallel>0,
\qquad
\sigma_\perp\le\sigma_\parallel.
$$

The last condition is required for the on-axis normalization integral used here.

Example:

```python
packet = LGPacket(ell=2, sigma_perp=0.2, sigma_par=0.4, kbar_z=20.0)
```

### `central_energy(packet, m=ELECTRON_MASS)`

Computes the central on-axis energy:

$$
\varepsilon=\sqrt{m^2+\bar k_z^2}.
$$

Example:

```python
eps = central_energy(packet)
```

### `normalization_constant(packet, m=ELECTRON_MASS, accuracy=None)`

Computes the relativistic normalization constant $N_\ell$ for the on-axis packet.

The packet is normalized by

$$
\int \frac{d^3k}{(2\pi)^3 2E_k}\,|\phi_\ell(\mathbf k)|^2=1.
$$

For $\bar{\mathbf k}_\perp=0$, the code uses the one-dimensional formula obtained after analytic integration over azimuth and longitudinal momentum:

$$
N_\ell=
\left[
\frac{e^{2m^2/\sigma_\parallel^2}}
{4\pi^2\sigma_\perp^{2|\ell|}|\ell|!}
\int_0^\infty dk_\perp\,
 k_\perp^{2|\ell|+1}
 e^{-\left(1/\sigma_\perp^2-1/\sigma_\parallel^2\right)k_\perp^2}
 K_0\!\left(
 \frac{2m\sqrt{m^2+k_\perp^2}}{\sigma_\parallel^2}
 \right)
\right]^{-1/2}.
$$

Implementation details:

- the integral is evaluated with `scipy.integrate.quad` on $[0,\infty)$;
- the modified Bessel function is `scipy.special.kv(0, ...)`;
- no normalization is cached inside `LGPacket`.

Example:

```python
N = normalization_constant(packet)
```

For scans:

```python
packets = [LGPacket(ell=ell, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
           for ell in range(-3, 4)]
normalizations = [normalization_constant(packet) for packet in packets]
```

### `spherical_normalization_constant(packet, m=ELECTRON_MASS)`

Closed expression for the spherical limit

$$
\sigma_\perp=\sigma_\parallel=\sigma.
$$

The formula used is

$$
N_\ell^{\mathrm{sph}}=
\frac{2^{3/2}\pi e^{-m^2/\sigma^2}}
{\sigma\sqrt{K_{|\ell|+1}(2m^2/\sigma^2)}}.
$$

This function is mainly a test/reference for `normalization_constant(...)`.

### `lg_packet_phi(k, packet, N, m=ELECTRON_MASS, impact_b=(0.0, 0.0))`

Evaluates the momentum-space packet at a three-momentum `k`.

The on-axis packet is

$$
\phi_\ell(\mathbf k)=
N_\ell
\frac{k_\perp^{|\ell|}}
{\sigma_\perp^{|\ell|}\sqrt{|\ell|!}}
\exp\left[
\frac{(E_k-\bar E)^2}{2\sigma_\parallel^2}
-
\frac{k_\perp^2}{2\sigma_\perp^2}
-
\frac{(k_z-\bar k_z)^2}{2\sigma_\parallel^2}
+i\ell\phi_k
+i\mathbf b_\perp\cdot\mathbf k_\perp
\right].
$$

Important: `impact_b` should normally be used only for the displaced second incoming packet. In the S-matrix functions the impact parameter is handled at the S-matrix level, so users usually do not need to call `lg_packet_phi(...)` directly.

---

## 6. `amplitudes.py`

### `moller_amplitude_impulse(...)`

```python
moller_amplitude_impulse(k1, k2, k3, k4, lam1, lam2, lam3, lam4)
```

Computes the ultrarelativistic paraxial impulse approximation to the plane-wave Møller amplitude:

$$
\mathcal M_{\mathrm{imp}}
=
-
\frac{8e^2\sqrt{E_1E_2E_3E_4}}
{|\mathbf k_{3\perp}-\mathbf k_{1\perp}|^2}
\delta_{\lambda_3\lambda_1}\delta_{\lambda_4\lambda_2}.
$$

Inputs:

- `k1`, `k2`: incoming three-momenta;
- `k3`, `k4`: final three-momenta;
- `lam1`, `lam2`, `lam3`, `lam4`: helicity labels;
- optional `m` and `e_charge`.

No regularization is applied. If

$$
|\mathbf k_{3\perp}-\mathbf k_{1\perp}|^2=0,
$$

the function raises `ZeroDivisionError`.

Use this function when checking the plane-wave impulse amplitude directly. The closed packet S-matrix functions call the transverse-integral version instead.

---

## 7. `transverse.py`

The functions in this module evaluate the transverse integral in the closed
impulse S-matrix.  The implemented integral is the first-order expansion of the
transverse denominator.

Define

$$
k_+=k_x+i k_y,
\qquad
k_-=k_x-i k_y,
$$

$$
K_+=K_x+iK_y,
\qquad
K_-=K_x-iK_y,
$$

and

$$
\chi_+=\frac{e^{-i\phi_3}}{k_{3\perp}},
\qquad
\chi_-=\frac{e^{+i\phi_3}}{k_{3\perp}}.
$$

The denominator expansion is

$$
\frac{1}{|\mathbf k_{3\perp}-\mathbf k_\perp|^2}
\simeq
\frac{1}{k_{3\perp}^2}
\left(1+\chi_+ k_+ + \chi_- k_-\right).
$$

The transverse Gaussian is

$$
\exp\left[
-\frac{\alpha}{2} k_\perp^2
+
\beta k_\perp\cdot K_\perp
-
\frac{\gamma}{2}|K_\perp-k_\perp|^2
-
i b_\perp\cdot k_\perp
\right].
$$

Introduce

$$
A=\alpha+\gamma,
$$

$$
J_0=(\beta+\gamma)K_\perp-i b_\perp,
$$

$$
p_+=\frac{J_{0x}+iJ_{0y}}{A},
\qquad
p_-=\frac{J_{0x}-iJ_{0y}}{A},
$$

$$
q_+=K_+-p_+,
\qquad
q_-=K_--p_-.
$$

The common factor in all closed transverse expressions is

$$
\mathcal P
=
\frac{2\pi}{A k_{3\perp}^2}
\exp\left[
\frac{J_0^2}{2A}
-
\frac{\gamma K_\perp^2}{2}
\right].
$$

Here all products are bilinear transverse products, not Hermitian scalar
products.

### `laguerre_derivative(a, b, c1, c2, c12)`

Computes

$$
D_{a,b}(c_1,c_2;c_{12})
=
\left.
\partial_{t_1}^a\partial_{t_2}^b
\exp(c_1t_1+c_2t_2-c_{12}t_1t_2)
\right|_{t_1=t_2=0}.
$$

The direct finite-sum identity is

$$
D_{a,b}
=
\sum_{r=0}^{\min(a,b)}
\frac{a!b!}{(a-r)!(b-r)!r!}
(-c_{12})^r c_1^{a-r} c_2^{b-r}.
$$

The Laguerre form used in the code is

$$
D_{a,b}
=
(-c_{12})^a a! c_2^{b-a}
L_a^{b-a}\left(\frac{c_1c_2}{c_{12}}\right),
\qquad b\ge a,
$$

and

$$
D_{a,b}
=
(-c_{12})^b b! c_1^{a-b}
L_b^{a-b}\left(\frac{c_1c_2}{c_{12}}\right),
\qquad a>b.
$$

There is no extra factor $(-c_2)^{b-a}$.  This is fixed by the limiting case

$$
D_{0,b}=c_2^b.
$$

Negative derivative orders are not accepted by `laguerre_derivative`; zero-OAM
cases are handled separately in `transverse_integral_explicit`.

### Closed transverse expressions

Let

$$
n=|\ell_1|,
\qquad
m=|\ell_2|.
$$

The implemented expressions are the following.

For $\ell_1=0,\ell_2=0$:

$$
I_\perp=\mathcal P\left(1+\chi_+p_++\chi_-p_-\right).
$$

For $\ell_1=0,\ell_2=m>0$:

$$
I_\perp=\mathcal P\left[
q_+^m
+
\chi_+p_+q_+^m
+
\chi_-\left(p_-q_+^m-\frac{2m}{A}q_+^{m-1}\right)
\right].
$$

For $\ell_1=0,\ell_2=-m<0$:

$$
I_\perp=\mathcal P\left[
q_-^m
+
\chi_+\left(p_+q_-^m-\frac{2m}{A}q_-^{m-1}\right)
+
\chi_-p_-q_-^m
\right].
$$

For $\ell_1=n>0,\ell_2=0$:

$$
I_\perp=\mathcal P\left[
p_+^n
+
\chi_+p_+^{n+1}
+
\chi_-\left(p_-p_+^n+\frac{2n}{A}p_+^{n-1}\right)
\right].
$$

For $\ell_1=-n<0,\ell_2=0$:

$$
I_\perp=\mathcal P\left[
p_-^n
+
\chi_+\left(p_+p_-^n+\frac{2n}{A}p_-^{n-1}\right)
+
\chi_-p_-^{n+1}
\right].
$$

For equal positive signs, $\ell_1=n>0,\ell_2=m>0$:

$$
I_\perp=\mathcal P\left[
p_+^n q_+^m
+
\chi_+p_+^{n+1}q_+^m
+
\chi_-\left(
p_-p_+^n q_+^m
+
\frac{2n}{A}p_+^{n-1}q_+^m
-
\frac{2m}{A}p_+^nq_+^{m-1}
\right)
\right].
$$

For equal negative signs, $\ell_1=-n<0,\ell_2=-m<0$:

$$
I_\perp=\mathcal P\left[
p_-^n q_-^m
+
\chi_+\left(
p_+p_-^n q_-^m
+
\frac{2n}{A}p_-^{n-1}q_-^m
-
\frac{2m}{A}p_-^nq_-^{m-1}
\right)
+
\chi_-p_-^{n+1}q_-^m
\right].
$$

For opposite signs $\ell_1=n>0,\ell_2=-m<0$, set

$$
D_{a,b}=D_{a,b}(p_+,q_-;2/A).
$$

Then

$$
I_\perp=\mathcal P\left[
D_{n,m}
+
\chi_+D_{n+1,m}
+
\chi_-\left(K_-D_{n,m}-D_{n,m+1}\right)
\right].
$$

For opposite signs $\ell_1=-n<0,\ell_2=m>0$, set

$$
D_{a,b}=D_{a,b}(p_-,q_+;2/A).
$$

Then

$$
I_\perp=\mathcal P\left[
D_{n,m}
+
\chi_-D_{n+1,m}
+
\chi_+\left(K_+D_{n,m}-D_{n,m+1}\right)
\right].
$$

These opposite-sign expressions are equivalent to the $k_\perp^2$ form in the
analytic derivation, but avoid negative derivative orders and make the zero-OAM
limits unambiguous.

### `transverse_integral_explicit(...)`

```python
transverse_integral_explicit(
    ell1,
    ell2,
    k3_perp,
    K_perp,
    b_perp,
    alpha,
    beta,
    gamma,
    return_case=False,
)
```

Computes the closed first-order transverse integral.  The returned value already
includes $\mathcal P$, so these factors must not be multiplied again outside
this function.

### `vortex_factor(z, ell)`

Returns the complex vortex factor used in direct numerical checks:

$$
V_\ell(z)=
\begin{cases}
z^\ell, & \ell>0,\\
1, & \ell=0,\\
(z^*)^{|\ell|}, & \ell<0.
\end{cases}
$$

Here $z=k_x+i k_y$ or $z=(K_x-k_x)+i(K_y-k_y)$.

### `transverse_integral_numeric_quad(...)`

Direct numerical check of the same transverse integral:

$$
I_\perp=
\int_0^\infty r\,dr
\int_0^{2\pi}d\phi\,F(r,\phi).
$$

The angular integral is evaluated on a uniform periodic grid; for each angular
node the radial integral over $[0,\infty)$ is computed by `scipy.integrate.quad`,
with real and imaginary parts integrated separately.  This function is a
verification routine.  Use `transverse_integral_explicit(...)` for scans.

---

## 8. `smatrix.py`

### `impulse_parameters(packet1, packet2, k3, k4, impact_b, m=ELECTRON_MASS)`

Computes intermediate parameters used by the closed impulse S-matrix.

Main returned quantities:

- `K_vec`: $\mathbf K=\mathbf k_3+\mathbf k_4$;
- `K_perp`: $\mathbf K_\perp$;
- `E3`, `E4`, `E_K`;
- `eps1`, `eps2`: central packet energies;
- `v1`, `v2`: longitudinal velocities $\bar k_z/\varepsilon$;
- `DeltaKz`: $K_z-\bar k_{1z}-\bar k_{2z}$;
- `Xi0`: constant part of the exponent outside the transverse integral;
- `A_long`, `Omega_long`: longitudinal impulse integration parameters;
- `alpha`, `beta`, `gamma`: transverse Gaussian coefficients.

This function is useful for debugging and for inspecting the decomposition. Most calculations should call `S_impulse_closed_form(...)` instead.

Example:

```python
pars = impulse_parameters(packet1, packet2, k3, k4, impact_b)
print(pars["alpha"], pars["beta"], pars["gamma"])
```

### `S_impulse_common_factor(...)`

Computes the factor multiplying the transverse integral in the closed impulse S-matrix.

The full structure is

$$
S_{fi}^{\mathrm{imp}}
=
\mathcal C\,I_\perp.
$$

This function returns

```python
common_factor, details
```

where `common_factor` is $\mathcal C$ and `details` contains intermediate quantities.

It also applies the helicity deltas. If the impulse helicity condition fails,

$$
\delta_{\lambda_3\lambda_1}\delta_{\lambda_4\lambda_2}=0,
$$

then the returned common factor is zero and `details` contains a reason.

Use this function when you want to compare different transverse-integral implementations with the same outer prefactor.

### `S_impulse_closed_form(...)`

```python
S_impulse_closed_form(
    k3,
    k4,
    packet1,
    packet2,
    lam1,
    lam2,
    lam3,
    lam4,
    impact_b=(0.0, 0.0),
    N1=None,
    N2=None,
    accuracy=None,
    return_details=False,
)
```

Main production function for the closed impulse-approximation S-matrix.

It performs:

1. helicity selection;
2. computation or use of supplied normalizations `N1`, `N2`;
3. construction of the common prefactor;
4. analytic transverse integration via `transverse_integral_explicit(...)`;
5. multiplication into the final complex S-matrix element.

Use this function for parameter scans.

Recommended for scans:

```python
N1 = normalization_constant(packet1)
N2 = normalization_constant(packet2)

S = S_impulse_closed_form(
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

To inspect internals:

```python
S, details = S_impulse_closed_form(..., return_details=True)
print(details["transverse_case"])
print(details["Iperp"])
print(details["Xi0"])
```

### `S_impulse_numeric_transverse_quad(...)`

Same S-matrix factorization as `S_impulse_closed_form(...)`, but replaces the analytic transverse integral by `transverse_integral_numeric_quad(...)`.

Use it for checks:

```python
S_closed = S_impulse_closed_form(...)
S_numeric = S_impulse_numeric_transverse_quad(..., n_phi=64)
print(relative_error(S_closed, S_numeric))
```

Do not use this for large scans unless needed; it is much slower.

---

## 9. Built-in error-reporting functions

The project does not use a separate pytest-style test folder. Instead, numerical consistency checks are ordinary package functions that can be called directly from `analysis_workspace.ipynb` or from a script. They do not decide whether a result has passed or failed. They only compute and print numerical errors.

### `check_normalization(...)`

Compares `normalization_constant(...)` with the closed spherical formula for

$$
\sigma_\perp=\sigma_\parallel.
$$

This avoids the circular check of computing a normalization from an integral and substituting it back into the same integral.

Example:

```python
errors = check_normalization(accuracy=accuracy)
```

The returned object is a dictionary mapping a label to a relative error.

### `check_transverse_integral(...)`

Compares:

```python
transverse_integral_explicit(...)
```

against

```python
transverse_integral_numeric_quad(...)
```

for several sign combinations of $\ell_1,\ell_2$. This checks the case logic and the Laguerre derivative formula.

Example:

```python
errors = check_transverse_integral(
    accuracy=accuracy,
    n_phi=16,
)
```

The parameter `n_phi` controls the angular resolution of the direct numerical transverse integral. It is intentionally an explicit argument of the check function, not a hidden field of `NumericalAccuracy`.

### `check_smatrix(...)`

Compares:

```python
S_impulse_closed_form(...)
```

against

```python
S_impulse_numeric_transverse_quad(...)
```

The two functions share the same common prefactor but use different transverse integrations. This check tests the assembly of the full impulse S-matrix.

Example:

```python
errors = check_smatrix(
    accuracy=accuracy,
    n_phi=16,
)
```

### `run_all_checks(...)`

Runs all built-in error-reporting functions:

```python
accuracy = NumericalAccuracy(
    quad_epsabs=1.0e-10,
    quad_epsrel=1.0e-10,
    quad_limit=300,
    root_residual_atol=1.0e-10,
)

results = run_all_checks(
    accuracy=accuracy,
    n_phi=16,
    verbose=True,
)
```

The returned object has the form:

```python
{
    "normalization": {...},
    "transverse_integral": {...},
    "smatrix": {...},
}
```

No tolerance is applied inside these functions. The notebook or script that calls them decides how to interpret the returned errors.

---

## 10. Recommended patterns for future work

### Scan over a discrete OAM charge

```python
base1 = LGPacket(ell=0, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
packet2 = LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)
N2 = normalization_constant(packet2)

results = []
for ell in range(-4, 5):
    packet1 = LGPacket(
        ell=ell,
        sigma_perp=base1.sigma_perp,
        sigma_par=base1.sigma_par,
        kbar_z=base1.kbar_z,
    )
    N1 = normalization_constant(packet1)
    S = S_impulse_closed_form(
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
    results.append((ell, S))
```

### Scan over a continuous width

```python
sigma_values = np.linspace(0.12, 0.25, 20)
results = []

for sigma_perp in sigma_values:
    packet1 = LGPacket(ell=1, sigma_perp=sigma_perp, sigma_par=0.35, kbar_z=20.0)
    N1 = normalization_constant(packet1)
    S = S_impulse_closed_form(
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
    results.append((sigma_perp, S))
```

### Keep normalization explicit

Do this:

```python
N1 = normalization_constant(packet1)
S = S_impulse_closed_form(..., N1=N1)
```

rather than recomputing `N1` inside every call during a scan.

### Use `return_details=True` when debugging

```python
S, details = S_impulse_closed_form(..., return_details=True)
print(details.keys())
```

Useful keys include:

```text
K_perp, E_K, eps1, eps2, v1, v2, Xi0, alpha, beta, gamma,
Iperp, transverse_case, common_factor
```

---

## 11. What not to assume

- The code is on-axis: $\bar{\mathbf k}_\perp=0$.
- The S-matrix functions implement impulse approximation, not the full Møller amplitude.
- The transverse integral uses the first-order expansion of the transverse denominator.
- No artificial regularization is applied.
- The impact parameter is associated with the second incoming packet in the S-matrix assembly.
