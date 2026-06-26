"""Exact-time S-matrix representations."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.special import ive

from .constants import (
    COMPLEX_DTYPE,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
)
from .kinematics import vec2, vec3
from .packets import LGPacket, central_energy, resolve_normalizations
from .quadrature import ExactTimeQuadrature, nodes_and_weights
from .smatrix_common import _mode, _packet_denominator, _spinor_scale


def _time_model(model: str, theta_method: str) -> tuple[str, bool]:
    """Return internal spinor model and whether theta is analytic."""
    model = _mode(model, ("ultrarelativistic", "massive"), "model")
    analytic_theta = theta_method == "analytic"
    if model == "massive" and analytic_theta:
        raise ValueError(
            "model='massive' requires numerical theta quadrature. "
            "Use ExactTimeQuadrature(theta_method='trapezoid')."
        )
    spinor_model = "ultrarelativistic" if model == "ultrarelativistic" else "massive"
    return spinor_model, analytic_theta


def _vortex_monomial_z(z, ell: int):
    """Return the Cartesian vortex monomial for scalar or array ``z``."""
    if ell >= 0:
        return z ** ell
    return np.conjugate(z) ** (-ell)


@lru_cache(maxsize=64)
def _cached_theta_nodes(n_theta: int, theta_method: str):
    """Return cached theta nodes, weights and trigonometric arrays."""
    if theta_method == "analytic":
        raise ValueError("theta_method='analytic' has no theta nodes.")
    theta_nodes, theta_weights = nodes_and_weights(
        (0.0, 2.0 * PI),
        n_theta,
        method=theta_method,
        endpoint=False,
    )
    return theta_nodes, theta_weights, np.cos(theta_nodes), np.sin(theta_nodes)


@lru_cache(maxsize=64)
def _cached_unit_kappa_nodes(n_kappa: int, kappa_method: str):
    """Return cached unit-interval kappa nodes and weights."""
    return nodes_and_weights(
        (0.0, 1.0),
        n_kappa,
        method=kappa_method,
        endpoint=False,
    )


def _transverse_factor(
    qx,
    qy,
    *,
    Kx,
    Ky,
    packet1_kbar_z,
    k3x,
    k3y,
    k3z,
    E3,
    k3_perp_sq,
    b_perp,
    s1p: float,
    s2p: float,
    ell1: int,
    ell2: int,
    xi=None,
    m: float = ELECTRON_MASS,
    spinor_model: str,
):
    """Return the exact-time transverse integrand on a batched q-grid."""
    K_minus_qx = Kx - qx
    K_minus_qy = Ky - qy

    if spinor_model == "massive":
        if xi is None:
            raise ValueError("xi must be provided for the massive paraxial denominator.")
        k1z = packet1_kbar_z + xi
        q_sq = qx ** 2 + qy ** 2
        E1_internal = np.sqrt(m ** 2 + q_sq + k1z ** 2)
        diff_x = k3x - qx
        diff_y = k3y - qy
        diff_z = k3z - k1z
        t_channel = (
            (E1_internal - E3) ** 2
            - diff_x ** 2
            - diff_y ** 2
            - diff_z ** 2
        )
        denominator = -1.0 / t_channel
    else:
        denominator = (
            1.0
            / k3_perp_sq
            * (1.0 + 2.0 * (qx * k3x + qy * k3y) / k3_perp_sq)
        )

    q_sq = qx ** 2 + qy ** 2
    K_minus_q_sq = K_minus_qx ** 2 + K_minus_qy ** 2
    phase = b_perp[0] * K_minus_qx + b_perp[1] * K_minus_qy
    transverse_exp = np.exp(
        -0.5 * q_sq / (s1p ** 2)
        -0.5 * K_minus_q_sq / (s2p ** 2)
        + 1j * phase
    )
    z1 = qx + 1j * qy
    z2 = K_minus_qx + 1j * K_minus_qy
    vortex_factor = _vortex_monomial_z(z1, ell1) * _vortex_monomial_z(z2, ell2)
    return denominator * transverse_exp * vortex_factor


def _angular_kernels_with_exponent(min_m: int, max_m: int, u, v, C_perp):
    """Return exp(C_perp) K_m(u, v) for a contiguous integer m range."""
    u = np.asarray(u, dtype=COMPLEX_DTYPE)
    v = np.asarray(v, dtype=COMPLEX_DTYPE)
    C_perp = np.asarray(C_perp, dtype=COMPLEX_DTYPE)
    u, v, C_perp = np.broadcast_arrays(u, v, C_perp)

    zero_mask = (np.abs(u) == 0.0) | (np.abs(v) == 0.0)
    bessel_mask = ~zero_mask
    kernels = {}

    if np.any(bessel_mask):
        sqrt_u = np.sqrt(u[bessel_mask])
        sqrt_v = np.sqrt(v[bessel_mask])
        z = 2.0 * sqrt_u * sqrt_v
        scale = np.exp(C_perp[bessel_mask] + np.abs(np.real(z)))
        ratio_base = sqrt_v / sqrt_u

    if np.any(zero_mask):
        u_zero = u[zero_mask]
        v_zero = v[zero_mask]
        exp_C_zero = np.exp(C_perp[zero_mask])
        u_is_zero = np.abs(u_zero) == 0.0
        v_is_zero = np.abs(v_zero) == 0.0
        both_zero = u_is_zero & v_is_zero
        u_only_zero = u_is_zero & ~v_is_zero
        v_only_zero = v_is_zero & ~u_is_zero

    for m in range(min_m, max_m + 1):
        kernel = np.empty_like(u, dtype=COMPLEX_DTYPE)
        if np.any(bessel_mask):
            kernel[bessel_mask] = scale * ive(abs(m), z) * ratio_base ** m
        if np.any(zero_mask):
            zero_kernel = np.zeros_like(u_zero, dtype=COMPLEX_DTYPE)
            if m == 0:
                zero_kernel[both_zero] = 1.0
            if m >= 0:
                zero_kernel[u_only_zero] = (
                    v_zero[u_only_zero] ** m / math.factorial(m)
                )
            if m <= 0:
                zero_kernel[v_only_zero] = (
                    u_zero[v_only_zero] ** (-m) / math.factorial(-m)
                )
            kernel[zero_mask] = exp_C_zero * zero_kernel
        kernels[m] = kernel

    return kernels


def _binomial_terms(base, kappa, power: int, sign: float):
    """Return binomial terms for one source derivative block."""
    return [
        math.comb(power, index)
        * base ** (power - index)
        * (sign * kappa) ** index
        for index in range(power + 1)
    ]


def _product_terms(q0, p0, kappa, q_power: int, p_power: int):
    """Return coefficients summed over source derivatives with equal order."""
    q_terms = _binomial_terms(q0, kappa, q_power, +1.0)
    p_terms = _binomial_terms(p0, kappa, p_power, -1.0)
    coefficients = []
    for order in range(q_power + p_power + 1):
        coeff = np.zeros_like(kappa, dtype=COMPLEX_DTYPE)
        q_min = max(0, order - p_power)
        q_max = min(q_power, order)
        for q_index in range(q_min, q_max + 1):
            coeff = coeff + q_terms[q_index] * p_terms[order - q_index]
        coefficients.append(coeff)
    return coefficients


def _angular_polynomial(plus_terms, minus_terms, kernels_scaled):
    """Return exp(C_perp) D from precomputed source-derivative terms."""
    total = np.zeros_like(plus_terms[0], dtype=COMPLEX_DTYPE)
    for plus_order, plus_term in enumerate(plus_terms):
        for minus_order, minus_term in enumerate(minus_terms):
            total = total + (
                plus_term
                * minus_term
                * kernels_scaled[plus_order - minus_order]
            )
    return total


def _analytic_theta_integral(
    kappa,
    *,
    Kx,
    Ky,
    q0x,
    q0y,
    k3x,
    k3y,
    k3_perp_sq,
    b_perp,
    s1p: float,
    s2p: float,
    ell1: int,
    ell2: int,
):
    """Return the analytic theta integral for the ultrarelativistic integrand."""
    kappa = np.asarray(kappa, dtype=FLOAT_DTYPE)

    p0x = Kx - q0x
    p0y = Ky - q0y

    q0_plus = q0x + 1j * q0y
    q0_minus = q0x - 1j * q0y
    p0_plus = p0x + 1j * p0y
    p0_minus = p0x - 1j * p0y

    q0_sq = q0x ** 2 + q0y ** 2
    p0_sq = p0x ** 2 + p0y ** 2
    kappa_sq = kappa ** 2
    C_perp = (
        -0.5 * (q0_sq + kappa_sq) / (s1p ** 2)
        -0.5 * (p0_sq + kappa_sq) / (s2p ** 2)
        + 1j * (b_perp[0] * p0x + b_perp[1] * p0y)
    )

    alpha_x = kappa * (-q0x / (s1p ** 2) + p0x / (s2p ** 2) - 1j * b_perp[0])
    alpha_y = kappa * (-q0y / (s1p ** 2) + p0y / (s2p ** 2) - 1j * b_perp[1])
    u = 0.5 * (alpha_x - 1j * alpha_y)
    v = 0.5 * (alpha_x + 1j * alpha_y)

    L1 = abs(ell1)
    L2 = abs(ell2)
    a = L1 if ell1 >= 0 else 0
    b = 0 if ell1 >= 0 else L1
    c = L2 if ell2 >= 0 else 0
    d = 0 if ell2 >= 0 else L2

    min_m = -(b + d + 1)
    max_m = a + c + 1
    kernels_scaled = _angular_kernels_with_exponent(min_m, max_m, u, v, C_perp)

    plus_base = _product_terms(q0_plus, p0_plus, kappa, a, c)
    plus_q = _product_terms(q0_plus, p0_plus, kappa, a + 1, c)
    minus_base = _product_terms(q0_minus, p0_minus, kappa, b, d)
    minus_q = _product_terms(q0_minus, p0_minus, kappa, b + 1, d)

    D0 = _angular_polynomial(plus_base, minus_base, kernels_scaled)
    D_plus = _angular_polynomial(plus_q, minus_base, kernels_scaled)
    D_minus = _angular_polynomial(plus_base, minus_q, kernels_scaled)

    k3_plus = k3x + 1j * k3y
    k3_minus = k3x - 1j * k3y
    return 2.0 * PI * (
        D0 / k3_perp_sq
        + (k3_minus * D_plus + k3_plus * D_minus)
        / (k3_perp_sq ** 2)
    )


def S_time_grid(
    k3x,
    k3y,
    k3z,
    k4x,
    k4y,
    k4z,
    packet1: LGPacket,
    packet2: LGPacket,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    impact_b=(0.0, 0.0),
    N1: float | None = None,
    N2: float | None = None,
    quadrature: ExactTimeQuadrature | None = None,
    model: str = "ultrarelativistic",
    batch_size: int = 32,
):
    """Vectorized exact-time S matrix on broadcastable momentum grids.

    The outer probability grid is processed in batches. Numerical theta
    quadrature builds arrays of shape ``(batch, n_radial, n_theta)``; analytic
    theta integration uses ``(batch, n_radial)`` arrays. Both paths keep memory
    bounded while removing Python loops over individual final-momentum points.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    quadrature = ExactTimeQuadrature() if quadrature is None else quadrature
    spinor_model, analytic_theta = _time_model(model, quadrature.theta_method)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    arrays = np.broadcast_arrays(k3x, k3y, k3z, k4x, k4y, k4z)
    shape = arrays[0].shape
    out = np.zeros(shape, dtype=COMPLEX_DTYPE)

    k3x_f, k3y_f, k3z_f, k4x_f, k4y_f, k4z_f = (
        np.ravel(np.asarray(arr, dtype=FLOAT_DTYPE)) for arr in arrays
    )
    out_f = out.ravel()
    b_perp = vec2(impact_b)

    Kx = k3x_f + k4x_f
    Ky = k3y_f + k4y_f
    Kz = k3z_f + k4z_f
    K_perp_sq = Kx ** 2 + Ky ** 2
    k3_perp_sq = k3x_f ** 2 + k3y_f ** 2

    if model == "ultrarelativistic" and np.any(k3_perp_sq == 0.0):
        raise ValueError("Ultrarelativistic denominator requires nonzero |k3_perp|.")

    E3 = np.sqrt(m ** 2 + k3_perp_sq + k3z_f ** 2)
    E4 = np.sqrt(m ** 2 + k4x_f ** 2 + k4y_f ** 2 + k4z_f ** 2)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    spinor_scale = _spinor_scale(
        spinor_model,
        eps1,
        eps2,
        E3,
        E4,
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    if np.all(spinor_scale == 0.0):
        return out

    gamma1 = eps1 / m
    gamma2 = eps2 / m
    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s1z = packet1.sigma_par
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par
    inv_eps1_gamma1_2 = 1.0 / (eps1 * gamma1 ** 2)
    inv_eps2_gamma2_2 = 1.0 / (eps2 * gamma2 ** 2)
    inv_s1z2 = 1.0 / s1z ** 2
    inv_s2z2 = 1.0 / s2z ** 2

    A_z = 0.5 * (inv_eps1_gamma1_2 + inv_eps2_gamma2_2)
    B_z = 0.5 * (inv_s1z2 / gamma1 ** 2 + inv_s2z2 / gamma2 ** 2)
    C_z = v1 - v2 - DeltaKz * inv_eps2_gamma2_2
    D_z = (
        packet1.kbar_z * inv_s1z2
        - packet2.kbar_z * inv_s2z2
        - v1 * eps1 * inv_s1z2
        + v2 * eps2 * inv_s2z2
        + DeltaKz * inv_s2z2 / gamma2 ** 2
    )
    Omega_z = (
        eps1
        + eps2
        - E_K
        + v2 * DeltaKz
        + 0.5 * DeltaKz ** 2 * inv_eps2_gamma2_2
    )
    longitudinal_exp_arg = -0.5 * DeltaKz ** 2 * inv_s2z2 / gamma2 ** 2

    eta_perp = 1.0 / eps1 + 1.0 / eps2
    q0x = eps1 / (eps1 + eps2) * Kx
    q0y = eps1 / (eps1 + eps2) * Ky

    Delta0 = C_z ** 2 - 4.0 * A_z * (
        Omega_z + K_perp_sq / (2.0 * (eps1 + eps2))
    )
    active = Delta0 > 0.0
    if not np.any(active):
        return out

    sqrt_Delta0 = np.zeros_like(Delta0)
    sqrt_Delta0[active] = np.sqrt(Delta0[active])
    R = np.zeros_like(Delta0)
    R[active] = np.sqrt(Delta0[active] / (2.0 * A_z * eta_perp))

    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )
    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(E3 * E4 / (eps1 * eps2))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
        * spinor_scale
    )

    if not analytic_theta:
        _, theta_weights, cos_theta, sin_theta = _cached_theta_nodes(
            quadrature.n_theta,
            quadrature.theta_method,
        )
        theta_weights = theta_weights[None, None, :]
        cos_theta = cos_theta[None, None, :]
        sin_theta = sin_theta[None, None, :]

    active_indices = np.nonzero(active)[0]

    n_kappa = quadrature.n_kappa
    kappa_method = quadrature.kappa_method

    unit_nodes, unit_weights = _cached_unit_kappa_nodes(n_kappa, kappa_method)
    unit_nodes = unit_nodes[None, :, None]
    unit_weights = unit_weights[None, :, None]

    sigma_eff = 1.0 / np.sqrt(1.0 / (s1p ** 2) + 1.0 / (s2p ** 2))
    gaussian_fraction = s1p ** 2 / (s1p ** 2 + s2p ** 2)
    disk_curvature = np.zeros_like(Delta0)
    disk_curvature[active] = 2.0 * A_z * eta_perp / Delta0[active]

    for start in range(0, len(active_indices), batch_size):
        idx = active_indices[start : start + batch_size]
        bshape = (len(idx), 1, 1)

        if quadrature.kappa_n_sigma is None:
            radial_lower = np.zeros(len(idx), dtype=FLOAT_DTYPE)
            radial_upper = R[idx]
        else:
            gaussian_center_x = gaussian_fraction * Kx[idx]
            gaussian_center_y = gaussian_fraction * Ky[idx]
            center_distance = np.sqrt(
                (gaussian_center_x - q0x[idx]) ** 2
                + (gaussian_center_y - q0y[idx]) ** 2
            )
            support = quadrature.kappa_n_sigma * sigma_eff
            radial_lower = np.maximum(0.0, center_distance - support)
            radial_upper = np.minimum(R[idx], center_distance + support)

        width = np.maximum(radial_upper - radial_lower, 0.0)
        kappa_nodes = radial_lower.reshape(bshape) + width.reshape(bshape) * unit_nodes
        kappa_weights = width.reshape(bshape) * unit_weights

        curvature = disk_curvature[idx].reshape(bshape)
        discriminant_factor = 1.0 - curvature * kappa_nodes ** 2
        valid = discriminant_factor > 0.0
        sqrt_factor = np.sqrt(np.where(valid, discriminant_factor, 1.0))

        sqrt_D = sqrt_Delta0[idx].reshape(bshape)
        C = C_z[idx].reshape(bshape)
        D = D_z[idx].reshape(bshape)
        longitudinal_exp = longitudinal_exp_arg[idx].reshape(bshape)
        xi_plus = (-C + sqrt_D * sqrt_factor) / (2.0 * A_z)
        xi_minus = (-C - sqrt_D * sqrt_factor) / (2.0 * A_z)
        root_weight_plus = np.exp(
            longitudinal_exp - B_z * xi_plus ** 2 + D * xi_plus
        )
        root_weight_minus = np.exp(
            longitudinal_exp - B_z * xi_minus ** 2 + D * xi_minus
        )
        root_weight = root_weight_plus + root_weight_minus

        if analytic_theta:
            kappa_2d = kappa_nodes[:, :, 0]
            kappa_weights_2d = kappa_weights[:, :, 0]
            sqrt_factor_2d = sqrt_factor[:, :, 0]
            root_weight_2d = root_weight[:, :, 0]
            valid_2d = valid[:, :, 0]
            row_shape = (len(idx), 1)

            angular_integral = _analytic_theta_integral(
                kappa_2d,
                Kx=Kx[idx].reshape(row_shape),
                Ky=Ky[idx].reshape(row_shape),
                q0x=q0x[idx].reshape(row_shape),
                q0y=q0y[idx].reshape(row_shape),
                k3x=k3x_f[idx].reshape(row_shape),
                k3y=k3y_f[idx].reshape(row_shape),
                k3_perp_sq=k3_perp_sq[idx].reshape(row_shape),
                b_perp=b_perp,
                s1p=s1p,
                s2p=s2p,
                ell1=packet1.ell,
                ell2=packet2.ell,
            )
            weights = (
                kappa_weights_2d
                * kappa_2d
                / sqrt_factor_2d
                * root_weight_2d
                * valid_2d
            )
            integral = np.sum(weights * angular_integral, axis=1)
        else:
            qx = q0x[idx].reshape(bshape) + kappa_nodes * cos_theta
            qy = q0y[idx].reshape(bshape) + kappa_nodes * sin_theta
            weights = kappa_weights * theta_weights * kappa_nodes / sqrt_factor * valid

            if spinor_model == "massive":
                A_perp_plus = _transverse_factor(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    xi=xi_plus,
                    m=m,
                    spinor_model=spinor_model,
                )
                A_perp_minus = _transverse_factor(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    xi=xi_minus,
                    m=m,
                    spinor_model=spinor_model,
                )
                root_integrand = (
                    root_weight_plus * A_perp_plus
                    + root_weight_minus * A_perp_minus
                )
            else:
                A_perp = _transverse_factor(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    spinor_model=spinor_model,
                )
                root_integrand = root_weight * A_perp

            integral = np.sum(weights * root_integrand, axis=(1, 2))
        disk_factor = 1.0 / sqrt_Delta0[idx]
        out_f[idx] = (
            prefactor[idx]
            * disk_factor
            * integral
        )

    return out


def S_time(
    k3,
    k4,
    packet1: LGPacket,
    packet2: LGPacket,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    impact_b=(0.0, 0.0),
    N1: float | None = None,
    N2: float | None = None,
    quadrature: ExactTimeQuadrature | None = None,
    model: str = "ultrarelativistic",
    batch_size: int | None = None,
    return_details: bool = False,
):
    """S-matrix with the time integral evaluated by the delta representation."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    quadrature = ExactTimeQuadrature() if quadrature is None else quadrature
    k3_vec = vec3(k3)
    k4_vec = vec3(k4)
    S = S_time_grid(
        k3_vec[0],
        k3_vec[1],
        k3_vec[2],
        k4_vec[0],
        k4_vec[1],
        k4_vec[2],
        packet1,
        packet2,
        lam1=lam1,
        lam2=lam2,
        lam3=lam3,
        lam4=lam4,
        m=m,
        e_charge=e_charge,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
        quadrature=quadrature,
        model=model,
        batch_size=1 if batch_size is None else batch_size,
    ).reshape(-1)[0]

    if not return_details:
        return S

    spinor_model, analytic_theta = _time_model(model, quadrature.theta_method)
    K_vec = k3_vec + k4_vec
    K_perp = K_vec[:2]
    E3 = np.sqrt(m ** 2 + np.dot(k3_vec, k3_vec))
    E4 = np.sqrt(m ** 2 + np.dot(k4_vec, k4_vec))
    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    spinor_scale = _spinor_scale(
        spinor_model,
        eps1,
        eps2,
        E3,
        E4,
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    details = dict(
        model=model,
        analytic_theta=analytic_theta,
        spinor_scale=spinor_scale,
        K_vec=K_vec,
        K_perp=K_perp,
        Kz=K_vec[2],
        E3=E3,
        E4=E4,
        eps1=eps1,
        eps2=eps2,
        DeltaKz=K_vec[2] - packet1.kbar_z - packet2.kbar_z,
        impact_b=vec2(impact_b),
        quadrature=quadrature,
        S_time=S,
    )
    return S, details
