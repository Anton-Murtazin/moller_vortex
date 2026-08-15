"""Analytic paraxial ultrarelativistic S-matrix from PDF Eqs. (28)-(56)."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    COMPLEX_DTYPE,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PI,
    REAL_DTYPE,
    VortexPacket,
)
from .states import central_energy, effective_sigma


def _vectors(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _broadcast_vectors(first, second, size: int, names: tuple[str, str]):
    first = _vectors(first, size, names[0])
    second = _vectors(second, size, names[1])
    try:
        return np.broadcast_arrays(first, second)
    except ValueError as error:
        raise ValueError(f"{names[0]} and {names[1]} are not broadcastable.") from error


def _validate_norms(norms) -> tuple[float, float]:
    if len(norms) != 2:
        raise ValueError("norms must contain exactly (N1, N2).")
    values = (float(norms[0]), float(norms[1]))
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Both normalization constants must be positive and finite.")
    return values


def _mixed_polynomial(m: int, n: int, c1, c2, c12: float):
    """Return F_mn from PDF Eq. (53); negative indices give zero."""
    shape = np.broadcast(c1, c2).shape
    if m < 0 or n < 0:
        return np.zeros(shape, dtype=COMPLEX_DTYPE)
    result = np.zeros(shape, dtype=COMPLEX_DTYPE)
    for r in range(min(m, n) + 1):
        coefficient = math.comb(m, r) * math.comb(n, r) * math.factorial(r)
        result += (
            coefficient
            * (-c12) ** r
            * c1 ** (m - r)
            * c2 ** (n - r)
        )
    return result


def _dot_e(vector, sign: int):
    """Return the Euclidean dot product with e_+ or e_- from PDF Eq. (33)."""
    return vector[..., 0] + 1j * sign * vector[..., 1]


def longitudinal_factor(
    k3,
    k4,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    mass: float = ELECTRON_MASS,
):
    """Evaluate the separated longitudinal factor L in PDF Eq. (28)."""
    k3, k4 = _broadcast_vectors(k3, k4, 3, ("k3", "k4"))
    E3 = np.sqrt(mass**2 + np.sum(k3 * k3, axis=-1))
    E4 = np.sqrt(mass**2 + np.sum(k4 * k4, axis=-1))
    return _longitudinal_from_energies(
        E3,
        E4,
        k3[..., 2] + k4[..., 2],
        packet1,
        packet2,
        mass,
    )


def _longitudinal_from_energies(
    E3,
    E4,
    K_z,
    packet1: VortexPacket,
    packet2: VortexPacket,
    mass: float,
):
    E01 = central_energy(packet1, mass)
    E02 = central_energy(packet2, mass)
    v1 = packet1.k0_z / E01
    v2 = packet2.k0_z / E02
    delta_v = v1 - v2
    if abs(delta_v) <= np.finfo(REAL_DTYPE).eps:
        raise ValueError("The impulse approximation requires v1 != v2.")

    delta_Kz = K_z - packet1.k0_z - packet2.k0_z
    delta_E = E3 + E4 - E01 - E02
    sigma1 = effective_sigma(packet1, mass)
    sigma2 = effective_sigma(packet2, mass)
    exponent = (
        -(delta_E - v2 * delta_Kz) ** 2
        / (2.0 * sigma1**2 * delta_v**2)
        -(delta_E - v1 * delta_Kz) ** 2
        / (2.0 * sigma2**2 * delta_v**2)
    )
    return np.asarray(2.0 * PI / abs(delta_v) * np.exp(exponent), dtype=REAL_DTYPE)


def transverse_integral(
    k3_perp,
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    impact=(0.0, 0.0),
):
    """Evaluate T using the first-order propagator expansion in Eqs. (32)-(56).

    The function is fully vectorized over broadcastable ``(..., 2)`` momenta.
    It covers positive, negative, and zero OAM values. No pole cutoff is
    introduced because the already-expanded formulas (46) and (56) are used.
    """
    k3, K = _broadcast_vectors(k3_perp, K_perp, 2, ("k3_perp", "K_perp"))
    b = np.asarray(impact, dtype=REAL_DTYPE)
    if b.shape != (2,) or not np.all(np.isfinite(b)):
        raise ValueError("impact must be a finite vector with shape (2,).")

    k3_squared = np.sum(k3 * k3, axis=-1)
    if np.any(k3_squared <= 0.0):
        raise ZeroDivisionError("PDF Eq. (32) requires nonzero k3_perp.")

    sigma1_squared = packet1.sigma_perp**2
    sigma2_squared = packet2.sigma_perp**2
    sigma_total_squared = sigma1_squared + sigma2_squared
    sigma_overlap_squared = sigma1_squared * sigma2_squared / sigma_total_squared

    K_squared = np.sum(K * K, axis=-1)
    b_squared = float(np.dot(b, b))
    b_dot_K = b[0] * K[..., 0] + b[1] * K[..., 1]
    k3_dot_K = np.sum(k3 * K, axis=-1)
    k3_dot_b = k3[..., 0] * b[0] + k3[..., 1] * b[1]
    G0 = (
        2.0
        * PI
        * sigma_overlap_squared
        * np.exp(
            -K_squared / (2.0 * sigma_total_squared)
            -sigma_overlap_squared * b_squared / 2.0
            + 1j * sigma2_squared / sigma_total_squared * b_dot_K
        )
    )
    leading_correction = (
        sigma1_squared / sigma_total_squared * k3_dot_K
        - 1j * sigma_overlap_squared * k3_dot_b
    )

    ell1 = packet1.ell
    ell2 = packet2.ell
    m = abs(ell1)
    n = abs(ell2)

    same_sign_or_zero = ell1 == 0 or ell2 == 0 or (ell1 > 0) == (ell2 > 0)
    if same_sign_or_zero:
        sign = 1 if (ell1 > 0 or ell2 > 0 or (ell1 == 0 and ell2 == 0)) else -1
        K_e = _dot_e(K, sign)
        b_e = _dot_e(b, sign)
        k3_e = _dot_e(k3, sign)
        c1 = sigma1_squared / sigma_total_squared * K_e - 1j * sigma_overlap_squared * b_e
        c2 = sigma2_squared / sigma_total_squared * K_e + 1j * sigma_overlap_squared * b_e
        F = c1**m * c2**n
        derivative = np.zeros(np.broadcast(c1, c2).shape, dtype=COMPLEX_DTYPE)
        if m:
            derivative += m * c1 ** (m - 1) * c2**n
        if n:
            derivative -= n * c1**m * c2 ** (n - 1)
        source_correction = k3_e * derivative
    else:
        sign1 = 1 if ell1 > 0 else -1
        sign2 = 1 if ell2 > 0 else -1
        c1 = (
            sigma1_squared / sigma_total_squared * _dot_e(K, sign1)
            - 1j * sigma_overlap_squared * _dot_e(b, sign1)
        )
        c2 = (
            sigma2_squared / sigma_total_squared * _dot_e(K, sign2)
            + 1j * sigma_overlap_squared * _dot_e(b, sign2)
        )
        c12 = 2.0 * sigma_overlap_squared
        F = _mixed_polynomial(m, n, c1, c2, c12)
        source_correction = (
            m * _dot_e(k3, sign1) * _mixed_polynomial(m - 1, n, c1, c2, c12)
            - n * _dot_e(k3, sign2) * _mixed_polynomial(m, n - 1, c1, c2, c12)
        )

    result = G0 / k3_squared * (
        (1.0 + 2.0 * leading_correction / k3_squared) * F
        + 2.0 * sigma_overlap_squared * source_correction / k3_squared
    )
    result = np.asarray(result, dtype=COMPLEX_DTYPE)
    return result.item() if result.ndim == 0 else result


def _base_prefactor(
    packet1: VortexPacket,
    packet2: VortexPacket,
    norms,
    charge: float,
) -> complex:
    N1, N2 = _validate_norms(norms)
    if not np.isfinite(charge):
        raise ValueError("charge must be finite.")
    ell1 = abs(packet1.ell)
    ell2 = abs(packet2.ell)
    log_denominator = (
        ell1 * math.log(packet1.sigma_perp)
        + ell2 * math.log(packet2.sigma_perp)
        + 0.5 * (math.lgamma(ell1 + 1) + math.lgamma(ell2 + 1))
    )
    return COMPLEX_DTYPE(
        -1j
        * charge**2
        / (4.0 * PI**3)
        * N1
        * N2
        * math.exp(-log_denominator)
    )


def s_matrix(
    k3,
    k4,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    norms,
    impact=(0.0, 0.0),
    helicities=(0.5, -0.5, 0.5, -0.5),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    return_info: bool = False,
):
    """Compute the S-matrix in PDF Eq. (31), with T from Eqs. (46)/(56)."""
    k3, k4 = _broadcast_vectors(k3, k4, 3, ("k3", "k4"))
    if len(helicities) != 4 or any(value not in (-0.5, 0.5) for value in helicities):
        raise ValueError("helicities must contain four values equal to +/-0.5.")

    output_shape = k3.shape[:-1]
    if helicities[2] != helicities[0] or helicities[3] != helicities[1]:
        zero = np.zeros(output_shape, dtype=COMPLEX_DTYPE)
        zero = zero.item() if zero.ndim == 0 else zero
        return (zero, {"reason": "helicity delta is zero"}) if return_info else zero

    E3 = np.sqrt(mass**2 + np.sum(k3 * k3, axis=-1))
    E4 = np.sqrt(mass**2 + np.sum(k4 * k4, axis=-1))
    E01 = central_energy(packet1, mass)
    E02 = central_energy(packet2, mass)
    K = k3 + k4
    L = _longitudinal_from_energies(E3, E4, K[..., 2], packet1, packet2, mass)
    T = transverse_integral(
        k3[..., :2], K[..., :2], packet1, packet2, impact=impact
    )
    prefactor = _base_prefactor(packet1, packet2, norms, charge)
    result = prefactor * np.sqrt(E3 * E4 / (E01 * E02)) * L * T
    result = np.asarray(result, dtype=COMPLEX_DTYPE)
    result_out = result.item() if result.ndim == 0 else result

    if not return_info:
        return result_out
    k3_perp = np.sqrt(np.sum(k3[..., :2] ** 2, axis=-1))
    return result_out, {
        "longitudinal_factor": L,
        "transverse_integral": T,
        "base_prefactor": prefactor,
        "E3": E3,
        "E4": E4,
        "K": K,
        "transverse_expansion_ratio": (
            np.sqrt(abs(packet1.ell) + 1.0) * packet1.sigma_perp / k3_perp
        ),
    }
