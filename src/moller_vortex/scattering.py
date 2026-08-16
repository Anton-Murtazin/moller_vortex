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
)
from .numerics import (
    scalar_product,
    transverse_projection,
    vector,
)
from .states import VortexPacket, central_energy


def _mixed_polynomial(m: int, n: int, c1, c2, c12: float):
    """Return F_mn from PDF Eq. (53)."""
    result = c1**m * c2**n
    for r in range(1, min(m, n) + 1):
        coefficient = math.comb(m, r) * math.comb(n, r) * math.factorial(r)
        result += (
            coefficient
            * (-c12) ** r
            * c1 ** (m - r)
            * c2 ** (n - r)
        )
    return np.asarray(result, dtype=COMPLEX_DTYPE)


def longitudinal_factor(
    momentum_3,
    momentum_4,
    packet1: VortexPacket,
    packet2: VortexPacket,
):
    """Evaluate the separated longitudinal factor L in PDF Eq. (28)."""
    momentum_3 = vector(momentum_3, 3, "momentum_3")
    momentum_4 = vector(momentum_4, 3, "momentum_4")
    k3_z = momentum_3[..., 2]
    k4_z = momentum_4[..., 2]
    return _longitudinal_from_energies(
        np.sqrt(ELECTRON_MASS**2 + scalar_product(momentum_3, momentum_3)),
        np.sqrt(ELECTRON_MASS**2 + scalar_product(momentum_4, momentum_4)),
        k3_z + k4_z,
        packet1,
        packet2,
    )


def _longitudinal_from_energies(
    energy_3,
    energy_4,
    total_k_z,
    packet1: VortexPacket,
    packet2: VortexPacket,
):
    """Evaluate the longitudinal factor from precomputed final energies."""
    central_energy_1 = central_energy(packet1)
    central_energy_2 = central_energy(packet2)
    velocity_1 = packet1.k0_z / central_energy_1
    velocity_2 = packet2.k0_z / central_energy_2
    delta_velocity = velocity_1 - velocity_2
    if np.abs(delta_velocity) <= np.finfo(REAL_DTYPE).eps:
        raise ValueError("The two packet velocities must differ.")

    delta_k_z = total_k_z - packet1.k0_z - packet2.k0_z
    delta_energy = energy_3 + energy_4 - central_energy_1 - central_energy_2
    sigma_1 = (
        1.0 / packet1.sigma_parallel**2
        + velocity_1**2 / packet1.sigma_energy**2
    ) ** (-0.5)
    sigma_2 = (
        1.0 / packet2.sigma_parallel**2
        + velocity_2**2 / packet2.sigma_energy**2
    ) ** (-0.5)
    exponent = (
        -(delta_energy - velocity_2 * delta_k_z) ** 2
        / (2.0 * sigma_1**2 * delta_velocity**2)
        -(delta_energy - velocity_1 * delta_k_z) ** 2
        / (2.0 * sigma_2**2 * delta_velocity**2)
    )
    return np.asarray(
        2.0 * PI / np.abs(delta_velocity) * np.exp(exponent),
        dtype=REAL_DTYPE,
    )


def transverse_integral(
    k3_perp,
    total_k_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    impact=(0.0, 0.0),
):
    """Evaluate the closed transverse integral in PDF Eqs. (46) and (56)."""
    k3_perp = vector(k3_perp, 2, "k3_perp")
    total_k_perp = vector(total_k_perp, 2, "total_k_perp")
    impact = np.asarray(impact, dtype=REAL_DTYPE)
    if impact.shape != (2,):
        raise ValueError("impact must have shape (2,).")

    k3_perp_squared = scalar_product(k3_perp, k3_perp)
    if np.any(k3_perp_squared <= 0.0):
        raise ZeroDivisionError("PDF Eq. (32) requires nonzero k3_perp.")

    sigma_1_squared = packet1.sigma_perp**2
    sigma_2_squared = packet2.sigma_perp**2
    sigma_total_squared = sigma_1_squared + sigma_2_squared
    sigma_overlap_squared = (
        sigma_1_squared * sigma_2_squared / sigma_total_squared
    )

    total_k_perp_squared = scalar_product(total_k_perp, total_k_perp)
    impact_squared = scalar_product(impact, impact)
    impact_dot_total_k = scalar_product(total_k_perp, impact)
    k3_dot_total_k = scalar_product(k3_perp, total_k_perp)
    k3_dot_impact = scalar_product(k3_perp, impact)

    gaussian_integral = (
        2.0
        * PI
        * sigma_overlap_squared
        * np.exp(
            -total_k_perp_squared / (2.0 * sigma_total_squared)
            -sigma_overlap_squared * impact_squared / 2.0
            + 1j
            * sigma_2_squared
            / sigma_total_squared
            * impact_dot_total_k
        )
    )
    leading_correction = (
        sigma_1_squared / sigma_total_squared * k3_dot_total_k
        - 1j * sigma_overlap_squared * k3_dot_impact
    )

    ell_1 = packet1.ell
    ell_2 = packet2.ell
    m = abs(ell_1)
    n = abs(ell_2)

    same_sign = ell_1 == 0 or ell_2 == 0 or (ell_1 > 0) == (ell_2 > 0)
    if same_sign:
        sign = 1 if ell_1 > 0 or ell_2 > 0 else -1
        if ell_1 == 0 and ell_2 == 0:
            sign = 1

        total_k_projection = transverse_projection(total_k_perp, sign)
        impact_projection = transverse_projection(impact, sign)
        k3_projection = transverse_projection(k3_perp, sign)
        c1 = (
            sigma_1_squared / sigma_total_squared * total_k_projection
            - 1j * sigma_overlap_squared * impact_projection
        )
        c2 = (
            sigma_2_squared / sigma_total_squared * total_k_projection
            + 1j * sigma_overlap_squared * impact_projection
        )
        polynomial = c1**m * c2**n
        derivative_c1 = m * c1 ** (m - 1) * c2**n if m else 0.0
        derivative_c2 = n * c1**m * c2 ** (n - 1) if n else 0.0
        source_correction = k3_projection * (derivative_c1 - derivative_c2)
    else:
        sign_1 = 1 if ell_1 > 0 else -1
        sign_2 = 1 if ell_2 > 0 else -1
        c1 = (
            sigma_1_squared
            / sigma_total_squared
            * transverse_projection(total_k_perp, sign_1)
            - 1j * sigma_overlap_squared * transverse_projection(impact, sign_1)
        )
        c2 = (
            sigma_2_squared
            / sigma_total_squared
            * transverse_projection(total_k_perp, sign_2)
            + 1j * sigma_overlap_squared * transverse_projection(impact, sign_2)
        )
        c12 = 2.0 * sigma_overlap_squared
        polynomial = _mixed_polynomial(m, n, c1, c2, c12)
        source_correction = (
            m
            * transverse_projection(k3_perp, sign_1)
            * _mixed_polynomial(m - 1, n, c1, c2, c12)
            - n
            * transverse_projection(k3_perp, sign_2)
            * _mixed_polynomial(m, n - 1, c1, c2, c12)
        )

    result = gaussian_integral / k3_perp_squared * (
        (1.0 + 2.0 * leading_correction / k3_perp_squared) * polynomial
        + 2.0
        * sigma_overlap_squared
        * source_correction
        / k3_perp_squared
    )
    result = np.asarray(result, dtype=COMPLEX_DTYPE)
    return result[()] if result.ndim == 0 else result


def _base_prefactor(
    packet1: VortexPacket,
    packet2: VortexPacket,
) -> np.complexfloating:
    """Return the momentum-independent S-matrix prefactor."""
    ell_1 = abs(packet1.ell)
    ell_2 = abs(packet2.ell)
    denominator = (
        packet1.sigma_perp**ell_1
        * packet2.sigma_perp**ell_2
        * np.sqrt(math.factorial(ell_1) * math.factorial(ell_2))
    )
    return COMPLEX_DTYPE(
        -1j
        * ELECTRON_CHARGE**2
        / (4.0 * PI**3)
        * packet1.norm
        * packet2.norm
        / denominator
    )


def s_matrix(
    momentum_3,
    momentum_4,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    impact=(0.0, 0.0),
    helicities=(0.5, -0.5, 0.5, -0.5),
):
    """Compute the S-matrix in PDF Eq. (31)."""
    momentum_3 = vector(momentum_3, 3, "momentum_3")
    momentum_4 = vector(momentum_4, 3, "momentum_4")
    initial_1, initial_2, final_3, final_4 = helicities
    if final_3 != initial_1 or final_4 != initial_2:
        result = np.zeros_like(
            momentum_3[..., 0] + momentum_4[..., 0],
            dtype=COMPLEX_DTYPE,
        )
        return result[()] if result.ndim == 0 else result

    energy_3 = np.sqrt(
        ELECTRON_MASS**2 + scalar_product(momentum_3, momentum_3)
    )
    energy_4 = np.sqrt(
        ELECTRON_MASS**2 + scalar_product(momentum_4, momentum_4)
    )
    total_momentum = momentum_3 + momentum_4
    k3_perp = momentum_3[..., :2]
    total_k_perp = total_momentum[..., :2]
    total_k_z = total_momentum[..., 2]
    longitudinal = _longitudinal_from_energies(
        energy_3,
        energy_4,
        total_k_z,
        packet1,
        packet2,
    )
    transverse = transverse_integral(
        k3_perp,
        total_k_perp,
        packet1,
        packet2,
        impact=impact,
    )
    result = (
        _base_prefactor(packet1, packet2)
        * np.sqrt(
            energy_3
            * energy_4
            / (central_energy(packet1) * central_energy(packet2))
        )
        * longitudinal
        * transverse
    )
    result = np.asarray(result, dtype=COMPLEX_DTYPE)
    return result[()] if result.ndim == 0 else result
