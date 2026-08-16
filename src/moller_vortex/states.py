"""Normalized vortex states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .config import (
    COMPLEX_DTYPE,
    ELECTRON_MASS,
    PI,
    NormalizationGrid,
)
from .numerics import scalar_product, vector


@dataclass(frozen=True)
class VortexPacket:
    """Parameters of a normalized reference-frame vortex state."""

    ell: int
    sigma_perp: float
    sigma_parallel: float
    sigma_energy: float
    k0_z: float
    normalization_grid: NormalizationGrid

    @cached_property
    def norm(self) -> np.floating:
        """Return the normalization constant, computing it on first access."""
        return _normalization(self)


def energy(momentum):
    """Return the electron on-shell energy for momenta of shape ``(..., 3)``."""
    momentum = vector(momentum, 3, "momentum")
    return np.sqrt(ELECTRON_MASS**2 + scalar_product(momentum, momentum))


def central_energy(packet: VortexPacket) -> np.floating:
    """Return the packet's central on-shell energy."""
    return np.sqrt(ELECTRON_MASS**2 + packet.k0_z**2)


def effective_sigma(packet: VortexPacket) -> np.floating:
    """Return the effective longitudinal width from PDF Eq. (27)."""
    velocity = packet.k0_z / central_energy(packet)
    inverse_variance = (
        1.0 / packet.sigma_parallel**2
        + velocity**2 / packet.sigma_energy**2
    )
    return inverse_variance ** (-0.5)


def _vortex_factor(k_perp, ell: int):
    """Return ``k_perp**|ell| exp(i ell phi)`` in Cartesian coordinates."""
    if ell == 0:
        return np.ones(k_perp.shape[:-1], dtype=COMPLEX_DTYPE)
    sign = 1 if ell > 0 else -1
    return np.asarray(
        (k_perp[..., 0] + 1j * sign * k_perp[..., 1]) ** abs(ell),
        dtype=COMPLEX_DTYPE,
    )


def wave_packet(
    momentum,
    packet: VortexPacket,
    *,
    impact=(0.0, 0.0),
):
    """Evaluate the reference-frame state in PDF Eqs. (4) and (16)."""
    momentum = vector(momentum, 3, "momentum")
    impact = np.asarray(impact, dtype=momentum.dtype)
    if impact.shape != (2,):
        raise ValueError("impact must have shape (2,).")

    k_perp = momentum[..., :2]
    k_z = momentum[..., 2]
    k_perp_squared = scalar_product(k_perp, k_perp)
    energy_value = np.sqrt(ELECTRON_MASS**2 + k_perp_squared + k_z**2)
    ell_abs = abs(packet.ell)

    exponent = (
        -(energy_value - central_energy(packet)) ** 2
        / (2.0 * packet.sigma_energy**2)
        - k_perp_squared / (2.0 * packet.sigma_perp**2)
        - (k_z - packet.k0_z) ** 2 / (2.0 * packet.sigma_parallel**2)
        + 1j * scalar_product(impact, k_perp)
    )
    denominator = np.sqrt(math.factorial(ell_abs)) * packet.sigma_perp**ell_abs
    return np.asarray(
        packet.norm * _vortex_factor(k_perp, packet.ell) * np.exp(exponent)
        / denominator,
        dtype=COMPLEX_DTYPE,
    )


def _normalization(packet: VortexPacket) -> np.floating:
    """Compute the normalization using the expanded formula from the paper."""
    k_perp_nodes, k_perp_weights = (
        packet.normalization_grid.k_perp.nodes_weights()
    )
    k_z_nodes, k_z_weights = packet.normalization_grid.k_z.nodes_weights()
    k_perp = k_perp_nodes[:, None]
    k_z = k_z_nodes[None, :]
    energy_value = np.sqrt(ELECTRON_MASS**2 + k_perp**2 + k_z**2)
    central_energy_value = central_energy(packet)
    ell_abs = abs(packet.ell)

    outside_exponent = (
        (ELECTRON_MASS**2 + central_energy_value**2)
        / packet.sigma_energy**2
        + packet.k0_z**2 / packet.sigma_parallel**2
    )
    integral_exponent = (
        -(1.0 / packet.sigma_energy**2 + 1.0 / packet.sigma_perp**2)
        * k_perp**2
        -(1.0 / packet.sigma_energy**2 + 1.0 / packet.sigma_parallel**2)
        * k_z**2
        + 2.0
        * central_energy_value
        * energy_value
        / packet.sigma_energy**2
        + 2.0 * packet.k0_z * k_z / packet.sigma_parallel**2
    )

    # exp(outside_exponent) is moved under the integral to avoid inf / inf.
    integrand = (
        k_perp ** (2 * ell_abs + 1)
        / (2.0 * energy_value)
        * np.exp(integral_exponent - outside_exponent)
    )
    integral_over_k_z = np.sum(k_z_weights * integrand, axis=1)
    integral = np.sum(k_perp_weights * integral_over_k_z)
    norm_squared = (
        4.0
        * PI**2
        * math.factorial(ell_abs)
        * packet.sigma_perp ** (2 * ell_abs)
        / integral
    )
    return np.sqrt(norm_squared)
