"""Lorentz-covariant vortex states and their two-dimensional normalization."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from .config import (
    COMPLEX_DTYPE,
    ELECTRON_MASS,
    NormalizationGrid,
    PI,
    REAL_DTYPE,
    VortexPacket,
)


def energy(momentum, mass: float = ELECTRON_MASS):
    """Return ``sqrt(mass**2 + |momentum|**2)`` for vectors ending in 3."""
    momentum = np.asarray(momentum, dtype=REAL_DTYPE)
    if momentum.shape == () or momentum.shape[-1] != 3:
        raise ValueError("Momentum must have shape (..., 3).")
    return np.sqrt(mass**2 + np.sum(momentum * momentum, axis=-1))


def central_energy(packet: VortexPacket, mass: float = ELECTRON_MASS) -> float:
    """Return the packet central energy."""
    return float(np.sqrt(mass**2 + packet.k0_z**2))


def effective_sigma(packet: VortexPacket, mass: float = ELECTRON_MASS) -> float:
    """Return the effective longitudinal width from Eq. (27)."""
    velocity = packet.k0_z / central_energy(packet, mass)
    inverse_variance = (
        1.0 / packet.sigma_parallel**2
        + velocity**2 / packet.sigma_energy**2
    )
    return float(1.0 / np.sqrt(inverse_variance))


def _vortex_power(kx, ky, ell: int):
    """Return ``k_perp**|ell| exp(i ell phi)`` without computing phi."""
    if ell == 0:
        return np.ones(np.broadcast(kx, ky).shape, dtype=COMPLEX_DTYPE)
    sign = 1.0 if ell > 0 else -1.0
    return np.asarray((kx + 1j * sign * ky) ** abs(ell), dtype=COMPLEX_DTYPE)


def wave_packet(
    momentum,
    packet: VortexPacket,
    *,
    norm: float = 1.0,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
):
    """Evaluate the reference-frame state in Eq. (4).

    ``impact`` adds the phase ``exp(i b_perp . k_perp)`` used for packet 2.
    Arrays of momenta with shape ``(..., 3)`` are evaluated at once.
    """
    k = np.asarray(momentum, dtype=REAL_DTYPE)
    if k.shape == () or k.shape[-1] != 3:
        raise ValueError("momentum must have shape (..., 3).")
    b = np.asarray(impact, dtype=REAL_DTYPE)
    if b.shape != (2,):
        raise ValueError("impact must have shape (2,).")
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("norm must be positive and finite.")

    ell_abs = abs(packet.ell)
    exponent = (
        -(energy(k, mass) - central_energy(packet, mass)) ** 2
        / (2.0 * packet.sigma_energy**2)
        -(k[..., 0] ** 2 + k[..., 1] ** 2)
        / (2.0 * packet.sigma_perp**2)
        -(k[..., 2] - packet.k0_z) ** 2
        / (2.0 * packet.sigma_parallel**2)
        + 1j * (b[0] * k[..., 0] + b[1] * k[..., 1])
    )
    scale = norm / (
        np.sqrt(math.factorial(ell_abs)) * packet.sigma_perp**ell_abs
    )
    return np.asarray(
        scale * _vortex_power(k[..., 0], k[..., 1], packet.ell) * np.exp(exponent),
        dtype=COMPLEX_DTYPE,
    )


def _normalization_density(k_perp, k_z, packet: VortexPacket, mass: float):
    """Return the azimuth-integrated norm density without N squared."""
    k_perp = np.asarray(k_perp, dtype=REAL_DTYPE)
    k_z = np.asarray(k_z, dtype=REAL_DTYPE)
    ell_abs = abs(packet.ell)
    E = np.sqrt(mass**2 + k_perp**2 + k_z**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_power = (
            2.0 * ell_abs * np.log(k_perp / packet.sigma_perp)
            - gammaln(ell_abs + 1)
        )
    if ell_abs == 0:
        log_power = np.zeros(np.broadcast(k_perp, k_z).shape, dtype=REAL_DTYPE)

    log_density = (
        log_power
        -(E - central_energy(packet, mass)) ** 2 / packet.sigma_energy**2
        -k_perp**2 / packet.sigma_perp**2
        -(k_z - packet.k0_z) ** 2 / packet.sigma_parallel**2
    )
    density = np.where(np.isfinite(log_density), np.exp(log_density), 0.0)
    return k_perp * density / (8.0 * PI**2 * E)


def normalization(
    packet: VortexPacket,
    grid: NormalizationGrid,
    *,
    mass: float = ELECTRON_MASS,
    method: str = "quadrature",
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-9,
    limit: int = 150,
    return_info: bool = False,
):
    """Compute the state normalization with a finite two-dimensional integral.

    ``method="quadrature"`` uses the configured tensor-product rules and point
    counts.  ``method="adaptive"`` uses nested adaptive Gauss-Kronrod
    integration over exactly the same finite intervals.
    """
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("mass must be positive and finite.")
    if method not in {"quadrature", "adaptive"}:
        raise ValueError("method must be 'quadrature' or 'adaptive'.")

    evaluations = 0
    estimated_error = None

    if method == "quadrature":
        k_perp, w_perp = grid.k_perp.nodes_weights()
        k_z, w_z = grid.k_z.nodes_weights()
        density = _normalization_density(
            k_perp[:, None], k_z[None, :], packet, mass
        )
        integral = np.sum(w_perp[:, None] * w_z[None, :] * density)
        evaluations = k_perp.size * k_z.size
    else:
        inner_errors: list[float] = []

        def radial_integrand(k_perp: float) -> float:
            nonlocal evaluations

            def z_integrand(k_z: float) -> float:
                nonlocal evaluations
                evaluations += 1
                return float(_normalization_density(k_perp, k_z, packet, mass))

            value, error = quad(
                z_integrand,
                grid.k_z.start,
                grid.k_z.stop,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )
            inner_errors.append(error)
            return value

        integral, outer_error = quad(
            radial_integrand,
            grid.k_perp.start,
            grid.k_perp.stop,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        estimated_error = float(outer_error + max(inner_errors, default=0.0))

    integral = float(integral)
    if not np.isfinite(integral) or integral <= 0.0:
        raise FloatingPointError(
            "Normalization integral is not positive and finite; check the ranges."
        )
    norm = float(integral**-0.5)

    if not return_info:
        return norm
    return norm, {
        "integral_without_norm": integral,
        "estimated_error": estimated_error,
        "evaluations": evaluations,
        "method": method,
        "k_perp_range": (grid.k_perp.start, grid.k_perp.stop),
        "k_z_range": (grid.k_z.start, grid.k_z.stop),
    }
