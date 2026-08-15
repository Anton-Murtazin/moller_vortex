"""Vortex state, two-dimensional normalization, and approximation checks."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    COMPLEX_DTYPE,
    ELECTRON_MASS,
    NormalizationGrid,
    PI,
    REAL_DTYPE,
    VortexPacket,
)


def energy(momentum, mass: float = ELECTRON_MASS):
    """Return the on-shell energy for arrays with shape ``(..., 3)``."""
    momentum = np.asarray(momentum, dtype=REAL_DTYPE)
    if momentum.ndim == 0 or momentum.shape[-1] != 3:
        raise ValueError("momentum must have shape (..., 3).")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be positive and finite.")
    return np.sqrt(mass**2 + np.sum(momentum * momentum, axis=-1))


def central_energy(packet: VortexPacket, mass: float = ELECTRON_MASS) -> float:
    """Return ``sqrt(mass**2 + packet.k0_z**2)``."""
    return float(np.sqrt(mass**2 + packet.k0_z**2))


def effective_sigma(packet: VortexPacket, mass: float = ELECTRON_MASS) -> float:
    """Return the effective longitudinal width Sigma from PDF Eq. (27)."""
    velocity = packet.k0_z / central_energy(packet, mass)
    inverse_variance = (
        1.0 / packet.sigma_parallel**2
        + velocity**2 / packet.sigma_energy**2
    )
    return float(inverse_variance**-0.5)


def _vortex_power(kx, ky, ell: int):
    """Return ``k_perp**|ell| exp(i ell phi)`` without angle evaluation."""
    if ell == 0:
        return np.ones(np.broadcast(kx, ky).shape, dtype=COMPLEX_DTYPE)
    sign = 1 if ell > 0 else -1
    return np.asarray((kx + 1j * sign * ky) ** abs(ell), dtype=COMPLEX_DTYPE)


def wave_packet(
    momentum,
    packet: VortexPacket,
    *,
    norm: float = 1.0,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
):
    """Evaluate the reference-frame state in PDF Eqs. (4) and (16).

    ``impact`` adds ``exp(i b_perp . k_perp)`` and is normally nonzero only
    for the second incident packet.
    """
    k = np.asarray(momentum, dtype=REAL_DTYPE)
    if k.ndim == 0 or k.shape[-1] != 3:
        raise ValueError("momentum must have shape (..., 3).")
    b = np.asarray(impact, dtype=REAL_DTYPE)
    if b.shape != (2,):
        raise ValueError("impact must have shape (2,).")
    if not np.isfinite(norm) or norm <= 0.0:
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
    denominator = math.sqrt(math.factorial(ell_abs)) * packet.sigma_perp**ell_abs
    values = norm * _vortex_power(k[..., 0], k[..., 1], packet.ell)
    return np.asarray(values * np.exp(exponent) / denominator, dtype=COMPLEX_DTYPE)


def _normalization_log_density(k_perp, k_z, packet: VortexPacket, mass: float):
    """Log of the azimuth-integrated norm density without ``N**2``."""
    k_perp = np.asarray(k_perp, dtype=REAL_DTYPE)
    k_z = np.asarray(k_z, dtype=REAL_DTYPE)
    ell_abs = abs(packet.ell)
    E = np.sqrt(mass**2 + k_perp**2 + k_z**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_measure_and_power = (
            np.log(k_perp / E)
            + 2.0 * ell_abs * np.log(k_perp / packet.sigma_perp)
            - math.lgamma(ell_abs + 1)
        )
    exponent = (
        -(E - central_energy(packet, mass)) ** 2 / packet.sigma_energy**2
        -k_perp**2 / packet.sigma_perp**2
        -(k_z - packet.k0_z) ** 2 / packet.sigma_parallel**2
    )
    return log_measure_and_power + exponent - np.log(8.0 * PI**2)


def normalization(
    packet: VortexPacket,
    grid: NormalizationGrid,
    *,
    mass: float = ELECTRON_MASS,
    return_info: bool = False,
):
    """Compute the finite two-dimensional quadrature in PDF Eqs. (17)-(18).

    The interval and point count are exactly those in ``grid``. The radial
    nodes are processed in batches and the integral is accumulated in a
    scaled logarithmic form to avoid avoidable overflow and underflow.
    """
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be positive and finite.")

    k_perp, w_perp = grid.k_perp.nodes_weights()
    k_z, w_z = grid.k_z.nodes_weights()

    maximum = -np.inf
    for start in range(0, k_perp.size, grid.batch_size):
        stop = min(start + grid.batch_size, k_perp.size)
        log_density = _normalization_log_density(
            k_perp[start:stop, None], k_z[None, :], packet, mass
        )
        finite = log_density[np.isfinite(log_density)]
        if finite.size:
            maximum = max(maximum, float(np.max(finite)))
    if not np.isfinite(maximum):
        raise FloatingPointError("Normalization density is zero on the chosen grid.")

    scaled_integral = REAL_DTYPE(0.0)
    for start in range(0, k_perp.size, grid.batch_size):
        stop = min(start + grid.batch_size, k_perp.size)
        log_density = _normalization_log_density(
            k_perp[start:stop, None], k_z[None, :], packet, mass
        )
        weights = w_perp[start:stop, None] * w_z[None, :]
        scaled_integral += np.sum(weights * np.exp(log_density - maximum))

    if not np.isfinite(scaled_integral) or scaled_integral <= 0.0:
        raise FloatingPointError("Normalization integral is not positive and finite.")

    log_integral = float(np.log(scaled_integral) + maximum)
    norm = float(np.exp(-0.5 * log_integral))
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Normalization constant is not positive and finite.")

    if not return_info:
        return norm
    return norm, {
        "log_integral_without_norm": log_integral,
        "integral_without_norm": float(np.exp(log_integral)),
        "evaluations": int(k_perp.size * k_z.size),
        "k_perp_range": (grid.k_perp.start, grid.k_perp.stop),
        "k_z_range": (grid.k_z.start, grid.k_z.stop),
        "points": (grid.k_perp.points, grid.k_z.points),
        "rules": (grid.k_perp.rule, grid.k_z.rule),
    }


def approximation_parameters(
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    mass: float = ELECTRON_MASS,
) -> dict:
    """Return the dimensionless self-consistency parameters in PDF Sec. 3.

    The PDF requires these values to be much smaller than one. No arbitrary
    pass/fail threshold is imposed here.
    """
    packets = (packet1, packet2)
    energies = np.asarray([central_energy(p, mass) for p in packets])
    velocities = np.asarray([p.k0_z / E for p, E in zip(packets, energies)])
    sigmas = np.asarray([effective_sigma(p, mass) for p in packets])
    delta_v = abs(velocities[0] - velocities[1])
    if delta_v <= np.finfo(REAL_DTYPE).eps:
        raise ValueError("The impulse approximation requires different packet velocities.")

    sigma_overlap = float((sigmas[0] ** -2 + sigmas[1] ** -2) ** -0.5)
    overlap_time = float(1.0 / (delta_v * sigma_overlap))
    packet_reports = []
    beta_typical = []
    for packet, E, velocity, sigma in zip(packets, energies, velocities, sigmas):
        ell_factor = abs(packet.ell) + 1.0
        transverse_spreading_time = E / packet.sigma_perp**2
        longitudinal_spreading_time = E**3 / (mass**2 * sigma**2)
        beta = (
            ell_factor * packet.sigma_perp**2 / (2.0 * E)
            + mass**2 * sigma**2 / (4.0 * E**3)
        )
        beta_typical.append(beta)
        packet_reports.append(
            {
                "central_energy": float(E),
                "velocity": float(velocity),
                "effective_sigma": float(sigma),
                "energy_expansion_longitudinal": float(sigma / (np.sqrt(2.0) * E)),
                "energy_expansion_transverse": float(
                    np.sqrt(ell_factor) * packet.sigma_perp / E
                ),
                "mass_over_energy": float(mass / E),
                "smooth_prefactor": float(abs(velocity) * sigma / (np.sqrt(2.0) * E)),
                "transverse_spreading": float(overlap_time / transverse_spreading_time),
                "longitudinal_spreading": float(overlap_time / longitudinal_spreading_time),
                "transverse_phase": float(
                    ell_factor * packet.sigma_perp**2 * overlap_time / (2.0 * E)
                ),
                "longitudinal_phase": float(
                    mass**2 * sigma**2 * overlap_time / (4.0 * E**3)
                ),
            }
        )

    return {
        "packet1": packet_reports[0],
        "packet2": packet_reports[1],
        "sigma_overlap": sigma_overlap,
        "overlap_time": overlap_time,
        "impulse_phase_total": float(overlap_time * sum(beta_typical)),
    }
