"""Vectorized phase-space probabilities built from the PDF S-matrix."""

from __future__ import annotations

import numpy as np

from .config import (
    ELECTRON_MASS,
    PI,
    REAL_DTYPE,
    ProbabilityGrid,
)
from .numerics import independent_map, scalar_product
from .scattering import (
    _base_prefactor,
    _longitudinal_from_energies,
    transverse_integral,
)
from .states import VortexPacket, central_energy


def _inner_quadrature(grid: ProbabilityGrid):
    k3_perp, k3_perp_weights = grid.k3_perp.nodes_weights()
    k3_phi, k3_phi_weights = grid.k3_phi.nodes_weights()
    radius = k3_perp[:, None]
    k3_perp_vectors = np.stack(
        (
            radius * np.cos(k3_phi),
            radius * np.sin(k3_phi),
        ),
        axis=-1,
    ).reshape(-1, 2)
    transverse_weights = np.outer(
        k3_perp_weights * k3_perp,
        k3_phi_weights,
    ).ravel()
    k3_perp_squared = scalar_product(k3_perp_vectors, k3_perp_vectors)

    k3_z, k3_z_weights = grid.k3_z.nodes_weights()
    k4_z, k4_z_weights = grid.k4_z.nodes_weights()
    k3_z_squared = k3_z[:, None] ** 2
    k4_z_squared = k4_z[None, :] ** 2
    total_k_z = k3_z[:, None] + k4_z[None, :]
    longitudinal_weights = np.outer(k3_z_weights, k4_z_weights)
    return (
        k3_perp_vectors,
        transverse_weights,
        k3_perp_squared,
        k3_z_squared,
        k4_z_squared,
        total_k_z,
        longitudinal_weights,
    )


def _probability_prefactor(
    packet1: VortexPacket,
    packet2: VortexPacket,
) -> np.floating:
    """Return the constant remaining after the E3*E4 cancellation."""
    phase_space = 1.0 / (4.0 * (2.0 * PI) ** 6)
    return (
        phase_space
        * np.abs(_base_prefactor(packet1, packet2)) ** 2
        / (central_energy(packet1) * central_energy(packet2))
    )


def _probability_at_total_k(
    total_k_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact,
    inner_quadrature,
    prefactor: np.floating,
    k_z_moment: bool,
) -> np.floating | tuple[np.floating, np.floating]:
    """Return dP/d²K_perp, optionally with its K_z moment density."""
    total_k_perp = np.asarray(total_k_perp, dtype=REAL_DTYPE)
    if total_k_perp.shape != (2,):
        raise ValueError("total_k_perp must have shape (2,).")

    (
        k3_perp,
        transverse_weights,
        k3_perp_squared,
        k3_z_squared,
        k4_z_squared,
        total_k_z,
        longitudinal_weights,
    ) = inner_quadrature

    k4_perp = total_k_perp - k3_perp
    transverse = transverse_integral(
        k3_perp,
        total_k_perp,
        packet1,
        packet2,
        impact=impact,
    )

    k4_perp_squared = scalar_product(k4_perp, k4_perp)

    probability_integral = REAL_DTYPE(0.0)
    if k_z_moment:
        k_z_integral = REAL_DTYPE(0.0)
    for start in range(0, k3_perp.shape[0], grid.batch_size):
        stop = min(start + grid.batch_size, k3_perp.shape[0])
        energy_3 = np.sqrt(
            ELECTRON_MASS**2
            + k3_perp_squared[start:stop, None, None]
            + k3_z_squared
        )
        energy_4 = np.sqrt(
            ELECTRON_MASS**2
            + k4_perp_squared[start:stop, None, None]
            + k4_z_squared
        )
        longitudinal = _longitudinal_from_energies(
            energy_3,
            energy_4,
            total_k_z,
            packet1,
            packet2,
        )
        longitudinal_squared = longitudinal**2
        weighted_longitudinal = longitudinal_weights * longitudinal_squared
        longitudinal_integral = np.sum(weighted_longitudinal, axis=(1, 2))
        transverse_integrand = (
            transverse_weights[start:stop] * np.abs(transverse[start:stop]) ** 2
        )
        probability_integral += np.sum(
            transverse_integrand * longitudinal_integral
        )
        if k_z_moment:
            longitudinal_k_z_integral = np.sum(
                total_k_z * weighted_longitudinal,
                axis=(1, 2),
            )
            k_z_integral += np.sum(
                transverse_integrand * longitudinal_k_z_integral
            )

    probability = prefactor * probability_integral
    if k_z_moment:
        return probability, prefactor * k_z_integral
    return probability


def differential_probability(
    total_k_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
) -> np.floating:
    """Compute ``dP/d²K_perp`` on the configured finite domain."""
    inner_quadrature = _inner_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)
    return _probability_at_total_k(
        total_k_perp,
        packet1,
        packet2,
        grid,
        impact=impact,
        inner_quadrature=inner_quadrature,
        prefactor=prefactor,
        k_z_moment=False,
    )


def differential_probability_grid(
    total_k_x_values,
    total_k_y_values,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
    workers: int = 1,
) -> np.ndarray:
    """Evaluate ``dP/d²K_perp`` on a Cartesian grid."""
    total_k_x = np.asarray(total_k_x_values, dtype=REAL_DTYPE)
    total_k_y = np.asarray(total_k_y_values, dtype=REAL_DTYPE)
    if total_k_x.ndim != 1 or total_k_y.ndim != 1:
        raise ValueError(
            "total_k_x_values and total_k_y_values must be one-dimensional."
        )

    points = [(x, y) for y in total_k_y for x in total_k_x]
    inner_quadrature = _inner_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp):
        return _probability_at_total_k(
            total_k_perp,
            packet1,
            packet2,
            grid,
            impact=impact,
            inner_quadrature=inner_quadrature,
            prefactor=prefactor,
            k_z_moment=False,
        )

    values = independent_map(evaluate, points, workers)
    return np.asarray(values, dtype=REAL_DTYPE).reshape(
        total_k_y.size, total_k_x.size
    )


def _total_k_quadrature(grid: ProbabilityGrid):
    total_k_perp, radial_weights = grid.total_k_perp.nodes_weights()
    total_k_phi, angular_weights = grid.total_k_phi.nodes_weights()
    radial_weights = radial_weights * total_k_perp
    nonzero = radial_weights != 0.0
    total_k_perp = total_k_perp[nonzero]
    radial_weights = radial_weights[nonzero]

    radius = total_k_perp[:, None]
    total_k_x = radius * np.cos(total_k_phi)
    total_k_y = radius * np.sin(total_k_phi)
    points = np.column_stack((total_k_x.ravel(), total_k_y.ravel()))
    weights = np.outer(radial_weights, angular_weights).ravel()
    return points, weights


def total_probability(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
    workers: int = 1,
) -> np.floating:
    """Integrate the probability over the configured total-momentum domain."""
    points, weights = _total_k_quadrature(grid)

    inner_quadrature = _inner_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp_value):
        return _probability_at_total_k(
            total_k_perp_value,
            packet1,
            packet2,
            grid,
            impact=impact,
            inner_quadrature=inner_quadrature,
            prefactor=prefactor,
            k_z_moment=False,
        )

    probability_density = np.asarray(
        independent_map(evaluate, points, workers),
        dtype=REAL_DTYPE,
    )
    return np.sum(weights * probability_density)


def mean_total_momentum(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
    workers: int = 1,
) -> np.ndarray:
    """Return the probability-weighted ``(<Kx>, <Ky>, <Kz>)``."""
    points, weights = _total_k_quadrature(grid)
    inner_quadrature = _inner_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp_value):
        return _probability_at_total_k(
            total_k_perp_value,
            packet1,
            packet2,
            grid,
            impact=impact,
            inner_quadrature=inner_quadrature,
            prefactor=prefactor,
            k_z_moment=True,
        )

    values = np.asarray(
        independent_map(evaluate, points, workers),
        dtype=REAL_DTYPE,
    )
    probability_density = values[:, 0]
    k_z_moment_density = values[:, 1]
    weighted_probability = weights * probability_density
    probability = np.sum(weighted_probability)
    if not np.isfinite(probability) or probability <= 0.0:
        raise FloatingPointError("Integrated probability is not positive and finite.")

    momentum_integral = np.asarray(
        (
            np.sum(points[:, 0] * weighted_probability),
            np.sum(points[:, 1] * weighted_probability),
            np.sum(weights * k_z_moment_density),
        ),
        dtype=REAL_DTYPE,
    )
    return momentum_integral / probability
