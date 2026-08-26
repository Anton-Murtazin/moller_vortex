"""Sequential phase-space probabilities built from the impulse S-matrix."""

from __future__ import annotations

import numpy as np

from .config import (
    PI,
    REAL_DTYPE,
    ProbabilityGrid,
)
from .numerics import sequential_map
from .scattering import (
    _base_prefactor,
    _integrated_longitudinal_squared,
    transverse_integral,
)
from .states import VortexPacket, central_energy


def _transverse_quadrature(grid: ProbabilityGrid):
    """Return the nodes and weights of the remaining transverse integral."""
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
    return k3_perp_vectors, transverse_weights


def _probability_prefactor(
    packet1: VortexPacket,
    packet2: VortexPacket,
) -> np.floating:
    """Return the constant after phase-space and longitudinal integration."""
    phase_space = 1.0 / (4.0 * (2.0 * PI) ** 6)
    return (
        phase_space
        * np.abs(_base_prefactor(packet1, packet2)) ** 2
        / (central_energy(packet1) * central_energy(packet2))
        * _integrated_longitudinal_squared(packet1, packet2)
    )


def _probability_at_total_k(
    total_k_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    impact,
    transverse_quadrature,
    prefactor: np.floating,
) -> np.floating:
    """Return dP/d^2K_perp after analytic longitudinal integration."""
    total_k_perp = np.asarray(total_k_perp, dtype=REAL_DTYPE)
    if total_k_perp.shape != (2,):
        raise ValueError("total_k_perp must have shape (2,).")

    k3_perp, transverse_weights = transverse_quadrature
    transverse = transverse_integral(
        k3_perp,
        total_k_perp,
        packet1,
        packet2,
        impact=impact,
    )

    transverse_integral_squared = np.sum(
        transverse_weights * np.abs(transverse) ** 2
    )
    return prefactor * transverse_integral_squared


def differential_probability(
    total_k_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
) -> np.floating:
    """Compute ``dP/d^2K_perp`` in MeV^-2 on the configured finite domain."""
    transverse_quadrature = _transverse_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)
    return _probability_at_total_k(
        total_k_perp,
        packet1,
        packet2,
        impact=impact,
        transverse_quadrature=transverse_quadrature,
        prefactor=prefactor,
    )


def differential_probability_grid(
    total_k_x_values,
    total_k_y_values,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
    progress: bool = False,
) -> np.ndarray:
    """Evaluate ``dP/d^2K_perp`` in MeV^-2 on a Cartesian grid."""
    total_k_x = np.asarray(total_k_x_values, dtype=REAL_DTYPE)
    total_k_y = np.asarray(total_k_y_values, dtype=REAL_DTYPE)
    if total_k_x.ndim != 1 or total_k_y.ndim != 1:
        raise ValueError(
            "total_k_x_values and total_k_y_values must be one-dimensional."
        )

    points = [(x, y) for y in total_k_y for x in total_k_x]
    transverse_quadrature = _transverse_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp):
        return _probability_at_total_k(
            total_k_perp,
            packet1,
            packet2,
            impact=impact,
            transverse_quadrature=transverse_quadrature,
            prefactor=prefactor,
        )

    values = sequential_map(
        evaluate,
        points,
        progress=progress,
        description="Differential probability",
    )
    return np.asarray(values, dtype=REAL_DTYPE).reshape(
        total_k_y.size, total_k_x.size
    )


def _total_k_quadrature(grid: ProbabilityGrid):
    if grid.total_k_perp is None or grid.total_k_phi is None:
        raise ValueError(
            "The total_k_perp and total_k_phi axes are required for the "
            "outer momentum integral."
        )
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
    progress: bool = False,
) -> np.floating:
    """Integrate the probability over the configured total-momentum domain."""
    points, weights = _total_k_quadrature(grid)

    transverse_quadrature = _transverse_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp_value):
        return _probability_at_total_k(
            total_k_perp_value,
            packet1,
            packet2,
            impact=impact,
            transverse_quadrature=transverse_quadrature,
            prefactor=prefactor,
        )

    probability_density = np.asarray(
        sequential_map(
            evaluate,
            points,
            progress=progress,
            description="Total probability",
        ),
        dtype=REAL_DTYPE,
    )
    return np.sum(weights * probability_density)


def mean_total_momentum(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    impact=(0.0, 0.0),
    progress: bool = False,
) -> np.ndarray:
    """Return the probability-weighted ``(<Kx>, <Ky>, <Kz>)``."""
    points, weights = _total_k_quadrature(grid)
    transverse_quadrature = _transverse_quadrature(grid)
    prefactor = _probability_prefactor(packet1, packet2)

    def evaluate(total_k_perp_value):
        return _probability_at_total_k(
            total_k_perp_value,
            packet1,
            packet2,
            impact=impact,
            transverse_quadrature=transverse_quadrature,
            prefactor=prefactor,
        )

    values = np.asarray(
        sequential_map(
            evaluate,
            points,
            progress=progress,
            description="Mean momentum",
        ),
        dtype=REAL_DTYPE,
    )
    weighted_probability = weights * values
    probability = np.sum(weighted_probability)
    if not np.isfinite(probability) or probability <= 0.0:
        raise FloatingPointError("Integrated probability is not positive and finite.")

    momentum_integral = np.asarray(
        (
            np.sum(points[:, 0] * weighted_probability),
            np.sum(points[:, 1] * weighted_probability),
            (packet1.k0_z + packet2.k0_z) * probability,
        ),
        dtype=REAL_DTYPE,
    )
    return momentum_integral / probability
