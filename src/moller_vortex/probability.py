"""Vectorized phase-space probabilities built from the PDF S-matrix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PI,
    ProbabilityGrid,
    REAL_DTYPE,
    VortexPacket,
)
from .scattering import _base_prefactor, _longitudinal_from_energies, transverse_integral
from .states import central_energy


def _vector2(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=REAL_DTYPE)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector with shape (2,).")
    return value


def _worker_count(workers: int | None) -> int:
    if workers is None:
        return 1
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer or None.")
    return int(workers)


def _inner_nodes(grid: ProbabilityGrid):
    radius, radius_weight = grid.k3_perp.nodes_weights()
    phi, phi_weight = grid.k3_phi.nodes_weights()
    radius_mesh = radius[:, None]
    phi_mesh = phi[None, :]
    k3 = np.stack(
        (radius_mesh * np.cos(phi_mesh), radius_mesh * np.sin(phi_mesh)), axis=-1
    ).reshape(-1, 2)
    transverse_weight = (
        radius_weight[:, None] * phi_weight[None, :] * radius_mesh
    ).reshape(-1)
    z3, z3_weight = grid.k3_z.nodes_weights()
    z4, z4_weight = grid.k4_z.nodes_weights()
    return k3, transverse_weight, z3, z3_weight, z4, z4_weight


def _probability_constant(
    packet1: VortexPacket,
    packet2: VortexPacket,
    norms,
    mass: float,
    charge: float,
) -> float:
    """Constant after E3*E4 cancels against the final-state phase space."""
    base = _base_prefactor(packet1, packet2, norms, charge)
    phase_space = 1.0 / (4.0 * (2.0 * PI) ** 6)
    return float(
        phase_space
        * abs(base) ** 2
        / (central_energy(packet1, mass) * central_energy(packet2, mass))
    )


def _probability_at_K(
    K,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact,
    mass: float,
    charge: float,
    inner_nodes=None,
    constant: float | None = None,
) -> tuple[float, float]:
    """Return dP/d^2K_perp and its K_z-weighted inner integral."""
    K = _vector2(K, "K_perp")
    if inner_nodes is None:
        inner_nodes = _inner_nodes(grid)
    k3_perp, transverse_weight, z3, z3_weight, z4, z4_weight = inner_nodes
    k4_perp = K[None, :] - k3_perp
    T = transverse_integral(
        k3_perp,
        np.broadcast_to(K, k3_perp.shape),
        packet1,
        packet2,
        impact=impact,
    )

    K_z = z3[None, :, None] + z4[None, None, :]
    z_weight = z3_weight[None, :, None] * z4_weight[None, None, :]
    k3_perp_squared = np.sum(k3_perp * k3_perp, axis=1)
    k4_perp_squared = np.sum(k4_perp * k4_perp, axis=1)

    probability_sum = REAL_DTYPE(0.0)
    kz_sum = REAL_DTYPE(0.0)
    for start in range(0, k3_perp.shape[0], grid.batch_size):
        stop = min(start + grid.batch_size, k3_perp.shape[0])
        E3 = np.sqrt(
            mass**2
            + k3_perp_squared[start:stop, None, None]
            + z3[None, :, None] ** 2
        )
        E4 = np.sqrt(
            mass**2
            + k4_perp_squared[start:stop, None, None]
            + z4[None, None, :] ** 2
        )
        L = _longitudinal_from_energies(E3, E4, K_z, packet1, packet2, mass)
        longitudinal = np.sum(z_weight * L**2, axis=(1, 2))
        longitudinal_kz = np.sum(z_weight * K_z * L**2, axis=(1, 2))
        common = transverse_weight[start:stop] * np.abs(T[start:stop]) ** 2
        probability_sum += np.sum(common * longitudinal, dtype=REAL_DTYPE)
        kz_sum += np.sum(common * longitudinal_kz, dtype=REAL_DTYPE)

    if constant is None:
        constant = _probability_constant(packet1, packet2, norms, mass, charge)
    return float(constant * probability_sum), float(constant * kz_sum)


def differential_probability(
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> float:
    """Compute ``dP/d^2K_perp`` on the configured four-dimensional domain."""
    value, _ = _probability_at_K(
        K_perp,
        packet1,
        packet2,
        grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
    )
    return value


def differential_probability_grid(
    Kx_values,
    Ky_values,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
) -> np.ndarray:
    """Evaluate ``dP/d^2K_perp`` on a Cartesian map.

    Inner axes are vectorized and memory-batched. Independent map points are
    parallelized with ``workers``; result ordering remains deterministic.
    """
    Kx = np.asarray(Kx_values, dtype=REAL_DTYPE)
    Ky = np.asarray(Ky_values, dtype=REAL_DTYPE)
    if Kx.ndim != 1 or Ky.ndim != 1:
        raise ValueError("Kx_values and Ky_values must be one-dimensional.")
    points = [np.asarray((x, y), dtype=REAL_DTYPE) for y in Ky for x in Kx]
    inner_nodes = _inner_nodes(grid)
    constant = _probability_constant(packet1, packet2, norms, mass, charge)

    def evaluate(K):
        return _probability_at_K(
            K,
            packet1,
            packet2,
            grid,
            norms=norms,
            impact=impact,
            mass=mass,
            charge=charge,
            inner_nodes=inner_nodes,
            constant=constant,
        )[0]

    count = _worker_count(workers)
    if count == 1:
        values = [evaluate(point) for point in points]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            values = list(executor.map(evaluate, points))
    return np.asarray(values, dtype=REAL_DTYPE).reshape(Ky.size, Kx.size)


def _outer_integral(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact,
    mass: float,
    charge: float,
    workers: int | None,
) -> tuple[float, np.ndarray]:
    if grid.K_perp is None or grid.K_phi is None:
        raise ValueError("Outer axes K_perp and K_phi are required.")

    radius, radius_weight = grid.K_perp.nodes_weights()
    phi, phi_weight = grid.K_phi.nodes_weights()
    radius_mesh = radius[:, None]
    phi_mesh = phi[None, :]
    Kx = radius_mesh * np.cos(phi_mesh)
    Ky = radius_mesh * np.sin(phi_mesh)
    points = np.stack((Kx, Ky), axis=-1).reshape(-1, 2)
    weights = (
        radius_weight[:, None] * phi_weight[None, :] * radius_mesh
    ).reshape(-1)
    inner_nodes = _inner_nodes(grid)
    constant = _probability_constant(packet1, packet2, norms, mass, charge)

    def evaluate(K):
        return _probability_at_K(
            K,
            packet1,
            packet2,
            grid,
            norms=norms,
            impact=impact,
            mass=mass,
            charge=charge,
            inner_nodes=inner_nodes,
            constant=constant,
        )

    count = _worker_count(workers)
    if count == 1:
        values = [evaluate(point) for point in points]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            values = list(executor.map(evaluate, points))

    densities = np.asarray([value[0] for value in values], dtype=REAL_DTYPE)
    kz_densities = np.asarray([value[1] for value in values], dtype=REAL_DTYPE)
    total = float(np.sum(weights * densities, dtype=REAL_DTYPE))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Total probability is not positive and finite.")

    numerator = np.asarray(
        (
            np.sum(weights * points[:, 0] * densities, dtype=REAL_DTYPE),
            np.sum(weights * points[:, 1] * densities, dtype=REAL_DTYPE),
            np.sum(weights * kz_densities, dtype=REAL_DTYPE),
        ),
        dtype=REAL_DTYPE,
    )
    return total, numerator / total


def total_probability(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
    return_mean: bool = False,
):
    """Integrate over the configured K domain.

    With ``return_mean=True`` this returns ``(P, <K>)`` in one pass.
    """
    probability, mean = _outer_integral(
        packet1,
        packet2,
        grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
        workers=workers,
    )
    return (probability, mean) if return_mean else probability


def mean_total_momentum(
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
) -> np.ndarray:
    """Return the probability-weighted ``(<Kx>, <Ky>, <Kz>)``."""
    _, mean = _outer_integral(
        packet1,
        packet2,
        grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
        workers=workers,
    )
    return mean
